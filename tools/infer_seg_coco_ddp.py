import argparse
import os
import sys

sys.path.append(".")

from collections import OrderedDict
import joblib
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from datasets import coco as coco
from model.model_seg_neg import network
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.dcrf import DenseCRF
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", default="workdir_coco_final/2022-10-31-14-28-55-573606/checkpoints/model_iter_80000.pth", type=str, help="model_path")
parser.add_argument("--model_path", default="workdir_coco_final/2022-11-02-22-16-01-448587/checkpoints/model_iter_80000.pth", type=str, help="model_path")
parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--img_folder", default='../coco2014', type=str, help="dataset folder")
parser.add_argument("--label_folder", default='../MSCOCO/SegmentationClass', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/coco', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument('--backend', default='nccl')
parser.add_argument("--master_port", default=29501, type=int, help="master_port")
parser.add_argument("--nproc_per_node", default=8, type=int, help="nproc_per_node")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--scales", default=(1.0, 1.25, 1.5), help="multi_scales for seg")

def _validate(pid, model=None, dataset=None, args=None):

    model.eval()
    data_loader = DataLoader(dataset[pid], batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    with torch.no_grad():
        model.cuda()

        gts, seg_pred = [], []

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            # if idx >=1000:
            #     break
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            inputs  = F.interpolate(inputs, size=[448, 448], mode='bilinear', align_corners=False)
            _, _, h, w = inputs.shape

            seg_list = []
            for sc in args.scales:
                _h, _w = int(h*sc), int(w*sc)

                _inputs  = F.interpolate(inputs, size=[_h, _w], mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs = model(inputs_cat,)[1]
                segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

                seg = torch.max(segs[:1,...], segs[1:,...].flip(-1))

                seg_list.append(seg)

            seg = torch.sum(torch.stack(seg_list, dim=0), dim=0)
            seg_pred += list(torch.argmax(seg, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            np.save(args.logits_dir + "/" + name[0] + '.npy', {"msc_seg":seg.cpu().numpy()})

    seg_score = evaluate.scores(gts, seg_pred, num_classes=81)

    print(seg_score)
    
    return seg_score


def crf_proc():
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    if "val" in args.infer_set:
        images_path = os.path.join(args.img_folder, "val2014")
        labels_path = os.path.join(args.label_folder, "val2014")
    elif "train" in args.infer_set:
        images_path = os.path.join(args.img_folder, "train2014")
        labels_path = os.path.join(args.label_folder, "train2014")

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=1,   # 3
        pos_w=1,        # 3
        bi_xy_std=121,  # 121, 140
        bi_rgb_std=5,   # 5, 5
        bi_w=4,         # 4, 5
    )

    def _job(i):

        name = name_list[i]

        logit_name = args.logits_dir + "/" + name + ".npy"

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_seg']

        image_name = os.path.join(images_path, name + ".jpg")
        image = coco.robust_read_image(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if "test" in args.infer_set:
            label = image[:,:,0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(args.segs_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label
    
    n_jobs = int(os.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds, num_classes=81)
    print(crf_score)
    return crf_score


def validate(args=None):

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    val_dataset = coco.CocoSegDataset(
        img_dir=args.img_folder,
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False,
                            drop_last=False)

    model = network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
        # pooling=args.pooling,
        aux_layer=-3,
    )


    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v
    # new_state_dict.pop("conv.weight")
    # new_state_dict.pop("aux_conv.weight")
    model.to(torch.device(args.local_rank))
    model.load_state_dict(state_dict=new_state_dict, strict=True)
    model.eval()

    model = DistributedDataParallel(model, device_ids=[args.local_rank],)

    n_gpus = dist.get_world_size()
    split_dataset = [torch.utils.data.Subset(val_dataset, np.arange(i, len(val_dataset), n_gpus)) for i in range (n_gpus)]

    # split_dataset = [torch.utils.data.Subset(val_dataset, np.arange(i, len(val_dataset), n_gpus)) for i in range (n_gpus)]

    seg_score = _validate(pid=args.local_rank, model=model, dataset=split_dataset, args=args,)
    torch.cuda.empty_cache()
    
    return True

if __name__ == "__main__":

    args = parser.parse_args()

    base_dir = args.model_path.split("checkpoints")[0]
    args.logits_dir = os.path.join(base_dir, "logits", args.infer_set)
    args.segs_dir = os.path.join(base_dir, "segs", args.infer_set)

    os.makedirs(args.segs_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)

    if args.local_rank==0:
        print(args)
    validate(args=args)
    if args.local_rank==0:
        crf_proc()