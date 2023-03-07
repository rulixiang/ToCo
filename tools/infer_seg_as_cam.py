import argparse
import os
import sys

sys.path.append(".")

from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import voc
from model.model_seg_neg import network
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.camutils import get_valid_cam

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="./", type=str, help="model_path")
parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--data_folder", default='../VOCdevkit/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--cam_scales", default=(1.0, 1.25, 1.5), help="multi_scales for cam")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')

def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)

def _validate(pid, model=None, dataset=None, args=None):

    model.eval()
    data_loader = DataLoader(dataset[pid], batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    base_dir = args.model_path.split("checkpoint")[0]
    cam_dir = os.path.join(base_dir, "cams")
    os.makedirs(cam_dir, exist_ok=True)

    with torch.no_grad():
        model.cuda()

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()

            _, _, h, w = inputs.shape

            labels = labels.cuda()
            cls_label = cls_label.cuda()

            seg_list = []
            for sc in args.cam_scales:
                _h, _w = int(h*sc), int(w*sc)

                _inputs  = F.interpolate(inputs, size=[_h, _w], mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs = model(inputs_cat,)[1]
                segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

                seg = torch.max(segs[:1,...], segs[1:,...].flip(-1))

                seg_list.append(seg)

            seg = torch.sum(torch.stack(seg_list, dim=0), dim=0).softmax(dim=1)

            _cams = get_valid_cam(seg[:, 1:, ...], cls_label)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            resized_cam = get_valid_cam(resized_cam, cls_label)

            valid_label = torch.nonzero(cls_label[0])[:,0]

            valid_high_res = resized_cam[0, valid_label]
            npy_name = os.path.join(cam_dir, name[0] + '.npy')

            np.save(npy_name, {"keys": valid_label.cpu().numpy(), "high_res": valid_high_res.cpu().numpy()})

    return None


def validate():

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    model = network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
        aux_layer=-3
    )
    model.to(torch.device(args.local_rank))
    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v

    model.load_state_dict(state_dict=new_state_dict, strict=True)
    model.eval()
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    n_gpus = dist.get_world_size()
    split_dataset = [torch.utils.data.Subset(val_dataset, np.arange(i, len(val_dataset), n_gpus)) for i in range (n_gpus)]

    _validate(args.local_rank, model=model, dataset=split_dataset, args=args)

    torch.cuda.empty_cache()
    return True


if __name__ == "__main__":

    args = parser.parse_args()
    if args.local_rank == 0:
        print(args)
    validate()

