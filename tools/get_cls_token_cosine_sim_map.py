import argparse
import os
import pdb
import sys

sys.path.append(".")
from collections import OrderedDict

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import voc
from model.model_seg_neg2 import network
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.camutils import cam_to_label, get_valid_cam, multi_scale_cam2, cam_to_roi_mask2
# from utils.pyutils import AverageMeter, format_tabs

parser = argparse.ArgumentParser()
parser.add_argument("--eval_set", default="train", type=str, help="eval_set")
parser.add_argument("--bkg_score", default=0.5, type=float, help="work_dir")
parser.add_argument("--alpha", default=0.5, type=float, help="alpha")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--model_path", default="workdir_voc_final/2022-10-31-08-22-37-370565/checkpoints/model_iter_20000.pth", type=str, help="model_path")

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--data_folder", default='../VOCdevkit/VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")

def crop_from_roi4(images, roi_mask=None, crop_num=8, crop_size=96):

    crops = []
    b, c, h, w = images.shape

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    margin = crop_size//2

    for i1 in range(b):
        # print(roi_mask.shape)
        roi_index = (roi_mask[:, margin:(h-margin), margin:(w-margin)] == 1).nonzero()
        if roi_index.shape[0]<crop_num:
            # print(roi_index.shape[0])
            roi_index = (roi_mask[:, margin:(h-margin), margin:(w-margin)] >= 0).nonzero() ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])
        crop_index = roi_index[rand_index[:crop_num], :]
        # print(crop_index)
        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 1], crop_index[i2, 2] # centered at (h0, w0)
            # print(crop_index[i2])
            # print(images.shape)
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0+crop_size), w0:(w0+crop_size)]

    
    _crops = torch.chunk(temp_crops, chunks=crop_num, dim=1,)
    crops = [c[:, 0] for c in _crops]

    return crops

def _validate(model=None, data_loader=None, args=None):

    model.eval()

    base_dir = args.model_path.split("checkpoints")[0]
    img_dir_l = os.path.join(base_dir, "cls_token_maps/img_l")
    img_dir_g = os.path.join(base_dir, "cls_token_maps/img_g")
    sal_dir_l = os.path.join(base_dir, "cls_token_maps/sal_l")
    sal_dir_g = os.path.join(base_dir, "cls_token_maps/sal_g")

    os.makedirs(img_dir_l, exist_ok=True)
    os.makedirs(sal_dir_l, exist_ok=True)
    os.makedirs(img_dir_g, exist_ok=True)
    os.makedirs(sal_dir_g, exist_ok=True)
    color_map = plt.get_cmap("magma")

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()

        gts, cams, aux_cams = [], [], []

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            # if idx >=100:
            #     break

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            

            inputs  = F.interpolate(inputs, size=[448, 448], mode='bilinear', align_corners=False)
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            ###
            _cams, _cams_aux = multi_scale_cam2(model, inputs, [1.0,])
            _cams_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            roi_mask = cam_to_roi_mask2(_cams_aux.detach(), cls_label=cls_label, low_thre=args.low_thre, hig_thre=args.high_thre)
            local_crops = crop_from_roi4(images=inputs, roi_mask=roi_mask, crop_num=1, crop_size=96)[0]

            img_l = imutils.denormalize_img(local_crops)[0].permute(1,2,0).cpu().numpy()
            imageio.imsave(os.path.join(img_dir_l, name[0] + ".jpg"), img_l.astype(np.uint8))
            img_g = imutils.denormalize_img(inputs)[0].permute(1,2,0).cpu().numpy()
            imageio.imsave(os.path.join(img_dir_g, name[0] + ".jpg"), img_g.astype(np.uint8))

            _, _, patch_token, _, attn, cls_token = model(inputs)
            ###
            # patch_token = F.normalize(patch_token, dim=1, p=2)
            # cls_token = F.normalize(cls_token, dim=1, p=2)
            # cls_token = cls_token.unsqueeze(1)
            # sal_g = torch.einsum("abc,acde->abde", cls_token, patch_token)
            # sal_g = torch.abs(sal_g)
            ###
            # pdb.set_trace()
            sal_g = attn[:, :, 0, 1:].reshape(1, 12, 28, 28)
            # pdb.set_trace()
            sal_g = F.interpolate(sal_g, size=inputs.shape[2:], mode="bilinear", align_corners=False)[0,].cpu().numpy()
            sal_g = sal_g.mean(0)
            sal_g -= sal_g.min()
            sal_g /= sal_g.max()
            # sal_g += 0.1
            # sal_g[sal_g>=1]=1
            # sal_g = sal_g ** (0.5)
            sal_g = color_map(sal_g)[:,:,:3] * 255
            # sal_g = np.stack((sal_g, sal_g, sal_g), axis=-1)
            # sal_g = (sal_g) * img_g
            # pdb.set_trace()
            imageio.imsave(os.path.join(sal_dir_g, name[0] + ".jpg"), sal_g.astype(np.uint8))

            # sal_l = model(F.interpolate(local_crops, scale_factor=2, mode="bilinear", align_corners=False))[-1]

            _, _, patch_token, _, attn, cls_token = model(F.interpolate(local_crops, size=(448, 448), mode="bilinear", align_corners=False))

            sal_l = attn[:, :, 0, 1:].reshape(1, 12, 28, 28)
            sal_l = F.interpolate(sal_l, size=[96,96], mode="bilinear", align_corners=False)[0,].cpu().numpy()
            sal_l = sal_l.mean(0)
            sal_l -= sal_l.min()
            sal_l /= sal_l.max()
            # sal_l += 0.1
            # sal_l[sal_l>=1]=1
            sal_l = sal_l ** (0.5)
            sal_l = color_map(sal_l)[:,:,:3] * 255
            # sal_l = np.stack((sal_l, sal_l, sal_l), axis=-1)
            # sal_l = (sal_l) * img_l
            imageio.imsave(os.path.join(sal_dir_l, name[0] + ".jpg"), sal_l.astype(np.uint8))

    return None


def validate(args=None):

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.eval_set,
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

    model.load_state_dict(state_dict=new_state_dict, strict=True)
    model.eval()

    results = _validate(model=model, data_loader=val_loader, args=args)
    torch.cuda.empty_cache()

    print(results)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    print(args)

    validate(args=args)

