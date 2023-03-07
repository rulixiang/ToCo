import argparse
import os
import sys

sys.path.append(".")

import imageio
import numpy as np
from datasets import coco as coco
from tqdm import tqdm
from utils.evaluate import scores
from utils.pyutils import format_tabs

parser = argparse.ArgumentParser()
parser.add_argument("--eval_set", default='val_part', type=str, help="eval set")
parser.add_argument("--cam_folder", default='', type=str, help="eval folder")
parser.add_argument("--data_folder", default='../../VOCdevkit/VOC2012', type=str, help="dataset folder")
parser.add_argument("--bkg_thre", default=0.4, type=float, help="bkg_thre")
parser.add_argument("--img_folder", default='../../coco2014', type=str, help="dataset folder")
parser.add_argument("--label_folder", default='../../MSCOCO/SegmentationClass', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/coco', type=str, help="train/val/test list file")
args = parser.parse_args()

def load_txt(txt_name):
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]
        return name_list

if __name__=="__main__":
    # config = OmegaConf.load(args.config)
    print(args)
    print('Evaluating:')
    split_path = os.path.join(args.list_folder, args.eval_set + ".txt")
    eval_list = load_txt(split_path)
    npy_dir = args.cam_folder

    if "val" in args.eval_set:
        label_dir = os.path.join(args.label_folder, "val2014")
    elif "train" in args.eval_set:
        label_dir = os.path.join(args.label_folder, "train2014")
    
    preds = []
    labels = []

    for i in tqdm(eval_list, total=len(eval_list), ncols=100,):
        npy_name = os.path.join(npy_dir, i) + '.npy'
        cam_dict = np.load(npy_name, allow_pickle=True).item()
        label = imageio.imread(os.path.join(label_dir, i) + '.png')

        cams = cam_dict['high_res']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.bkg_thre)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
        cls_labels = np.argmax(cams, axis=0)
        cls_labels = keys[cls_labels]

        preds.append(cls_labels.copy())
        labels.append(label)

    miou = scores(label_preds=preds, label_trues=labels, num_classes=81)
    # print('')
    format_out = format_tabs(scores=[miou], name_list=["CAMs"], cat_list=coco.class_list)
    print(format_out)
