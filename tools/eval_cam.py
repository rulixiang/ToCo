import argparse
import os
import sys

sys.path.append(".")

import imageio
import numpy as np
from datasets import voc
from tqdm import tqdm
from utils.evaluate import scores
from utils.pyutils import format_tabs

parser = argparse.ArgumentParser()
parser.add_argument("--eval_set", default='val', type=str, help="eval set")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--cam_folder", default='.', type=str, help="logit")
parser.add_argument("--data_folder", default='../VOCdevkit/VOC2012', type=str, help="dataset folder")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_thre")

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
    label_dir = os.path.join(args.data_folder, 'SegmentationClassAug')
    
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

    miou = scores(label_preds=preds, label_trues=labels)
    # print('')
    format_out = format_tabs(scores=[miou], name_list=["CAMs"], cat_list=voc.class_list)
    print(format_out)
