
import argparse
import os
import sys

import joblib

sys.path.append(".")


import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch import multiprocessing
from utils.dcrf import DenseCRF
from utils.evaluate import scores
from utils.imutils import encode_cmap
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default='../VOCdevkit/VOC2012', type=str, help="root_dir")
parser.add_argument("--txt_dir", default='datasets/voc/', type=str, help="txt_dir")
parser.add_argument("--eval_set", default='val', type=str, help="eval_set")
parser.add_argument("--cam_path", default='.', type=str, help="logit")
parser.add_argument("--dst_dir", default='cam_label_crf2', type=str, help="dst")
parser.add_argument("--bkg_thre_h", default=0.7, type=float, help='bkg_thre')
parser.add_argument("--bkg_thre_l", default=0.25, type=float, help='bkg_thre')
parser.add_argument("--save_label", action="store_true", default=False, help='save crf label')

def crf_proc(args):
    print("crf post-processing...")

    txt_name = os.path.join(args.txt_dir, args.eval_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(args.root_dir, 'JPEGImages',)
    labels_path = os.path.join(args.root_dir, 'SegmentationClassAug')
    cam_path = args.cam_path

    args.dst_dir = args.cam_path.replace("cams", args.dst_dir)
    crf_pred_path = os.path.join(args.dst_dir, args.eval_set, 'pred')
    crf_pred_rgb_path = os.path.join(args.dst_dir, args.eval_set, 'pred_rgb')

    os.makedirs(crf_pred_path, exist_ok=True)
    os.makedirs(crf_pred_rgb_path, exist_ok=True)

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=3,   # 3
        pos_w=3,        # 3
        bi_xy_std=36,  # 121, 140 ok
        bi_rgb_std=3,   # 5, 5
        bi_w=4,         # 4, 5
    )

    def _job(i):

        name = name_list[i]
        ##
        cam_name = os.path.join(cam_path, name + ".npy")
        cam_dict = np.load(cam_name, allow_pickle=True).item()
        logit = cam_dict['high_res']

        logit_l = np.zeros((1, logit.shape[0]+1, logit.shape[1], logit.shape[2]), dtype=np.float32)
        logit_h = logit_l.copy()

        logit_l[0,0,:,:] = args.bkg_thre_l
        logit_l[0,1:,:,:] = logit

        logit_h[0,0,:,:] = args.bkg_thre_h
        logit_h[0,1:,:,:] = logit
        ##

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        if args.eval_set == "test":
            label = image[:, :, 0]
        else:
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        image = image.astype(np.uint8)

        logit_h = torch.from_numpy(logit_h)#[None, ...]
        logit_h = F.interpolate(logit_h, size=(H, W), mode="bilinear", align_corners=False)
        prob_h = F.softmax(logit_h, dim=1)[0].numpy()

        logit_l = torch.from_numpy(logit_l)#[None, ...]
        logit_l = F.interpolate(logit_l, size=(H, W), mode="bilinear", align_corners=False)
        prob_l = F.softmax(logit_l, dim=1)[0].numpy()

        prob_h = post_processor(image, prob_h)
        prob_l = post_processor(image, prob_l)
        #####
        keys = np.zeros(shape=[cam_dict['keys'].shape[0]+1])
        keys[1:] = cam_dict['keys']+1
        #####
        pred_h = np.argmax(prob_h, axis=0)
        pred_h = keys[pred_h]
        pred_l = np.argmax(prob_l, axis=0)
        pred_l = keys[pred_l]
        #####
        pred = pred_h.copy()
        pred[pred_h == 0] = 255
        pred[(pred_h + pred_l) == 0] = 0
        ####

        _pred = np.squeeze(pred).astype(np.uint8)
        _pred_cmap = encode_cmap(_pred)

        if args.save_label:
            imageio.imsave(crf_pred_path+'/'+name+'.png', _pred)
            imageio.imsave(crf_pred_rgb_path+'/'+name+'.png', _pred_cmap)

        return _pred, label

    n_jobs = int(multiprocessing.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)
    score = scores(gts, preds)
        
    print('Prediction results saved to %s.'%(crf_pred_path))
    print('Pixel acc is %f, mean IoU is %f.'%(score['pAcc'], score['miou']))
    
    return True

if __name__=="__main__":
    args = parser.parse_args()
    print(args)
    crf_score = crf_proc(args)
