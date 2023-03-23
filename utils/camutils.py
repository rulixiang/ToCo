import pdb
import torch
import torch.nn.functional as F

def cam_to_label(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False, ignore_index=None):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value<=bkg_thre] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value<=high_thre] = ignore_index
        _pseudo_label[cam_value<=low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return valid_cam, pseudo_label

def cam_to_roi_mask2(cam, cls_label, hig_thre=None, low_thre=None):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _ = valid_cam.max(dim=1, keepdim=False)
    # _pseudo_label += 1
    roi_mask = torch.ones_like(cam_value, dtype=torch.int16)
    roi_mask[cam_value<=low_thre] = 0
    roi_mask[cam_value>=hig_thre] = 2

    return roi_mask

def get_valid_cam(cam, cls_label):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam

    return valid_cam

def ignore_img_box(label, img_box, ignore_index):

    pseudo_label = torch.ones_like(label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return pseudo_label

def crop_from_roi_neg(images, roi_mask=None, crop_num=8, crop_size=96):

    crops = []
    
    b, c, h, w = images.shape

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    flags = torch.ones(size=(b, crop_num+2)).to(images.device)
    margin = crop_size//2

    for i1 in range(b):
        roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] <= 1).nonzero()
        if roi_index.shape[0] < crop_num:
            roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] >= 0).nonzero() ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])
        crop_index = roi_index[rand_index[:crop_num], :]
        
        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1] # centered at (h0, w0)
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0+crop_size), w0:(w0+crop_size)]
            temp_mask = roi_mask[i1, h0:(h0+crop_size), w0:(w0+crop_size)]
            if temp_mask.sum() / (crop_size*crop_size) <= 0.2:
                ## if ratio of uncertain regions < 0.2 then negative
                flags[i1, i2+2] = 0
    
    _crops = torch.chunk(temp_crops, chunks=crop_num, dim=1,)
    crops = [c[:, 0] for c in _crops]

    return crops, flags

def multi_scale_cam2(model, inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam_aux, _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux

def label_to_aff_mask(cam_label, ignore_index=255):
    
    b,h,w = cam_label.shape

    _cam_label = cam_label.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0,2,1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)
    
    for i in range(b):
        aff_label[i, :, _cam_label_rep[i, 0, :]==ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :]==ignore_index, :] = ignore_index
    aff_label[:, range(h*w), range(h*w)] = ignore_index
    return aff_label


def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, high_thre=None, low_thre=None, ignore_index=False,  img_box=None, down_scale=2):

    b,_,h,w = images.shape
    _images = F.interpolate(images, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b,1,h,w))*high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b,1,h,w))*low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b,1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()
    
    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)

    for idx, coord in enumerate(img_box):

        valid_key = torch.nonzero(cls_labels[idx,...])[:,0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_h, valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_l, valid_key=valid_key, orig_size=(h, w))
        
        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1], coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1], coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label

def _refine_cams(ref_mod, images, cams, valid_key, orig_size):

    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label