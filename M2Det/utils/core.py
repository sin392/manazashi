import time,os
import torch
import shutil
import argparse
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from layers.functions import PriorBox
from layers.modules import MultiBoxLoss
from data import mk_anchors
# from data import COCODetection, VOCDetection, detection_collate, preproc
from configs.CC import Config
from utils.nms_wrapper import nms
import numpy as np

def anchors(cfg):
    return mk_anchors(cfg.model.input_size,
                               cfg.model.input_size,
                               cfg.model.anchor_config.size_pattern, 
                               cfg.model.anchor_config.step_pattern)

def init_net(net, cfg, resume_net, device):    
    if cfg.model.init_net and not resume_net:
        net.init_model(cfg.model.pretrained)
    else:
        # print('Loading resume network...')
        state_dict = torch.load(resume_net, map_location=device)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict,strict=False)


def save_checkpoint(net, cfg, final=True, datasetname='COCO',epoch=10):
    if final:
        torch.save(net.state_dict(), cfg.model.weights_save + \
                'Final_M2Det_{}_size{}_net{}.pth'.format(datasetname, cfg.model.input_size, cfg.model.m2det_config.backbone))
    else:
        torch.save(net.state_dict(), cfg.model.weights_save + \
                'M2Det_{}_size{}_net{}_epoch{}.pth'.format(datasetname, cfg.model.input_size, cfg.model.m2det_config.backbone,epoch))

def image_forward(img, net, cuda, priors, detector, transform):
    w,h = img.shape[1],img.shape[0]
    scale = torch.Tensor([w,h,w,h])
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if cuda:
            x = x.cuda()
            scale = scale.cuda()
    out = net(x)
    boxes, scores = detector.forward(out, priors)
    boxes = (boxes[0] * scale).cpu().numpy()
    scores = scores[0].cpu().numpy()
    return boxes, scores
   
def nms_process(num_classes, i, scores, boxes, cfg, min_thresh, all_boxes, max_per_image):
    for j in range(1, num_classes): # ignore the bg(category_id=0)
        inds = np.where(scores[:,j] > min_thresh)[0]
        if len(inds) == 0:
            all_boxes[j][i] = np.empty([0,5], dtype=np.float32)
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)

        soft_nms = cfg.test_cfg.soft_nms
        keep = nms(c_dets, cfg.test_cfg.iou, force_cpu=soft_nms)
        keep = keep[:cfg.test_cfg.keep_per_class] # keep only the highest boxes
        c_dets = c_dets[keep, :]
        all_boxes[j][i] = c_dets
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]


