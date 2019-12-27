import numpy as np
import cv2
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
from torch.multiprocessing import Pool
import sys

sys.path.append("./M2Det")
from utils.timer import Timer
from configs.CC import Config
from layers.functions import Detect, PriorBox
from m2det import build_net
from data import BaseTransform
from utils.core import *
from utils.pycocotools.coco import COCO
from utils.nms_wrapper import nms

def make_outlier_criteria(col):
    #arrayに対して標準偏差と平均を算出。
    average = np.mean(col)
    sd = np.std(col)
    #2σよりのデータ位置を指定。
    outlier_min = average - (sd) * 2
    outlier_max = average + (sd) * 2
    return outlier_max,outlier_min 

def get_iou(a, b):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    iou_x1 = np.maximum(a[0], b[0])
    iou_y1 = np.maximum(a[1], b[1])
    iou_x2 = np.minimum(a[2], b[2])
    iou_y2 = np.minimum(a[3], b[3])

    iou_w = iou_x2 - iou_x1
    iou_h = iou_y2 - iou_y1

    # no overlap
    if iou_w < 0 or iou_h < 0:
        return 0.0

    area_iou = iou_w * iou_h
    # get overlap ratio between IoU and all area
    # iou = area_iou / (area_a + area_b - area_iou)
    iou = area_iou / area_a
    return iou

class PersonFaceDetector():
    def __init__(self, cfg, weight):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("using :", self.device)
        # m2det
        anchor_config = anchors(cfg)
        priorbox = PriorBox(anchor_config)
        net = build_net('test',
                    size = cfg.model.input_size,
                    config = cfg.model.m2det_config)
        init_net(net, cfg, weight)
        net.eval()
        with torch.no_grad():
            priors = priorbox.forward()
            if self.device == "cuda":
                net = net.cuda()
                priors = priors.cuda()
                cudnn.benchmark = True
            else:
                net = net.cpu()
        self.priors = priors
        self.cfg = cfg
        self._preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
        self.detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)
        self.m2det = net
        # facenet
        self.mtcnn = MTCNN(image_size=512, margin=0, keep_all=True, device=self.device)

    def remove_outlier(self, rects, array):
        areas = (rects[:,2] - rects[:,0]) * (rects[:,3]- rects[:,1])
        outlier_max, outlier_min = make_outlier_criteria(areas)
        outlier_idx = np.where(areas > outlier_max)[0]
        array = np.delete(array, outlier_idx, axis=0)
        return array
    
    def thresh(self, scores, array, thr=0.2):
        array = array[scores > thr]
        return array

    def face_detect(self, img, land=False):
        img_pil = Image.fromarray(img)
        # img.to(self.device)
        if land:
            face_rects, probs, landmarks = mtcnn.detect(img_pil, landmarks=True)
        else:
            landmarks = ()
            face_rects, probs = mtcnn.detect(img_pil, landmarks=False)
        return face_rects, probs, landmarks
    
    def person_detect(self, img):
        h,w = img.shape[:2]
        img = self._preprocess(img).unsqueeze(0)
        scale = torch.Tensor([w,h,w,h])
        out = self.m2det(img.to(self.device))
        boxes, scores = self.detector.forward(out, self.priors)
        boxes = (boxes[0]*scale).cpu().numpy()
        scores = scores[0].cpu().numpy()
        allboxes = []
        for j in range(1, self.cfg.model.m2det_config.num_classes):
            inds = np.where(scores[:,j] > self.cfg.test_cfg.score_threshold)[0]
            if len(inds) == 0:
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            soft_nms = self.cfg.test_cfg.soft_nms
            keep = nms(c_dets, self.cfg.test_cfg.iou, force_cpu = soft_nms) #min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
            keep = keep[:self.cfg.test_cfg.keep_per_class]
            c_dets = c_dets[keep, :]
            allboxes.extend([_.tolist()+[j] for _ in c_dets])
            allboxes = np.array(allboxes)

            # 確率で足きり
            allboxes = self.thresh(allboxes[:,4], allboxes, thr=0.2)

            # 面積で大きすぎたり小さすぎたりするbboxをはじく
            print(allboxes)
            allboxes = self.remove_outlier(allboxes[:,:4], allboxes)

            boxes = allboxes[:,:4]
            scores = allboxes[:,4]
            cls_inds = allboxes[:,5]

            return boxes, scores, cls_inds

if __name__ == "__main__":
    img = cv2.imread("sample.png")
    cfg = Config.fromfile("M2Det/configs/m2det512_vgg.py")
    weight = "/content/drive/My Drive/m2det512_vgg.pth"
    pf_detector = PersonFaceDetector(cfg, weight)
    boxes, scores, cls_inds = pf_detector.person_detect(img)
    
    result = np.hstack((boxes, scores[:, np.newaxis]))
    print(result.shape)