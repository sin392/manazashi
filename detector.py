import numpy as np
import cv2
from PIL import Image
from models import MTCNN
import torch
import sys

sys.path.append("M2Det")
from M2Det.configs.CC import Config
from M2Det.layers.functions import Detect, PriorBox
from M2Det.m2det import build_net
from M2Det.data import BaseTransform
from M2Det.utils.core import *
from M2Det.utils.nms_wrapper import nms

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
        net.eval()
        with torch.no_grad():
            priors = priorbox.forward()
            net = net.to(self.device)
            priors = priors.to(self.device)
            if self.device == "cuda":
                cudnn.benchmark = True
        init_net(net, cfg, weight, self.device)

        self.priors = priors
        self.cfg = cfg
        self._preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
        self.detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)
        self.m2det = net
        # facenet
        self.mtcnn = MTCNN(image_size=512, margin=0, thresholds=[0.55, 0.6, 0.6], keep_all=True, device=self.device)

        self.person_list = []
        self.face_list = []

    def remove_outlier(self, rects, array):
        areas = (rects[:,2] - rects[:,0]) * (rects[:,3]- rects[:,1])
        outlier_max, outlier_min = make_outlier_criteria(areas)
        outlier_idx = np.where(areas > outlier_max)[0]
        array = np.delete(array, outlier_idx, axis=0)
        return array
    
    def length_rest(self, rects, max_length=500):
        y_min = rects[:,1]
        y_max = rects[:,3]
        res = y_max - y_min
        idxs = np.where(res > max_length)[0]
        rects[idxs, 3] = rects[idxs, 1] + max_length
        return rects

    def thresh(self, probs, array, thr=0.2):
        array = array[probs > thr]
        return array

    def face_sup(self, f_rects, match_idx_list):
        idxs = np.unique(np.array([idx[0] for idx in match_idx_list]))
        f_rects = f_rects[idxs]
        self.f_rects = f_rects
        return f_rects

    def face_detect(self, img, land=False):
        img_pil = Image.fromarray(img)
        # img.to(self.device)
        if land:
            rects, probs, landmarks = self.mtcnn.detect(img_pil, landmarks=True)
        else:
            landmarks = ()
            rects, probs = self.mtcnn.detect(img_pil, landmarks=False)
        self.f_rects = rects
        self.f_num = len(rects) if len(rects) > 0 else 0
        return rects, probs, landmarks
    
    def person_detect(self, img):
        h,w = img.shape[:2]
        img = self._preprocess(img).unsqueeze(0)
        scale = torch.Tensor([w,h,w,h])
        out = self.m2det(img.to(self.device))
        rects, probs = self.detector.forward(out, self.priors)
        rects = (rects[0]*scale).cpu().numpy()
        probs = probs[0].cpu().numpy()
        allinfo = []
        # num_classes = self.cfg.model.m2det_config.num_classes
        num_classes = 2 # person only
        for j in range(1, num_classes):
            inds = np.where(probs[:,j] > self.cfg.test_cfg.score_threshold)[0]
            if len(inds) == 0:
                continue
            c_brects = rects[inds]
            c_probs = probs[inds, j]
            c_dets = np.hstack((c_brects, c_probs[:, np.newaxis])).astype(np.float32, copy=False)
            soft_nms = self.cfg.test_cfg.soft_nms
            keep = nms(c_dets, self.cfg.test_cfg.iou, force_cpu = soft_nms) #min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
            keep = keep[:self.cfg.test_cfg.keep_per_class]
            c_dets = c_dets[keep, :]
            allinfo.extend([_.tolist()+[j] for _ in c_dets])
            allinfo = np.array(allinfo)

            # 確率で足きり
            allinfo = self.thresh(allinfo[:,4], allinfo, thr=0.2)

            # 面積で大きすぎたり小さすぎたりするbboxをはじく
            allinfo = self.remove_outlier(allinfo[:,:4], allinfo)

            rects = allinfo[:,:4]
            probs = allinfo[:,4]
            cls_inds = allinfo[:,5]

            # bboxの縦方向最大長に制限
            rects = self.length_rest(rects)
            self.p_rects = rects
            self.p_num = len(rects) if len(rects) > 0 else 0
            return rects, probs, cls_inds

    def get_match_idx_list(self, f_rects, p_rects):
        match_idx_list = []
        for i in range(self.f_num):
            for j in range(self.p_num):
                iou = get_iou(f_rects[i], p_rects[j])
                half = (p_rects[j, 3] - p_rects[j, 1]) / 2
                above_hh = f_rects[i, 3] <= p_rects[j, 3] - half
                if 0.5 < iou and above_hh:
                    match_idx_list.append((i,j))
        return match_idx_list

    def get_score(self, f_rects, p_rects):
        pf_rate = self.f_num / self.p_num * 100
        return pf_rate

if __name__ == "__main__":
    img = cv2.imread("sample.png")
    cfg = Config.fromfile("M2Det/configs/m2det512_vgg.py")
    weight = "/content/drive/My Drive/m2det512_vgg.pth"
    pf_detector = PersonFaceDetector(cfg, weight)
    rects, probs, cls_inds = pf_detector.person_detect(img)
    
    result = np.hstack((rects, probs[:, np.newaxis]))
    print(result.shape)