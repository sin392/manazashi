import os
import cv2
import numpy as np
import time
from torch.multiprocessing import Pool
import sys

sys.path.append("M2Det")
from utils.nms_wrapper import nms
from utils.timer import Timer
from configs.CC import Config
import argparse
from layers.functions import Detect, PriorBox
from m2det import build_net
from data import BaseTransform
from utils.core import *
from utils.pycocotools.coco import COCO

from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser(description='M2Det Testing')
parser.add_argument('-c', '--config', default='M2Det/configs/m2det512_vgg.py', type=str)
parser.add_argument('-f', '--directory', default='imgs/', help='the path to demo images')
parser.add_argument('-m', '--trained_model', default='weights/m2det512_vgg.pth', type=str, help='Trained state_dict file path to open')
parser.add_argument('--video', default=False, type=bool, help='videofile mode')
parser.add_argument('--cam', default=-1, type=int, help='camera device id')
parser.add_argument('--show', action='store_true', help='Whether to display the images')
parser.add_argument('--crop', action='store_true', help='Crop Bbox of Person Class')
args = parser.parse_args()

print_info(' ----------------------------------------------------------------------\n'
           '|                       M2Det Demo Program                             |\n'
           ' ----------------------------------------------------------------------', ['yellow','bold'])

global cfg
cfg = Config.fromfile(args.config)
anchor_config = anchors(cfg)
print_info('The Anchor info: \n{}'.format(anchor_config))
priorbox = PriorBox(anchor_config)

net = build_net('test',
                size = cfg.model.input_size,
                config = cfg.model.m2det_config)
init_net(net, cfg, args.trained_model)
print_info('===> Finished constructing and loading model',['yellow','bold'])
net.eval()
with torch.no_grad():
    priors = priorbox.forward()
    if cfg.test_cfg.cuda:
        print("Device : cuda")
        net = net.cuda()
        priors = priors.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
_preprocess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))
detector = Detect(cfg.model.m2det_config.num_classes, cfg.loss.bkg_label, anchor_config)

def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127
base = int(np.ceil(pow(cfg.model.m2det_config.num_classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(cfg.model.m2det_config.num_classes)]
cats = [_.strip().split(',')[-1] for _ in open('M2Det/data/coco_labels.txt','r').readlines()]
labels = tuple(['__background__'] + cats)

def draw_detection(im, bboxes, scores, cls_inds, fps,  match_idx_list, thr=0.2):
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape
    for i, box in enumerate(bboxes):
        if i in match_idx_list:
            color = (0, 0, 255)
        else:
            color = (255, 176, 0)

        if scores[i] < thr:
            continue
        cls_indx = int(cls_inds[i])
        box = [int(_) for _ in box]
        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      color, thick)
        # mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        mess = '%.3f' % (scores[i])
        cv2.putText(imgcv, mess, (box[0], box[1] - 7),
                    0, 1e-3 * h, color, thick // 5 + 1 , lineType = cv2.LINE_AA)
        if fps >= 0:
            cv2.putText(imgcv, '%.2f' % fps + ' fps', (w - 160, h - 15), 0, 2e-3 * h, (255, 255, 255), thick // 2)

    return imgcv

def make_outlier_criteria(col):
    #arrayに対して標準偏差と平均を算出。
    average = np.mean(col)
    sd = np.std(col)
    #2σよりのデータ位置を指定。
    outlier_min = average - (sd) * 2
    outlier_max = average + (sd) * 2
    return outlier_max,outlier_min 

def get_iou(a, b):
    # get area of a
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    # get area of b
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    # get left top x of IoU
    iou_x1 = np.maximum(a[0], b[0])
    # get left top y of IoU
    iou_y1 = np.maximum(a[1], b[1])
    # get right bottom of IoU
    iou_x2 = np.minimum(a[2], b[2])
    # get right bottom of IoU
    iou_y2 = np.minimum(a[3], b[3])

    # get width of IoU
    iou_w = iou_x2 - iou_x1
    # get height of IoU
    iou_h = iou_y2 - iou_y1

    # no overlap
    if iou_w < 0 or iou_h < 0:
        return 0.0

    # get area of IoU
    area_iou = iou_w * iou_h
    # get overlap ratio between IoU and all area
    # iou = area_iou / (area_a + area_b - area_iou)
    iou = area_iou / area_a

    return iou

def crop_person(image, rect):
    # 負値が出る場合があり、挙動がおかしくなるのでカット
    rect = [x if x >= 0 else 0 for x in list(map(int, rect))]
    im = np.asarray(image)
    im = im[rect[1]:rect[3]+1, rect[0]:rect[2]+1]
    return im

im_path = args.directory
cam = args.cam
video = args.video
if cam >= 0:
    capture = cv2.VideoCapture(cam)
    video_path = './cam'
if video:
    while True:
        video_path = input('Please enter video path: ')
        capture = cv2.VideoCapture(video_path)
        if capture.isOpened():
            break
        else:
            print('No file!')
if cam >= 0 or video:
    video_name = os.path.splitext(video_path)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out_video = cv2.VideoWriter(video_name[0] + '_m2det.mp4', fourcc, capture.get(cv2.CAP_PROP_FPS), (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
im_fnames = sorted((fname for fname in os.listdir(im_path) if os.path.splitext(fname)[-1] == '.jpg' or os.path.splitext(fname)[-1] == '.png'))

im_fnames = (os.path.join(im_path, fname) for fname in im_fnames[451:])

im_iter = iter(im_fnames)

# print(list(im_fnames))

while True:
    if cam < 0 and not video:
        try:
            fname = next(im_iter)
        except StopIteration:
            break
        if 'm2det' in fname: continue # ignore the detected images
        image = cv2.imread(fname, cv2.IMREAD_COLOR)

    else:
        ret, image = capture.read()
        if not ret:
            cv2.destroyAllWindows()
            capture.release()
            break
    print(fname)
    loop_start = time.time()
    w,h = image.shape[1],image.shape[0]
    img = _preprocess(image).unsqueeze(0)
    if cfg.test_cfg.cuda:
        img = img.cuda()
    scale = torch.Tensor([w,h,w,h])
    inf_start = time.time()
    out = net(img)
    inf_end = time.time()
    print("Inference Time :", inf_end - inf_start)

    boxes, scores = detector.forward(out, priors)
    boxes = (boxes[0]*scale).cpu().numpy()
    scores = scores[0].cpu().numpy()
    allboxes = []
    for j in range(1, cfg.model.m2det_config.num_classes):
        inds = np.where(scores[:,j] > cfg.test_cfg.score_threshold)[0]
        if len(inds) == 0:
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
        soft_nms = cfg.test_cfg.soft_nms
        keep = nms(c_dets, cfg.test_cfg.iou, force_cpu = soft_nms) #min_thresh, device_id=0 if cfg.test_cfg.cuda else None)
        keep = keep[:cfg.test_cfg.keep_per_class]
        c_dets = c_dets[keep, :]
        allboxes.extend([_.tolist()+[j] for _ in c_dets])

    loop_time = time.time() - loop_start
    print("Time :", loop_time)
    allboxes = np.array(allboxes)
    allboxes = allboxes[allboxes[:, 5] == 1]

    # 確率でthresh →　もとは描画側でやってた
    thr = 0.2
    print(len(allboxes))
    allboxes = allboxes[allboxes[:,4] > thr]
    print(len(allboxes))

    # 面積で大きすぎるbbox弾きたい
    areas = (allboxes[:,2] - allboxes[:,0]) * (allboxes[:,3]- allboxes[:,1])
    outlier_max, outlier_min = make_outlier_criteria(areas)
    outlier_idx = np.where(areas > outlier_max)[0]

    N = len(outlier_idx)
    print(f"{N} outlier found")
    print(outlier_idx)
    
    allboxes = np.delete(allboxes, outlier_idx, axis=0)

    boxes = allboxes[:,:4]
    scores = allboxes[:,4]
    cls_inds = allboxes[:,5]


    print('\n'.join(['pos:{}, ids:{}, score:{:.3f}'.format('(%.1f,%.1f,%.1f,%.1f)' % (o[0],o[1],o[2],o[3]) \
            ,labels[int(oo)],ooo) for o,oo,ooo in zip(boxes,cls_inds,scores)]))
    fps = 1.0 / float(loop_time) if cam >= 0 or video else -1

    ##############################################################################
    im = Image.open(fname)

    if args.crop == True:
        if not os.path.exists('cropped'):
            os.mkdir('cropped')
        for i in range(len(boxes)):
            cropped_im = crop_person(im, boxes[i])
            try:
                cropped_im = cv2.cvtColor(cropped_im, cv2.COLOR_RGB2BGR)
            except:
                print(boxes[i])
                print(im.size)
                print(cropped_im.shape)
                
            fname_c = os.path.basename(fname.split(".")[0])
            cv2.imwrite(f"cropped/{fname_c}_person_{i}.jpg", cropped_im)