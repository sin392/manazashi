import os
import cv2
import numpy as np
import time
from torch.multiprocessing import Pool
import sys

from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import argparse

from detector import PersonFaceDetector
from M2Det.configs.CC import Config

parser = argparse.ArgumentParser(description='M2Det Testing')
parser.add_argument('-c', '--config', default='M2Det/configs/m2det512_vgg.py', type=str)
parser.add_argument('-f', '--directory', default='imgs/', help='the path to demo images')
parser.add_argument('-m', '--trained_model', default='weights/m2det512_vgg.pth', type=str, help='Trained state_dict file path to open')
parser.add_argument('--video', default=False, type=bool, help='videofile mode')
parser.add_argument('--cam', default=-1, type=int, help='camera device id')
parser.add_argument('--show', action='store_true', help='Whether to display the images')
parser.add_argument('--crop', action='store_true', help='Crop Bbox of Person Class')
args = parser.parse_args()

cfg = Config.fromfile(args.config)
detector = PersonFaceDetector(cfg, args.trained_model)


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

def draw_detection(img, rects, probs, cls_inds, match_idx_list, thr=0.2):
    imgcv = np.copy(img)
    h, w, _ = imgcv.shape
    for i, box in enumerate(rects):
        if i in match_idx_list:
            color = (0, 0, 255)
        else:
            color = (255, 176, 0)

        if probs[i] < thr:
            continue
        cls_indx = int(cls_inds[i])
        box = [int(_) for _ in box]
        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      color, thick)
    return imgcv

def crop_person(img, rect):
    rect = list(map(int, rect))
    # img = np.asarray(img)
    img_cropped = img[rect[1]:rect[3]+1, rect[0]:rect[2]+1]
    return img_cropped

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
im_fnames = sorted((fname for fname in os.listdir(im_path) if os.path.splitext(fname)[-1] == '.jpg'))
im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)
im_iter = iter(im_fnames)
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

    start = time.time()
    p_rects, p_probs, p_cls_inds = detector.person_detect(image)
    f_rects, f_probs, landmarks = detector.face_detect(image)
    end = time.time()
    loop_time = end - start

    print(loop_time)
    print(p_rects)
    print(f_rects)

    pf_rate, match_idx_list = detector.get_score(f_rects, p_rects)
    print(match_idx_list)
    im2show = draw_detection(image, p_rects, p_probs, p_cls_inds, match_idx_list=match_idx_list)
    print(pf_rate)

    cv2.putText(im2show, f'face/person : {pf_rate:.2f}%', (20, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (100, 255, 100), 5, cv2.LINE_AA)

    for i, rect in enumerate(f_rects):
        rect = tuple(map(int, rect.tolist()))
        cv2.rectangle(im2show, pt1=rect[:2], pt2=rect[2:], color=(0,255,0))
        for landmark in landmarks[i]:
            cv2.drawMarker(im2show, tuple(map(int, landmark.tolist())), (0,0,255), markerSize=10)

    if im2show.shape[0] > 1100:
        im2show = cv2.resize(im2show,
                             (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))
    if args.show:
        cv2.imshow('test', im2show)
        if cam < 0 and not video:
            cv2.waitKey(5000)
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                out_video.release()
                capture.release()
                break
    if cam < 0 and not video:
        cv2.imwrite('{}_m2det_facenet.jpg'.format(fname.split('.')[0]), im2show)
    else:
        out_video.write(im2show)