import os
import cv2
import numpy as np
import time
from torch.multiprocessing import Pool
import sys

import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import argparse

from detector import PersonFaceDetector
from M2Det.configs.CC import Config
from M2Det.utils.nms.py_cpu_nms import py_cpu_nms

def draw_detection(img, rects, match_idx_list=()):
    imgcv = np.copy(img)
    h, w, _ = imgcv.shape
    for i, box in enumerate(rects):
        color = (255, 176, 0)
        if len(match_idx_list) != 0:
            if i in match_idx_list:
                color = (0, 0, 255)

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

class DataHandler():
    def __init__(self, im_path, cam, video, show):
        self.im_path = im_path
        self.cam = cam
        self.video = video
        self.show = show
        if cam >= 0:
            capture = cv2.VideoCapture(cam)
            print("camera", capture.isOpened())
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
            self.out_video = cv2.VideoWriter(video_name[0] + '_processed.mp4', fourcc, capture.get(cv2.CAP_PROP_FPS), (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            self.capture = capture
        if os.path.isdir(im_path):
            im_fnames = sorted((fname for fname in os.listdir(im_path) if os.path.splitext(fname)[-1] in [".jpg", ".png"]))
            im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)
            self.im_iter = iter(im_fnames)

    def get_img(self):
        # -1: break, 0: sucess, 1: continue
        state = 0
        if self.cam < 0 and not self.video:
            if os.path.isdir(self.im_path):
                try:
                    fname = next(self.im_iter)
                except StopIteration:
                    state = -1
                if 'processed' in fname:
                    state = 1 # ignore the detected images
            else:
                fname = self.im_path
            self.fname = fname
            img = cv2.imread(fname, cv2.IMREAD_COLOR)
        else:
            ret, img = self.capture.read()
            if not ret:
                state = -1
                cv2.destroyAllWindows()
                self.capture.release()
        return img, state

    def out(self, img):
        state = 0
        if img.shape[0] > 1100:
            img = cv2.resize(img, (int(500. * float(img.shape[1]) / img.shape[0]), 500))
        if self.show:
            cv2.imshow('test', img)
            if self.cam < 0 and not self.video:
                cv2.waitKey(0)
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    self.out_video.release()
                    self.capture.release()
                    state = -1
        if self.cam < 0 and not self.video:
            cv2.imwrite('{}_processed.jpg'.format(self.fname.split('.')[0]), img)
            if not os.path.isdir(self.im_path):
                state = -1
        else:
            self.out_video.write(img)
        
        return state

def get_fixed_rects(imgs):
    p_rects_probs = np.empty((0, 5))
    f_rects_probs = np.empty((0, 5))
    for i, img in enumerate(imgs):
        f_rects, f_probs, landmarks = detector.face_detect(img)
        p_rects, p_probs, _ = detector.person_detect(img)
        f_rects_probs = np.concatenate((f_rects_probs, np.hstack((f_rects, f_probs[:, np.newaxis]))), axis=0)
        p_rects_probs = np.concatenate((p_rects_probs, np.hstack((p_rects, p_probs[:, np.newaxis]))), axis=0)
    f_keep = py_cpu_nms(f_rects_probs, 0.5)
    p_keep = py_cpu_nms(p_rects_probs, 0.5)
    f_rects = f_rects_probs[f_keep, :4]
    p_rects = p_rects_probs[p_keep, :4]
    return f_rects, p_rects

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='M2Det Testing')
    parser.add_argument('-c', '--config', default='M2Det/configs/m2det512_vgg.py', type=str)
    parser.add_argument('-f', '--directory', default='sample.png', help='the path to demo images')
    parser.add_argument('-m', '--trained_model', default='weights/m2det512_vgg.pth', type=str, help='Trained state_dict file path to open')
    parser.add_argument('--video', default=False, type=bool, help='videofile mode')
    parser.add_argument('--cam', default=-1, type=int, help='camera device id')
    parser.add_argument('--show', action='store_true', help='Whether to display the images')
    parser.add_argument('--crop', action='store_true', help='Crop Bbox of Person Class')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    detector = PersonFaceDetector(cfg, args.trained_model)

    im_path = args.directory
    cam = args.cam
    video = args.video

    handler = DataHandler(im_path, cam, video, args.show)

    count = 0
    while True:
        img, state = handler.get_img()
        if state == -1:
            print("break")
            break
        elif state == 1:
            print("continue")
            continue
        
        imgs = []
        if count < 5:
            imgs.append(img)
            if count == 4:
                f_rects, p_rects = get_fixed_rects(imgs)
                fixed_f_rects, fixed_p_rects = f_rects, p_rects
                landmarks = ()
                count += 1
            else:
                count += 1
                continue

        else:
            # img = cv2.resize(img, (512, 512))
            start = time.time()
            p_rects, p_probs, _ = detector.person_detect(img)
            print(time.time() - start)
            f_rects, f_probs, landmarks = detector.face_detect(img)
            end = time.time()
            loop_time = end - start

            print("loop_time", loop_time)
            # print("p_rects", p_rects)
            # print("f_rects", f_rects)
            count += 1

        pf_rate, match_idx_list = detector.get_score(f_rects, p_rects)
            # print(match_idx_list)
        im2show = draw_detection(img, p_rects, match_idx_list=match_idx_list)
        print("pfrate", pf_rate)

        cv2.putText(im2show, f'face/person : {pf_rate:.2f}%', (20, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (100, 255, 100), 5, cv2.LINE_AA)

        for i, rect in enumerate(f_rects):
            rect = tuple(map(int, rect.tolist()))
            cv2.rectangle(im2show, pt1=rect[:2], pt2=rect[2:], color=(0,255,0))
            if len(landmarks) > 0:
                for landmark in landmarks[i]:
                    cv2.drawMarker(im2show, tuple(map(int, landmark.tolist())), (0,0,255), markerSize=10)
        # if count == 5:
        #     cv2.imwrite("im2show.jpg", im2show)


        # 描画・保存
        state = handler.out(im2show)
        if state == -1:
            break