import os
import sys
import cv2
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt

from detector import PersonFaceDetector
from M2Det.configs.CC import Config
from M2Det.utils.nms_wrapper import nms

def draw_p_det(img, rects, match_idx_list=()):
    imgcv = np.copy(img)
    h, w, _ = imgcv.shape
    for i, box in enumerate(rects):
        color = (255, 176, 0)
        if len(match_idx_list) != 0:
            if i in [idx[1] for idx in match_idx_list]:
                color = (0, 0, 255)

        box = [int(_) for _ in box]
        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      color, thick)
    return imgcv

def draw_f_det(img, rects, landmarks=()):
    imgcv = np.copy(img)
    for i, rect in enumerate(rects):
        rect = tuple(map(int, rect.tolist()))
        cv2.rectangle(imgcv, pt1=rect[:2], pt2=rect[2:], color=(0,255,0))
        if len(landmarks) > 0:
            for landmark in landmarks[i]:
                cv2.drawMarker(imgcv, tuple(map(int, landmark.tolist())), (0,0,255), markerSize=10)
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
            # fps = capture.get(cv2.CAP_PROP_FPS)
            fps = 1
            self.out_video = cv2.VideoWriter(video_name[0] + '_processed.mp4', fourcc, fps, (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
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
    force_cpu = detector.device == "cpu"
    f_keep = nms(f_rects_probs, 0.5, force_cpu)
    p_keep = nms(p_rects_probs, 0.5, force_cpu)
    f_rects = f_rects_probs[f_keep, :4]
    p_rects = p_rects_probs[p_keep, :4]
    return f_rects, p_rects

class AnimationGraph():
    def __init__(self):
        self.x = np.zeros(100)
        self.y = np.zeros(100)
        fig = plt.figure(figsize=(8, 4))
        self.line, = plt.plot(self.x, self.y)
        # plt.xlim(0,100)
        plt.ylim(0,100)
        plt.xlabel("time [s]")
        plt.ylabel("score [f/p]")

        plt.ion()

    def update(self, x, y):
        self.x = np.append(self.x, x)
        self.x = np.delete(self.x, 0)
        self.y = np.append(self.y, y)
        self.y = np.delete(self.y, 0)
        self.line.set_data(self.x, self.y)
        plt.xlim(min(self.x), max(self.x))
        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='M2Det Testing')
    parser.add_argument('-c', '--config', default='M2Det/configs/m2det512_vgg.py', type=str)
    parser.add_argument('-f', '--directory', default='sample.png', help='the path to demo images')
    parser.add_argument('-w', '--weight', default='weights/m2det512_vgg.pth', type=str, help='Trained state_dict file path to open')
    parser.add_argument('--cam', default=-1, type=int, help='camera device id')
    parser.add_argument('--video', action='store_true', help='videofile mode')
    parser.add_argument('--show', action='store_true', help='Whether to display the images')
    parser.add_argument('--crop', action='store_true', help='Crop Bbox of Person Class')
    parser.add_argument('--fixed', action='store_true')
    args = parser.parse_args()

    im_path = args.directory
    cam = args.cam
    video = args.video

    cfg = Config.fromfile(args.config)
    detector = PersonFaceDetector(cfg, args.weight)
    handler = DataHandler(im_path, cam, video, args.show)
    graph = AnimationGraph()

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
        start = time.time()
        if count < 5:
            imgs.append(img)
            if count == 4:
                f_rects, p_rects = get_fixed_rects(imgs)
                fixed_f_rects, fixed_p_rects = f_rects, p_rects
                landmarks = ()
            else:
                count += 1
                continue
        else:
            if args.fixed:
                f_rects, p_rects = fixed_f_rects, fixed_p_rects
            else:
                p_rects, p_probs, _ = detector.person_detect(img)
            f_rects, f_probs, landmarks = detector.face_detect(img)

        end = time.time()
        print("loop_time", end - start)

        match_idx_list = detector.get_match_idx_list(f_rects, p_rects)

        # 顔の誤検出抑制
        f_rects = detector.face_sup(f_rects, match_idx_list)

        # スコア算出
        pf_rate = detector.get_score(f_rects, p_rects)
        print("pfrate", pf_rate)
        # animation
        graph.update(count, pf_rate)


        # 描画
        im2show = draw_p_det(img, p_rects, match_idx_list=match_idx_list)
        im2show = draw_f_det(im2show, f_rects, landmarks=landmarks)
        cv2.putText(im2show, f'face/person : {pf_rate:.2f}%', (20, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (100, 255, 100), 5, cv2.LINE_AA)

        # 出力
        state = handler.out(im2show)
        if state == -1:
            break
    
        count += 1