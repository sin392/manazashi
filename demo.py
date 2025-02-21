import os
import sys
import cv2
import time
import argparse
import numpy as np
import matplotlib as mlp
from matplotlib import pyplot as plt

from detector import PersonFaceDetector
from gui import GUI
from M2Det.configs.CC import Config
from M2Det.utils.nms_wrapper import nms

def draw_p_det(img, rects, match_idx_list=(), sleep_idx_list=()):
    imgcv = np.copy(img)
    h, w, _ = imgcv.shape
    for i, box in enumerate(rects):
        color = (0, 128, 0)
        if len(match_idx_list) != 0:
            if i in [idx[1] for idx in match_idx_list]:
                color = (0, 50, 255)
        if len(sleep_idx_list) != 0:
            if i in sleep_idx_list:
                color = (255, 176, 0)

        box = [int(_) for _ in box]
        thick = int((h + w) / 300)
        cv2.rectangle(imgcv,
                      (box[0], box[1]), (box[2], box[3]),
                      color, thick)

    for i, box in enumerate(rects):
        cv2.putText(imgcv, text=str(i), org=(int(box[0]+5), int(box[1])+15), 
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,255,255), thickness=2)
    return imgcv

def draw_f_det(img, rects, landmarks=()):
    imgcv = np.copy(img)
    for i, rect in enumerate(rects):
        rect = tuple(map(int, rect.tolist()))
        cv2.rectangle(imgcv, pt1=rect[:2], pt2=rect[2:], color=(0,255,0), thickness=2)
        if len(landmarks) > 0:
            for landmark in landmarks[i]:
                cv2.drawMarker(imgcv, tuple(map(int, landmark.tolist())), (0,0,255), markerSize=10)
    return imgcv

class DataHandler():
    def __init__(self, im_path, cam, video_path, show):
        video = (type(video_path) != None)
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
                # video_path = input('Please enter video path: ')
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
        else:
            im_fnames = [im_path]
        self.im_iter = iter(im_fnames)

    def get_img(self):
        # -1: break, 0: sucess, 1: continue
        state = 0
        if self.cam < 0 and not self.video:
            try:
                fname = next(self.im_iter)
                if 'processed' in fname:
                    state = 1 # ignore the detected images
                self.fname = fname
                img = cv2.imread(fname, cv2.IMREAD_COLOR)
            except StopIteration:
                img = None
                state = -1
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

def get_fixed_rects(imgs, thresh=0.5):
    p_rects_probs = np.empty((0, 5))
    f_rects_probs = np.empty((0, 5))
    for i, img in enumerate(imgs):
        f_rects, f_probs, landmarks = detector.face_detect(img)
        p_rects, p_probs, class_inds = detector.person_detect(img)
        f_rects_probs = np.concatenate((f_rects_probs, np.hstack((f_rects, f_probs[:, np.newaxis]))), axis=0)
        p_rects_probs = np.concatenate((p_rects_probs, np.hstack((p_rects, p_probs[:, np.newaxis]))), axis=0)
    force_cpu = detector.device == "cpu"
    f_keep = nms(f_rects_probs, thresh, force_cpu)
    p_keep = nms(p_rects_probs, thresh, force_cpu)
    f_rects = f_rects_probs[f_keep, :4]
    p_rects = p_rects_probs[p_keep, :4]

    return f_rects, p_rects

class AnimationGraph():
    def __init__(self, show):
        if not show:
            mlp.use("Agg")
        self.x = np.zeros(100)
        # self.x = np.arange(0,100,1)
        self.y = np.zeros(100)
        self.fig = plt.figure(figsize=(6, 3))
        self.line, = plt.plot(self.x, self.y)
        plt.ylim(0,100)
        plt.xlim(0,100)
        plt.xlabel("time [s]")
        plt.ylabel("score [f/p]")
        self.fig.subplots_adjust(bottom=0.2)
        plt.ion()

    def update(self, x, y):
        self.x = np.append(self.x, x)
        self.x = np.delete(self.x, 0)
        self.y = np.append(self.y, y)
        self.y = np.delete(self.y, 0)
        self.line.set_data(self.x, self.y)
        if self.x[-1] >= 100:
            plt.xlim(min(self.x), max(self.x))
        plt.draw()
        plt.pause(0.001)

    def convert_fig2array(self):
        img = np.array(self.fig.canvas.renderer.buffer_rgba())
        return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='M2Det Testing')
    parser.add_argument('-c', '--config', default='M2Det/configs/m2det512_vgg.py', type=str)
    parser.add_argument('-i', '--image', default='sample.png', help='the path to demo images')
    parser.add_argument('-w', '--weight', default='weights/m2det512_vgg.pth', type=str, help='Trained state_dict file path to open')
    parser.add_argument('-v', '--video', default=None, type=str, help='the path to video file')
    parser.add_argument('--cam', default=-1, type=int, help='camera device id')
    parser.add_argument('--show', action='store_true', help='Whether to display the images')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--fixed', action='store_true')
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    im_path = args.image
    cam = args.cam
    video_path = args.video
    video = (type(args.video) != None)

    cfg = Config.fromfile(args.config)
    detector = PersonFaceDetector(cfg, args.weight)
    handler = DataHandler(im_path, cam, video_path, args.show)
    if (cam >= 0 or video): graph = AnimationGraph(args.show)
    if args.gui:
        gui = GUI()

    if cam >= 0 or video:
        calibration_num = 1 if args.fast else 3
    else:
        calibration_num = 0

    imgs = []
    count = 0
    while True:
        if args.gui:
            if gui.pause_flag:
                # 一時停止中の処理
                gui.root.update()
                continue
        img, state = handler.get_img()
        if state == -1:
            print("break")
            break
        elif state == 1:
            print("continue")
            continue
        
        start = time.time()
        if count < calibration_num:
            # 最初の3frameで人物領域をキャリブレーション
            imgs.append(img)
            if count == (calibration_num - 1):
                f_rects, p_rects = get_fixed_rects(imgs)
                fixed_f_rects, fixed_p_rects = f_rects, p_rects
                landmarks = ()
                if args.gui:
                    gui.set_score()
                    gui.set_state(detector.p_num)
            else:
                count += 1
                continue
        else:
            if args.fixed:
                f_rects, p_rects = fixed_f_rects, fixed_p_rects
            else:
                p_rects, p_probs, class_inds = detector.person_detect(img)
            f_rects, f_probs, landmarks = detector.face_detect(img)

        end = time.time()
        print(count)
        print("loop_time", end - start)

        # 顔bboxと人物bboxが両方検出されたもののリスト [..., (i,j), ...]
        match_idx_list = detector.get_match_idx_list(f_rects, p_rects)

        # 過去frameの比較 -> LSTM等に置き換え
        # 数フレーム顔が未検出状態の人物bboxのidリスト [..., j, ...]
        sleep_idx_list = detector.get_sleep_idx_list(match_idx_list) if args.fixed else []

        # 顔の誤検出抑制
        f_rects = detector.face_sup(f_rects, match_idx_list)

        # スコア算出
        pf_rate = detector.get_score(f_rects, p_rects)
        print("pfrate", pf_rate)
        # animation
        if (cam >= 0 or video): graph.update(count, pf_rate)

        # 描画
        im2show = draw_p_det(img, p_rects, match_idx_list, sleep_idx_list)
        im2show = draw_f_det(im2show, f_rects, landmarks=landmarks)
        if not args.gui:
            cv2.putText(im2show, f'face/person : {pf_rate:.2f}%', (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (100, 255, 100), 5, cv2.LINE_AA)
        elif (cam >= 0 or video):
            fig = graph.convert_fig2array()
            gui.update_img(gui.graph, fig)
            gui.update_img(gui.video, im2show[:,:,::-1])
            gui.update_text(gui.label_person, f"person       : {detector.p_num}")
            gui.update_text(gui.label_face,   f"focusing     : {detector.f_num}")
            gui.update_text(gui.label_bad,    f"looking away : {len(sleep_idx_list)}")
            gui.update_text(gui.label_score,  f"score        : {pf_rate:.2f}")
            # State : update_text使ったほうが良い？
            gui.update_state(detector.p_num, match_idx_list, sleep_idx_list)

        # 出力
        state = handler.out(im2show)
        if state == -1:
            break
        count += 1

    if args.gui: gui.root.mainloop()