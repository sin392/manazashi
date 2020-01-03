import tkinter as tk
from PIL import Image,ImageTk
import cv2
import numpy as np

class GUI:
    def __init__(self):
        self.cvv=CV2()
        self.root=tk.Tk()
        self.ROOT_X = 1000
        self.ROOT_Y = 700
        self.CANVAS_X=640
        self.CANVAS_Y=480
        self.root.title(u"tkcv")
        self.root.geometry(str(self.ROOT_X) + "x" + str(self.ROOT_Y))
        self.root.resizable(width=0, height=0)

        self.count_num=0

        self.firstFrame()
        self.afterMSec()

    def afterMSec(self):
        self.count_num += 1
        self.label_count.configure(text=str(self.count_num))

        self.cvv.cameraFrame()

        self.loop_img = Image.fromarray(self.cvv.frame_flip)

        self.canvas_img = ImageTk.PhotoImage(self.loop_img)
        self.canvas.create_image(self.CANVAS_X / 2, self.CANVAS_Y / 2, image=self.canvas_img)

        self.root.after(10, self.afterMSec)

    def firstFrame(self):
        self.first_frame = tk.Frame(self.root, bd=2, relief="ridge", bg="white",
                                    width=self.ROOT_X, height=self.ROOT_Y)
        self.first_frame.grid(row=0, column=0)

        self.label_count = tk.Label(self.first_frame, text=str(self.count_num),font=("", 40))
        self.label_count.place(x=50,y=50,width=600)

        self.canvas = tk.Canvas(self.root, width=self.CANVAS_X, height=self.CANVAS_Y)
        self.canvas.create_rectangle(0, 0, self.CANVAS_X, self.CANVAS_Y, fill="#696969")
        self.canvas.place(x=300, y=200)


class CV2:
    def __init__(self):
        self.openCamera()

    def openCamera(self):
        self.cap = cv2.VideoCapture(0)

    def cameraFrame(self):
        _,self.frame=self.cap.read()
        self.frame_flip = cv2.flip(self.frame, 1)
        self.frame_flip = self.frame_flip[:, :, ::-1]


class Main:
    def __init__(self):
        self.gui=GUI()
        self.gui.root.mainloop()


if __name__=="__main__":
    Main()
