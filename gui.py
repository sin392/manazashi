import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import time

class GUI(tk.Frame):
    def __init__(self):
        root = tk.Tk()

        video = tk.Canvas(root, width=600, height=300, bg="gray")
        video.grid(column=0, row=0, padx=5, pady=2)
        video.create_text(
            300, 150, text="Video", font=("Helvetica", 18, "bold"), fill="white", justify="center"
        )
        video.bind("<1>", self.pause_handler)

        statelist = tk.Canvas(root, width=200, height=300, bg="gray")
        statelist.grid(column=1, row=0, padx=5, pady=2)
        statelist.create_text(
            100, 150, text="State", font=("Helvetica", 18, "bold"), fill="white", justify="center"
        )

        graph = tk.Canvas(root, width=600, height=300, bg="gray")
        graph.grid(column=0, row=1, padx=5, pady=2)
        graph.create_text(
            300, 150, text="Graph", font=("Helvetica", 18, "bold"), fill="white", justify="center"
        )

        score = tk.Canvas(root, width=200, height=300, bg="gray")
        score.grid(column=1, row=1, padx=5, pady=2)
        score.pack_propagate(0)
        score.create_text(
            100, 150, text="Score", font=("Helvetica", 18, "bold"), fill="white", justify="center"
        )
        pad = tk.Frame(score, width=1, height=2, bg="gray")
        pad.pack(pady=2)
        self.score_1 = self.make_label(score, "person")
        self.score_2 = self.make_label(score, "face")
        self.score_3 = self.make_label(score, "score")

        self.root = root
        self.video = video
        self.graph = graph
        self.score = score
        # False:run, True:pause
        self.pause_flag = False

    def make_label(self, root, text):
        label = tk.Label(root, width=20, bg="white", text=text, font="Helvetica", anchor="w")
        label.pack(pady=2)
        return label

    def update_text(self, label, text):
        label["text"] = text
        self.root.update()

    def update_img(self, frame, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")
        x = frame.winfo_width()
        y = frame.winfo_height()
        scale = 0.95
        img = img.resize((int(x*scale),int(y*scale)))
        frame.tkimg = ImageTk.PhotoImage(img)
        frame.create_image(x/2, y/2, image=frame.tkimg)
        self.root.update()

    def pause(self):
        if not self.pause_flag:
            print("pause")
        else:
            print("restart")
        self.pause_flag = not self.pause_flag    
    
    def pause_handler(self, event):
        self.pause()

if __name__ == "__main__":
    img = Image.open("sample_processed.jpg").convert("RGB")
    img = img.resize((300, 300))

    gui = GUI()
    gui.root.update()
    gui.update_img(gui.graph, img)
    # gui.update_img(gui.video, img)
    gui.root.mainloop()