import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

class GUI(tk.Frame):
    def __init__(self):
        root = tk.Tk()

        video = tk.Canvas(root, width=600, height=300, bg="gray")
        video.grid(column=0, row=0, padx=5, pady=2)
        video.create_text(
            300, 150, text="Video", font=("Helvetica", 18, "bold"), fill="white", justify="center"
        )
        video.bind("<1>", self.pause)

        statelist = tk.Canvas(root, width=200, height=300, bg="gray")
        statelist.grid(column=1, row=0, padx=5, pady=2)
        statelist.propagate(0)
        statelist.create_text(
            100, 150, text="State", font=("Helvetica", 18, "bold"), fill="white", justify="center"
        )
        bar = tk.Scrollbar(statelist, orient=tk.VERTICAL)
        bar.pack(side=tk.RIGHT, fill=tk.Y)
        bar.config(command=statelist.yview)
        bar.bind("<B1-Motion>", self.pause)

        statelist.config(scrollregion=(0,0,1000,1000))
        # statelist.configure(scrollregion=statelist.bbox("all"))
        statelist.config(yscrollcommand=bar.set)

        statelist_frame = tk.Frame(statelist, width=180, height=1000, bg="gray")
        statelist_frame.propagate(0)

        pad = tk.Frame(statelist_frame, width=1, height=2, bg="")
        pad.pack(pady=2)

        graph = tk.Canvas(root, width=600, height=300, bg="gray")
        graph.grid(column=0, row=1, padx=5, pady=2)
        graph.create_text(
            300, 150, text="Graph", font=("Helvetica", 18, "bold"), fill="white", justify="center"
        )

        score = tk.Canvas(root, width=200, height=300, bg="gray")
        score.grid(column=1, row=1, padx=5, pady=2)
        score.propagate(0)
        score.create_text(
            100, 150, text="Score", font=("Helvetica", 18, "bold"), fill="white", justify="center"
        )
        pad = tk.Frame(score, width=1, height=2, bg="")
        pad.pack(pady=2)

        root.update()

        self.root = root
        self.video = video
        self.graph = graph
        self.statelist = statelist
        self.statelist_frame = statelist_frame
        self.bar = bar
        self.score = score
        # False:run, True:pause
        self.pause_flag = False

    def set_score(self):
        self.label_person = self.make_label(self.score, "person", padx=5)
        self.label_face = self.make_label(self.score, "good", padx=5)
        self.label_bad = self.make_label(self.score, "bad", padx=5)
        self.label_score = self.make_label(self.score, "score", padx=5)
        self.score.update()

    def set_state(self, num):
        self.person = []
        self.statelist.create_window((5,2), window=self.statelist_frame, anchor="nw", tag="state")
        for i in range(num):
            person_state = "normal"
            fg = "black"
            self.person.append(self.make_label(self.statelist_frame, f"p_{i} : {person_state}", fg=fg))
        self.statelist.update()


    def make_label(self, frame, text="", width=20, height=1, bg="white", fg="black", padx=2, pady=2):
        label = tk.Label(frame, width=width, height=height, bg=bg, fg=fg, text=text, font="Helvetica", anchor="w")
        label.pack(padx=padx, pady=pady, fill=tk.X)
        frame.update()
        return label
    
    def update_state(self, num, match_idx_list, sleep_idx_list):
        self.statelist.delete("state")
        self.statelist.create_window((5,2), window=self.statelist_frame, anchor="nw", tag="state")
        for i in range(1):
            if i in [idx[1] for idx in match_idx_list]:
                person_state = "focusing"
                fg = "orange red"
            elif i in sleep_idx_list:
                person_state = "looking away"
                fg = "RoyalBlue1"
            else:
                person_state = "normal"
                fg = "green4"
            self.person[i]["text"] = f"p_{i} : {person_state}"
            self.person[i]["fg"] = fg
        self.statelist_frame.update()
        self.statelist.update()

    def update_text(self, label, text):
        label["text"] = text
        label.update()

    def update_img(self, frame, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")
        x = frame.winfo_width()
        y = frame.winfo_height()
        scale = 0.95
        img = img.resize((int(x*scale),int(y*scale)), resample=Image.BICUBIC)
        frame.tkimg = ImageTk.PhotoImage(img)
        frame.create_image(x/2, y/2, image=frame.tkimg)
        frame.update()

    def pause(self, event):
        if not self.pause_flag:
            print("pause")
        else:
            print("restart")
        self.pause_flag = not self.pause_flag    

    def destroy_child(self, frame, keep_pad=False):
        children = frame.winfo_children()
        if keep_pad:
            start = 1
        else:
            start = 0
        for child in children[start:]:
            child.pack_forget()
            # child.destroy()
        frame.update()

if __name__ == "__main__":
    img = Image.open("sample_processed.jpg").convert("RGB")
    img = img.resize((300, 300))

    gui = GUI()
    # gui.update_img(gui.graph, img)
    # gui.update_img(gui.video, img)
    gui.set_state(3)
    # statelist_frame = tk.Frame(gui.statelist, width=100, height=100, bg="blue")
    # statelist_frame.propagate(0)
    # gui.statelist.delete("state")
    # label = tk.Label(statelist_frame, bg="white", text="aaa")
    # label.pack()
    gui.statelist.create_window((0,0), window=gui.statelist_frame, anchor="nw", tag="state")
    gui.root.mainloop()