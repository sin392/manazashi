# %%
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
from time import time
from IPython.display import display
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
# thresholds = [0.5, 0.6, 0.6]
mtcnn = MTCNN(image_size=1440, margin=0, keep_all=True, device=device, thresholds=thresholds)

# %%
img_path = "image_00032.png"
img = Image.open(img_path)
img_cropped = mtcnn(img)

# %%
display(T.ToPILImage()(img_cropped))

# %%
rects, probs, landmarks = mtcnn.detect(img, landmarks=True)
img_copy = img.copy()

img_copy = np.asarray(img_copy)
for i in range(len(rects)):
    for landmark in landmarks[i]:
        cv2.drawMarker(img_copy, tuple(map(int, landmark.tolist())), (0,0,255), markerSize=10)

img_copy = Image.fromarray(img_copy)
draw = ImageDraw.Draw(img_copy)
for rect in rects:
    draw.rectangle(tuple(rect.tolist()), outline="green")
display(img_copy)
# %%
img_copy.save("output_facenet.png")

# %%
