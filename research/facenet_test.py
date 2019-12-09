# %%
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
from time import time
from IPython.display import display
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import numpy as np

# %%
mtcnn = MTCNN(image_size=1440, margin=0)

# %%
img_path = "image_00032.png"
img = Image.open(img_path)
img_cropped = mtcnn(img)

# %%
display(T.ToPILImage()(img_cropped))

# %%
rects, probs, landmarks = mtcnn.detect(img, landmarks=True)
img_copy = img.copy()
draw = ImageDraw.Draw(img_copy)
for i in range(len(rects)):
    print(i, probs[i])
    draw.rectangle(rects[i].tolist(), outline="blue")
    # draw.point(landmarks[i], fill="blue")
    img_copy = np.asarray(img_copy)
    cv2.drawMarker(img_copy, position, "blue", markerSize=20)
    img_copy = Image.fromarray(img_copy)
display(img_copy)
# %%
img_copy.save("output_facenet.png")

# %%
