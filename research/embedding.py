import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

class CustomResnet(nn.Module):
    def __init__(self):
        super(CustomResnet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        res = torchvision.models.resnet152(pretrained=True)
        self.layer = res._modules.get('avgpool')
        self.output_size = 2048

        res.eval()
        self.res = res.to(self.device)


    def get_feature_vector(self, images):
        embedding = torch.zeros(self.output_size).to(self.device)
        def copy_data(m, input, output):
            # print("output", output.size())
            embedding.copy_(output.data.squeeze())

        h = self.layer.register_forward_hook(copy_data)

        inputs = images.to(self.device)

        self.res(inputs)
        h.remove()
        return embedding

    def forward(self, x):
        return res(x)

def get_features(files, embedder):
    tensor_img_size = 224
    resize_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((tensor_img_size, tensor_img_size)),
        transforms.ToTensor(),
    ])

    device = embedder.device

    features = torch.zeros(0).to(device)
    label_imgs = torch.zeros(0).to(device)
    images = torch.zeros(0).to(device)
    labels = []
    img_size = 56  # thumnbail image size

    for i in tqdm(range(len(files))):
        img = resize_transform(Image.open(files[i])).to(device)
        img = img.unsqueeze(0)
        # images = torch.cat((images, img))
        label_img = F.interpolate(img, size=img_size)  # use 1st frame for visualize
        label_imgs = torch.cat((label_imgs, label_img))
        feature = embedder.get_feature_vector(img)
        features = torch.cat((features, feature.unsqueeze(0)))
    
    return features, label_imgs


if __name__ == "__main__":
    writer = SummaryWriter("logs")
    files = sorted(glob("/home/shimine/face/manazashi/cropped/*"))
    embedder = CustomResnet()
    features, label_imgs = get_features(files, embedder)
    writer.add_embedding(features, label_img=label_imgs)
    # print(features.size(), label_imgs.size())
    writer.close()
