import os
import torch
import numpy as np
from torchvision import transforms,utils
from torch.utils.data import Dataset,DataLoader
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transform, loader=default_loader):
        fh = open(txt, 'r',encoding='utf-8')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            location=line.find('/')
            imgs.append((line, int(line[location+1:location+4])-1))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader('../../新菊花数据集/' + fn)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)