import os
import random
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=224, mode='train', augmentation_prob=0.4):
        """"Initializes image paths and preprocessing module"""
        self.root = root

        # GT: Ground Truth
        self.GT_paths = root[:-1] + '_GT/'  # 指向地面真实标签（Ground Truth，GT）图像存储目录的路径
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))   # 函数将map返回的迭代器转换为一个列表。这个列表包含了目录 root 中所有文件的完整路径
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """"Reads an image from path and preprocesses it and returns """
        image_path = self.image_paths[index]
        filename = image_path.split('_')[-1][:-len(".jpg")]  # 留下纯粹的文件名
        GT_path = self.GT_paths + 'ISIC_' + filename + '_segmentation.png'

        image = Image.open(image_path)
        GT = Image.open(GT_path)

        aspect_ratio = image.size[1] / image.size[0]

        Transform = []

        ResizeRange = random.randint(300, 320)
        Transform.append(T.Resize((int(ResizeRange*aspect_ratio), ResizeRange)))
        p_transform = random.random()

        if (self.mode == 'train') and p_transform <self.