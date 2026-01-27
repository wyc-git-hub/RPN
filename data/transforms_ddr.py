"""图像增强占位：翻转、旋转、直方图均衡化等"""

import torch
import numpy as np
import cv2
import random
from torchvision.transforms import functional as TF


class Compose:
    """组合多个变换操作"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize:
    """
    调整大小
    论文提到 DDR 缩放到 960x1000
    """

    def __init__(self, size=(1000, 960)):  # (H, W)
        self.size = size

    def __call__(self, image, mask):
        # OpenCV resize dsize is (W, H)
        image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        # Mask 使用最近邻插值，保证值依然是 0/1
        mask = cv2.resize(mask, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
        return image, mask


class HistogramEqualization:
    """
    直方图均衡化
    对应论文: "histogram equalization was applied to all images"
    只对图像做，不对 Mask 做
    """

    def __init__(self, clahe=True):
        self.clahe = clahe
        # 使用 CLAHE (限制对比度自适应直方图均衡化) 通常比全局均衡化效果更好，且不易产生噪声
        if clahe:
            self.clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, image, mask):
        # image is numpy array (H, W, 3) in BGR or RGB
        # 转换到 LAB 空间处理 L 通道是标准做法
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)

        if self.clahe:
            l = self.clahe_obj.apply(l)
        else:
            l = cv2.equalizeHist(l)

        img_lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

        return image, mask


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return image, mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        return image, mask


class RandomRotate:
    """
    随机旋转
    对应论文: "random rotations"
    """

    def __init__(self, limit=15, p=0.5):
        self.limit = limit
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            angle = random.uniform(-self.limit, self.limit)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
        return image, mask


class ToTensor:
    """转换为 PyTorch Tensor 并归一化"""

    def __call__(self, image, mask):
        # Image: [H, W, 3] -> [3, H, W], 0-255 -> 0-1
        image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

        # 标准化 (使用 ImageNet 均值方差或自定义)
        # 这里使用简单的 0.5 均值标准化，或者你可以计算 DDR 数据集的均值
        norm = torch.nn.Sequential(
            torch.nn.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 示例值
        )
        image = norm(image)

        # Mask: [H, W] -> [1, H, W], 保持 0/1
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask