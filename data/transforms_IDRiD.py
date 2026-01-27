import torch
import numpy as np
import cv2
import random
from torchvision import transforms

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
    IDRiD 原图很大 (4288x2848)，必须缩放。
    通常缩放到 (width=960, height=640) 或 (width=1024, height=720)
    或者为了网络下采样方便，设为 32 的倍数。
    """

    def __init__(self, size=(640, 640)):  # (H, W) 注意顺序
        self.size = size

    def __call__(self, image, mask):
        # OpenCV resize dsize is (W, H)
        image = cv2.resize(image, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR)
        # Mask 使用最近邻插值，保证值依然是 0/1
        mask = cv2.resize(mask, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)
        return image, mask


class HistogramEqualization:
    """
    直方图均衡化 (CLAHE)
    这对 IDRiD 数据集非常重要，因为部分图像光照不均。
    """

    def __init__(self, clahe=True):
        self.clahe = clahe
        if clahe:
            self.clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, image, mask):
        # image is numpy array (H, W, 3) in RGB
        # 转换到 LAB 空间处理 L 通道
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

        # 标准化 (建议计算 IDRiD 自己的均值方差，这里使用通用值)
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        # Mask: [H, W] -> [1, H, W]
        # 确保 mask 只有 0 和 1 (IDRiD 的 tif 有时是 0/255)
        if mask.max() > 1:
            mask = mask / 255.0

        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask