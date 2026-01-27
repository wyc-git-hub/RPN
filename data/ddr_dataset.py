"""自定义 Dataset 类（DDR 读取）占位实现"""
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.label_generator import LabelGenerator


class DDRDataset(Dataset):
    """
    DDR 数据集加载器

    功能:
    1. 读取原始图像和 GT Mask
    2. 根据 lesion_type 提取对应的二值 Mask
    3. 应用数据增强 (Transforms)
    4. 调用 LabelGenerator 生成 RSM 和 PFM
    """

    # 颜色映射 (RGB) -> 对应论文 Figure 4 Caption
    # 注意: OpenCV 读取是 BGR，这里我们统一转为 RGB 处理
    LESION_COLORS = {
        'EX': [255, 0, 0],  # Red
        'HE': [0, 255, 0],  # Green
        'MA': [0, 0, 255],  # Blue
        'SE': [255, 0, 255]  # Pink (Magenta)
    }

    def __init__(self,
                 root_dir,
                 mode='train',
                 lesion_type='MA',
                 transforms=None,
                 rsm_kernel_size=35,
                 pfm_kernel_size=31):
        """
        Args:
            root_dir (str): 数据集根目录 (包含 'train', 'valid', 'test' 子文件夹)
            mode (str): 'train', 'valid' 或 'test'
            lesion_type (str): 'EX', 'HE', 'MA', 'SE'
            transforms (callable): 数据增强操作
            rsm_kernel_size (int): LabelGenerator 参数
            pfm_kernel_size (int): LabelGenerator 参数
        """
        self.root_dir = root_dir
        self.mode = mode
        self.lesion_type = lesion_type.upper()
        self.transforms = transforms

        if self.lesion_type not in self.LESION_COLORS:
            raise ValueError(f"Unknown lesion type: {self.lesion_type}. Choose from EX, HE, MA, SE.")

        # 初始化 LabelGenerator
        self.label_gen = LabelGenerator(rsm_kernel_size, pfm_kernel_size)

        # 获取文件列表
        # 假设 DDR 目录结构:
        # root/train/image/*.jpg
        # root/train/label/*.png
        self.img_dir = os.path.join(root_dir, mode, 'image')
        self.mask_dir = os.path.join(root_dir, mode, 'label')

        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')]

        # 简单的验证
        if len(self.img_names) == 0:
            print(f"Warning: No images found in {self.img_dir}")

    def __len__(self):
        return len(self.img_names)

    def _decode_mask(self, mask_rgb):
        """
        从彩色 Mask 中提取特定病灶的二值 Mask
        """
        target_color = self.LESION_COLORS[self.lesion_type]

        # 创建二值 Mask
        # 匹配 target_color 的像素设为 1，其他为 0
        # 允许一点点颜色误差 (针对压缩图片)
        lower_bound = np.array([c - 20 for c in target_color])
        upper_bound = np.array([c + 20 for c in target_color])

        binary_mask = cv2.inRange(mask_rgb, lower_bound, upper_bound)

        # 归一化到 0/1
        binary_mask = (binary_mask > 0).astype(np.float32)

        return binary_mask

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 假设 mask 文件名和图片一致，或者是 .png 后缀
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 1. 读取图像 (OpenCV read as BGR)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转 RGB

        # 2. 读取 Mask
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # 转 RGB
            # 提取特定病灶
            mask = self._decode_mask(mask)
        else:
            # 如果没有 Mask (比如测试集或者图片没有该病灶)，返回全黑
            mask = np.zeros(image.shape[:2], dtype=np.float32)

        # 3. 应用数据增强 (Transforms)
        if self.transforms:
            image, mask = self.transforms(image, mask)

        # 注意: 如果 transforms 里包含了 ToTensor，这里 image 已经是 Tensor [3, H, W]
        # mask 是 Tensor [1, H, W]

        # 4. 在线生成 RSM 和 PFM (核心步骤)
        # LabelGenerator 需要 Tensor 输入，且在 CPU 上运行
        # mask shape: [1, H, W]

        # 生成 RSM (概率图)
        rsm_gt = self.label_gen.generate_rsm(mask)  # 返回 Tensor [1, H, W]

        # 生成 PFM (0,1,2 三值图)
        # generate_pfm 接收 numpy [H, W]，所以需要先转回去一下
        mask_np = mask.squeeze().numpy()
        pfm_np = self.label_gen.generate_pfm(mask_np)
        pfm_gt = torch.from_numpy(pfm_np).long()  # 返回 Tensor [H, W] (Long类型用于CrossEntropy)

        return {
            "image": image,
            "rsm_gt": rsm_gt,
            "pfm_gt": pfm_gt,
            "original_mask": mask,  # 原始二值 mask，用于 Validation 计算 Dice/IoU
            "img_name": img_name
        }


# --- 单元测试代码 ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from transforms_ddr import Compose, Resize, HistogramEqualization, ToTensor

    # 模拟路径
    root_dir = "/dataset/DR_grading/DR_grading"  # 请替换为你实际的 DDR 路径

    # 定义变换
    transforms = Compose([
        Resize((1000, 960)),  # Paper size
        HistogramEqualization(),  # Paper preprocessing
        ToTensor()
    ])

    # 假设我们有一个包含假数据的目录，这里仅作实例化演示
    try:
        dataset = DDRDataset(
            root_dir=root_dir,
            mode='train',
            lesion_type='MA',  # 训练微动脉瘤
            transforms=transforms,
            rsm_kernel_size=35,  # 针对 MA 的参数
            pfm_kernel_size=17
        )

        print(f"Dataset initialized. Length: {len(dataset)}")

        # 如果有数据，尝试读取一个
        if len(dataset) > 0:
            sample = dataset[0]
            print("Keys:", sample.keys())
            print("Image Shape:", sample['image'].shape)
            print("RSM Shape:", sample['rsm_gt'].shape)
            print("PFM Shape:", sample['pfm_gt'].shape)
            print("PFM Unique Values:", torch.unique(sample['pfm_gt']))

    except Exception as e:
        print(f"Test skipped or failed: {e}")
        print("请确保 'root_dir' 指向真实的 DDR 数据集路径以运行完整测试。")