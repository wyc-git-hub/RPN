import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.label_generator import LabelGenerator


class IDRiDDataset(Dataset):
    """
    IDRiD (Indian Diabetic Retinopathy Image Dataset) 加载器

    特点:
    1. 目录结构较深 (A.Segmentation/...)
    2. 不同病灶分在不同文件夹
    3. Mask 文件名通常包含后缀 (如 IDRiD_01_MA.tif)
    """

    # 官方文件夹名称映射
    LESION_FOLDERS = {
        'MA': '1. Microaneurysms',
        'HE': '2. Haemorrhages',
        'EX': '3. Hard Exudates',
        'SE': '4. Soft Exudates',
        'OD': '5. Optic Disc'  # IDRiD 也有视盘分割
    }

    # Mask 文件名后缀映射
    # IDRiD mask 通常命名为: IDRiD_01_MA.tif
    LESION_SUFFIX = {
        'MA': '_MA',
        'HE': '_HE',
        'EX': '_EX',
        'SE': '_SE',
        'OD': '_OD'
    }

    def __init__(self,
                 root_dir,
                 mode='train',
                 lesion_type='MA',
                 transforms=None,
                 rsm_kernel_size=35,
                 pfm_kernel_size=17):
        """
        Args:
            root_dir (str): IDRiD 根目录 (包含 'A.Segmentation')
            mode (str): 'train' 或 'valid'/'test'
            lesion_type (str): 'MA', 'HE', 'EX', 'SE'
        """
        self.root_dir = root_dir
        self.mode = mode
        self.lesion_type = lesion_type.upper()
        self.transforms = transforms

        # 初始化 LabelGenerator
        self.label_gen = LabelGenerator(rsm_kernel_size, pfm_kernel_size)

        # 1. 构建路径
        # 根据 mode 选择子文件夹 ('a. Training Set' 或 'b. Testing Set')
        if mode == 'train':
            subset_folder = 'a. Training Set'
        else:
            subset_folder = 'b. Testing Set'

        # 图像路径
        self.img_dir = os.path.join(root_dir, 'A. Segmentation', '1. Original Images', subset_folder)

        # Mask 路径 (根据 lesion_type 决定进入哪个子文件夹)
        lesion_folder_name = self.LESION_FOLDERS[self.lesion_type]
        self.mask_dir = os.path.join(root_dir, 'A. Segmentation', '2. All Segmentation Groundtruths', subset_folder,
                                     lesion_folder_name)

        # 2. 获取文件列表
        # 只读取 jpg 或 tif
        if os.path.exists(self.img_dir):
            self.img_names = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.tif', '.png'))]
            self.img_names.sort()  # 排序保证一致性
        else:
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        print(f"[{mode.upper()}] Loading IDRiD {self.lesion_type} from: {self.img_dir}")
        print(f"Found {len(self.img_names)} images.")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 构建 Mask 路径
        # IDRiD 图片名: IDRiD_01.jpg
        # Mask 名通常是: IDRiD_01_MA.tif
        name_no_ext = os.path.splitext(img_name)[0]
        suffix = self.LESION_SUFFIX[self.lesion_type]
        mask_name = f"{name_no_ext}{suffix}.tif"
        mask_path = os.path.join(self.mask_dir, mask_name)
        # 1. 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 读取 Mask
        # IDRiD 某些图片可能没有某种病灶，此时 Mask 文件可能不存在
        if os.path.exists(mask_path):

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读为单通道
            # 确保二值化 (0, 1)
            mask = (mask > 0).astype(np.float32)
            # === 🕵️‍♂️ 侦探插入点 BEGIN ===
            # if idx == 0:
            #     print(f"\n[DEBUG] Mask File: {mask_path}")
            #     print(f"[DEBUG] Mask Shape: {mask.shape}")
            #     print(f"[DEBUG] Mask Unique Values: {np.unique(mask)}")  # 看看到底有没有大于 127 的数
            # === 🕵️‍♂️ 侦探插入点 END ===
        else:
            # 如果没有 Mask，生成全黑图
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        # 3. 应用数据增强 (Transforms)
        if self.transforms:
            image, mask = self.transforms(image, mask)

        # image 现在是 Tensor [3, H, W]
        # mask 现在是 Tensor [1, H, W] (Float)

        # 需要转 numpy
        # mask_np = mask.squeeze().numpy().astype(np.uint8)
        # # 注意：这里调用 label_gen 的 generate_pfm 方法
        # # 虽然 label_gen 变成了 nn.Module，但 CPU 方法依然可用
        # pfm_np = self.label_gen.generate_pfm(mask_np)
        # pfm_gt = torch.from_numpy(pfm_np).long()

        # 2. RSM 不在这里生成了！
        # 直接返回原始的 mask，稍后在 GPU 上做卷积

        return {
            "image": image,
            "mask_binary": mask,  # <--- 新增：返回原始二值 mask (用于生成 RSM)
            # "pfm_gt": pfm_gt,  # PFM 还是这里产出
            "original_mask": mask,
            "img_name": img_name
        }


# --- 单元测试代码 ---
if __name__ == "__main__":
    from transforms_IDRiD import Compose, Resize, HistogramEqualization, ToTensor

    # 请修改为你的 IDRiD 真实路径
    # 目录结构必须符合 A.Segmentation/...
    root_dir = "/path/to/IDRiD_dataset"

    transforms = Compose([
        Resize((640, 960)),  # IDRiD 推荐尺寸 (H, W)
        HistogramEqualization(),
        ToTensor()
    ])

    try:
        # 测试读取微动脉瘤 (MA)
        dataset = IDRiDDataset(
            root_dir=root_dir,
            mode='train',
            lesion_type='MA',
            transforms=transforms,
            rsm_kernel_size=35,
            pfm_kernel_size=17
        )

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Image Shape: {sample['image'].shape}")
            print(f"RSM Shape: {sample['rsm_gt'].shape}")
            print(f"PFM Shape: {sample['pfm_gt'].shape}")
            print(f"Mask path exists: {sample['original_mask'].max() > 0}")
            print("IDRiD Dataset test passed!")

    except Exception as e:
        print(f"Skipping test due to path error: {e}")
        print("Tip: Ensure your IDRiD folder structure matches 'A.Segmentation/...'")