"""将 GT 转为 RSM 和 PFM 的占位实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class LabelGenerator(nn.Module):
    """
    RPN 标签生成器
    负责生成:
    1. RSM (Region-based Supervised Mask): 概率图，用于周边视觉分支
    2. PFM (Pixel-level Focal Mask): 三值掩模 (病灶/ROI背景/忽略背景)，用于中央视觉分支
    """

    """
        RPN 标签生成器 (GPU 加速版)
    """

    def __init__(self, rsm_kernel_size=35, pfm_kernel_size=31):
        super(LabelGenerator, self).__init__()
        self.rsm_k = rsm_kernel_size
        self.pfm_k = pfm_kernel_size

        # --- 2. 注册 Buffer (让权重随模型移动到 GPU) ---
        # K(m,n) = 1 / (M * N)
        value = 1.0 / (self.rsm_k * self.rsm_k)
        kernel = torch.full((1, 1, self.rsm_k, self.rsm_k), value)
        self.register_buffer('rsm_weight', kernel)

    def _create_rsm_kernel(self):
        """
        创建固定的平均池化卷积核
        K(m,n) = 1 / (M * N)
        """
        value = 1.0 / (self.rsm_k * self.rsm_k)
        # Shape: [Out_channels=1, In_channels=1, H, W]
        kernel = torch.full((1, 1, self.rsm_k, self.rsm_k), value)
        return kernel

    def generate_rsm(self, mask_tensor):
        """
        [GPU/Tensor 操作] 生成区域监督掩模 (RSM)

        Args:
            mask_tensor (torch.Tensor): 二值 Mask, Shape [B, 1, H, W] 或 [B, H, W]
                                      值必须为 0.0 或 1.0
        Returns:
            rsm (torch.Tensor): 概率图, Shape [B, 1, H, W]
        """
        # 1. 维度检查与调整
        # 输入可能是 [B, H, W] (Dataloader 默认输出)，需要升维到 [B, 1, H, W]
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(1)

        # 2. 确保数据类型为 float (卷积要求)
        if mask_tensor.dtype != torch.float32:
            mask_tensor = mask_tensor.float()

        # 3. 执行卷积
        # self.rsm_weight 已经在正确的 device 上了
        # padding = k // 2 保证输出尺寸与输入一致 (Same Padding)
        rsm = F.conv2d(mask_tensor, self.rsm_weight, padding=self.rsm_k // 2)

        return rsm

    # def generate_pfm(self, mask_np):
        # """
        # [CPU/Numpy 操作] 生成像素级聚焦掩模 (PFM)
        #
        # 注意: 由于 OpenCV 不支持 GPU Tensor，此函数通常在 Dataset 或 CPU 数据预处理阶段调用。
        #
        # Args:
        #     mask_np (numpy.ndarray): 二值 Mask, Shape [H, W], 值 0 或 1
        #
        # Returns:
        #     pfm (numpy.ndarray): 三值 Mask, Shape [H, W]
        #                          1: Lesion (病灶)
        #                          0: Surrounding (ROI 背景)
        #                          2: Ignore (忽略背景)
        # """
        # # 1. 确保输入是 uint8 类型 (OpenCV 要求)
        # mask_uint8 = mask_np.astype(np.uint8)
        #
        # # 2. 确定膨胀区域 (Surrounding Area)
        # # 对应论文: B' = B ⊕ D
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.pfm_k, self.pfm_k))
        # dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)
        #
        # # 3. 构建三值 Mask
        # # 初始化全图为 2 (Ignore)
        # pfm = np.full_like(mask_np, 2, dtype=np.uint8)
        #
        # # 将膨胀区域设为 0 (ROI Background)
        # pfm[dilated_mask == 1] = 0
        #
        # # 将原始病灶区域设为 1 (Lesion) -> 覆盖掉刚才的 0
        # pfm[mask_uint8 == 1] = 1
        #
        # return pfm
    def generate_pfm(self, mask_tensor):
        """
        [GPU] 生成像素级聚焦掩模 (PFM)
        利用 MaxPool2d 实现形态学膨胀，替代 cv2.dilate

        Args:
            mask_tensor (torch.Tensor): 二值 Mask, Shape [B, 1, H, W], float32, 值 0.0/1.0

        Returns:
            pfm (torch.Tensor): 三值 Mask, Shape [B, H, W], dtype=torch.long
                                1: Lesion (病灶)
                                0: Surrounding (ROI 背景)
                                2: Ignore (忽略背景)
        """
        # 1. 维度与类型检查
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(1)  # [B, 1, H, W]

        # 2. 利用 MaxPool2d 实现膨胀 (Dilation)
        # kernel_size 必须是奇数，stride=1, padding=k//2 可保持尺寸不变
        padding = self.pfm_k // 2

        # MaxPool 对二值图操作：窗口内有 1 则输出 1 => 等效于膨胀
        dilated_mask = F.max_pool2d(
            mask_tensor,
            kernel_size=self.pfm_k,
            stride=1,
            padding=padding
        )

        # 3. 构建三值 Mask (完全在 GPU 上操作)
        # 初始化全为 2 (Ignore / 远离病灶的背景)
        # 使用 full_like 保持 device 一致
        pfm = torch.full_like(mask_tensor, 2.0)

        # 逻辑：
        #   a. 膨胀区域 (dilated == 1) 设为 0 (ROI Background / 关注的难例背景)
        #   b. 原始病灶 (original == 1) 设为 1 (Lesion / 正样本)

        # 步骤 a: 将膨胀区域设为 0
        # 注意: 使用 > 0.5 作为阈值来处理浮点精度
        pfm[dilated_mask > 0.5] = 0.0

        # 步骤 b: 将原始病灶区域设为 1 (这会覆盖掉部分步骤 a 的结果，即病灶中心)
        pfm[mask_tensor > 0.5] = 1.0

        # 4. 移除 Channel 维度并转为 Long 类型 (适配 CrossEntropyLoss)
        # [B, 1, H, W] -> [B, H, W]
        return pfm.squeeze(1).long()
    def process_batch(self, masks):
        """
        批处理辅助函数: 同时生成 RSM (Tensor) 和 PFM (Tensor)
        通常在 Dataset 的 __getitem__ 或 Collate_fn 中使用

        Args:
            masks (torch.Tensor): [B, 1, H, W] 二值标签
        """
        # 1. 生成 RSM (GPU/Tensor friendly)
        rsm_gt = self.generate_rsm(masks)

        # 2. 生成 PFM (CPU/Numpy friendly)
        # 因为 OpenCV 不支持 Tensor，需要转 numpy 处理再转回来
        pfm_list = []
        masks_np = masks.cpu().numpy()

        for i in range(masks_np.shape[0]):
            # 取出单张图 [1, H, W] -> [H, W]
            single_mask = masks_np[i, 0, :, :]
            pfm = self.generate_pfm(single_mask)
            pfm_list.append(torch.from_numpy(pfm))

        # 堆叠回 Tensor: [B, H, W] (注意 PFM 通常是 Long 类型用于 CrossEntropy)
        pfm_gt = torch.stack(pfm_list).long()

        return rsm_gt, pfm_gt


# --- 单元测试代码 ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. 模拟一个简单的 Ground Truth (中间有个 10x10 的病灶)
    H, W = 100, 100
    gt_mask = np.zeros((H, W), dtype=np.uint8)
    gt_mask[45:55, 45:55] = 1  # 中心方块病灶

    # 转 Tensor
    gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).float()

    # 2. 初始化生成器 (模拟 MA 任务参数)
    # RSM kernel = 35, PFM kernel = 17
    gen = LabelGenerator(rsm_kernel_size=35, pfm_kernel_size=17)

    # 3. 生成标签
    rsm_out = gen.generate_rsm(gt_tensor)
    pfm_out = gen.generate_pfm(gt_mask)

    # 4. 打印统计信息
    print("=== RSM (Probability Map) Stats ===")
    print(f"Shape: {rsm_out.shape}")
    print(f"Max Value (Density): {rsm_out.max().item():.4f}")  # 理论上应该是 100 / (35*35) ≈ 0.0816
    print(f"Is Differentiable? {rsm_out.requires_grad}")  # 应该是 False

    print("\n=== PFM (3-Class Mask) Stats ===")
    print(f"Shape: {pfm_out.shape}")
    unique, counts = np.unique(pfm_out, return_counts=True)
    print(f"Classes found: {unique}")  # 应该包含 0, 1, 2
    print(f"Pixel Counts: {dict(zip(unique, counts))}")

    # 5. 可视化 (保存图片以便查看)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original GT")
    plt.imshow(gt_mask, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Generated RSM (Blur/Heatmap)")
    plt.imshow(rsm_out.squeeze().numpy(), cmap='jet')

    plt.subplot(1, 3, 3)
    plt.title("Generated PFM (0=ROI, 1=Lesion, 2=Ignore)")
    # 为了显示清楚，把 2 映射成灰色，1 白色，0 黑色
    plt.imshow(pfm_out, cmap='gray', vmin=0, vmax=2)

    plt.tight_layout()
    plt.show()
    print("\nLabel Generator Test Passed!")