"""可视化工具占位: 绘制 RSM 热力图和分割结果"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


class VisualizationUtils:
    """
    RPN 可视化工具箱
    用于将 Tensor 转换为可读的图像、热力图和叠加图。
    """

    @staticmethod
    def denormalize(img_tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        """
        反归一化: 将预处理后的 Tensor (Normalize过) 还原回原始 RGB 图像
        Args:
            img_tensor: [3, H, W], value range usually [-1, 1] or standardized
        Returns:
            img_np: [H, W, 3], value range [0, 255], uint8
        """
        img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]

        # 反归一化: x = z * std + mean
        mean = np.array(mean)
        std = np.array(std)
        img = img * std + mean

        # 截断到 [0, 1] 并转为 0-255
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        return img

    @staticmethod
    def colorize_pfm(mask_np):
        """
        将 PFM (0, 1, 2) 转换为论文 Fig 3 风格的 RGB 图像
        0: ROI Background -> 黑色
        1: Lesion -> 白色
        2: Ignore -> 绿色 (Lime Green)
        """
        H, W = mask_np.shape
        color_mask = np.zeros((H, W, 3), dtype=np.uint8)

        # 1. Lesion (White)
        color_mask[mask_np == 1] = [255, 255, 255]

        # 2. ROI Background (Black) - 已经是0了，不用动

        # 3. Ignore (Green)
        # 使用一种显眼的绿色，类似论文中的配色
        color_mask[mask_np == 2] = [50, 205, 50]

        return color_mask

    @staticmethod
    def overlay_mask(image, mask, color=(0, 0, 255), alpha=0.4):
        """
        在原图上叠加分割掩模 (用于定性分析 Fig 7)
        Args:
            image: [H, W, 3] uint8
            mask: [H, W] binary mask (0 or 1)
            color: 叠加颜色 (B, G, R) 格式 (OpenCV 使用 BGR)
            alpha: 透明度
        """
        # 确保 mask 是二值的
        mask = (mask > 0.5).astype(np.uint8)

        # 寻找轮廓 (可选，画轮廓会让边界更清晰)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        overlay = image.copy()
        # 填充颜色
        overlay[mask == 1] = color

        # 混合
        output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # 画轮廓 (如果不想要轮廓可以注释掉)
        cv2.drawContours(output, contours, -1, color, 1)

        return output

    @staticmethod
    def plot_batch_results(images, rsm_gt, rsm_pred, pfm_gt, pfm_pred,
                           save_path=None, index=0):
        """
        绘制一个 Batch 的详细对比图 (5列展示)
        Column 1: 原始图像 (Original Image)
        Column 2: RSM 真值 (Ground Truth RSM - Heatmap)
        Column 3: RSM 预测 (Predicted RSM - Heatmap)
        Column 4: PFM 真值 (GT PFM - 3 Colors)
        Column 5: PFM 预测 (Pred Segmentation - Overlay on Image)
        """
        batch_size = images.shape[0]
        # 限制最多画 4 张，防止图片太大
        n_samples = min(batch_size, 4)

        fig, axes = plt.subplots(n_samples, 5, figsize=(20, 4 * n_samples))

        # 如果只有一行，axes 是一维数组，需要变为二维
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_samples):
            # 1. 准备数据
            img_rgb = VisualizationUtils.denormalize(images[i])

            # RSM 处理: 它是 Tensor [1, H, W]，需要转 numpy [H, W]
            rsm_g = rsm_gt[i].detach().cpu().squeeze().numpy()
            rsm_p = rsm_pred[i].detach().cpu().squeeze().numpy()

            # PFM GT 处理: Tensor [H, W] (int)
            pfm_g = pfm_gt[i].detach().cpu().numpy()

            # PFM Pred 处理: Tensor [1, H, W] (logits or prob) -> Binary Mask
            pfm_p = pfm_pred[i].detach().cpu().squeeze().numpy()
            pfm_p_bin = (pfm_p > 0.5).astype(np.uint8)  # 假设输入已经是 Sigmoid 后的概率

            # --- Column 1: Origin Image ---
            axes[i, 0].imshow(img_rgb)
            axes[i, 0].set_title("Input Image")
            axes[i, 0].axis('off')

            # --- Column 2: RSM GT (Heatmap) ---
            im2 = axes[i, 1].imshow(rsm_g, cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title("GT RSM (Peripheral)")
            axes[i, 1].axis('off')

            # --- Column 3: RSM Pred (Heatmap) ---
            im3 = axes[i, 2].imshow(rsm_p, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title("Pred RSM")
            axes[i, 2].axis('off')

            # --- Column 4: PFM GT (3-Class Style) ---
            # 使用 colorize_pfm 显示绿/黑/白
            pfm_vis = VisualizationUtils.colorize_pfm(pfm_g)
            axes[i, 3].imshow(pfm_vis)
            axes[i, 3].set_title("GT PFM (3-Class)")
            axes[i, 3].axis('off')

            # --- Column 5: Final Pred (Overlay) ---
            # 将预测的二值 Mask 叠加在原图上，类似 Fig 7
            overlay = VisualizationUtils.overlay_mask(img_rgb, pfm_p_bin, color=(255, 0, 0))  # 红色覆盖
            axes[i, 4].imshow(overlay)
            axes[i, 4].set_title("Final Segmentation")
            axes[i, 4].axis('off')

        plt.tight_layout()

        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()


# --- 单元测试代码 ---
if __name__ == "__main__":
    # 模拟数据
    B, H, W = 2, 256, 256

    # 1. 模拟图像 (Normalized)
    images = torch.randn(B, 3, H, W)

    # 2. 模拟 RSM (高斯模糊效果)
    # 创建一个中间亮四周暗的图
    y, x = np.ogrid[-H / 2:H / 2, -W / 2:W / 2]
    mask = np.exp(-(x ** 2 + y ** 2) / (2 * (30 ** 2)))
    rsm_gt = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
    # 预测稍微有点噪声
    rsm_pred = rsm_gt + 0.1 * torch.randn_like(rsm_gt)
    rsm_pred = torch.clamp(rsm_pred, 0, 1)

    # 3. 模拟 PFM GT (0, 1, 2)
    pfm_gt = torch.full((B, H, W), 2).long()  # 背景是 2 (绿色)
    pfm_gt[:, 100:150, 100:150] = 0  # ROI 背景是 0 (黑色)
    pfm_gt[:, 120:130, 120:130] = 1  # 病灶是 1 (白色)

    # 4. 模拟 PFM 预测 (概率)
    pfm_pred = torch.zeros(B, 1, H, W).float()
    pfm_pred[:, :, 120:130, 120:130] = 0.9  # 预测出病灶

    print("Generating visualization test...")
    VisualizationUtils.plot_batch_results(
        images, rsm_gt, rsm_pred, pfm_gt, pfm_pred,
        save_path="./results/test_visualization.png"
    )
    print("Saved 'test_visualization.png'. Please check the image.")