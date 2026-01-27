"""测试与推理脚本占位"""

import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- 导入项目模块 ---
from utils.config import get_config
from data.idrid_dataset import IDRiDDataset  # 如果是DDR，请改为 from data.ddr_dataset import DDRDataset
from data.transforms import Compose, Resize, HistogramEqualization, ToTensor
from models.rpn_net import RPN
from utils.metrics import MetricCalculator
from utils.visualization import VisualizationUtils
save_dir = "./experiments/logs"
logger = get_logger(save_dir, run_name="RPN_ResNet50")

def save_prediction(image, pred_prob, target, save_path, threshold=0.5):
    """
    保存单张测试结果对比图
    """
    # 1. 反归一化图像
    img_vis = VisualizationUtils.denormalize(image)  # [H, W, 3] RGB

    # 2. 处理预测结果 (Binary Mask)
    pred_mask = (pred_prob > threshold).astype(np.uint8)

    # 3. 处理真值 (Binary Mask)
    target_mask = target.astype(np.uint8)

    # 4. 绘制叠加图
    # 绿色 = Ground Truth, 红色 = Prediction, 黄色 = Overlap (TP)
    overlay = img_vis.copy()

    # 绘制 GT (Green)
    overlay[target_mask == 1] = [0, 255, 0]

    # 绘制 Pred (Red) - 混合一下以显示重叠
    # 为了显示清晰，我们用轮廓或者半透明
    # 这里使用简单的加权叠加
    heatmap = np.zeros_like(img_vis)
    heatmap[pred_mask == 1] = [255, 0, 0]  # Red

    # 简单的混合: Image + GT(Green) + Pred(Red)
    # 实际论文中常分别展示，这里为了省空间画在一起

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 图1: 原图 + GT (绿)
    vis_gt = VisualizationUtils.overlay_mask(img_vis, target_mask, color=(0, 255, 0), alpha=0.3)
    axes[0].imshow(vis_gt)
    axes[0].set_title("Ground Truth (Green)")
    axes[0].axis('off')

    # 图2: 原图 + Pred (红)
    vis_pred = VisualizationUtils.overlay_mask(img_vis, pred_mask, color=(255, 0, 0), alpha=0.3)
    axes[1].imshow(vis_pred)
    axes[1].set_title("Prediction (Red)")
    axes[1].axis('off')

    # 图3: 概率热力图
    im = axes[2].imshow(pred_prob, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title("Probability Map")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def test(model, loader, metric_calc, device, save_dir=None):
    model.eval()
    metric_calc.reset()

    print(f"Start Testing... saving results to {save_dir}")

    with torch.no_grad():
        pbar = tqdm(loader, desc="Testing", unit="img")
        for i, batch in enumerate(pbar):
            images = batch['image'].to(device)
            original_mask = batch['original_mask']  # [B, 1, H, W]
            img_names = batch['img_name']

            # 推理
            # 注意: RPN 在 eval 模式下只返回 pfm_prob (经过 Sigmoid 和上采样的)
            pfm_prob = model(images)

            # 更新指标
            metric_calc.update(pfm_prob, original_mask)

            # 保存可视化结果
            if save_dir:
                # 逐张保存
                batch_size = images.shape[0]
                preds_np = pfm_prob.cpu().numpy()  # [B, 1, H, W]
                targets_np = original_mask.numpy()  # [B, 1, H, W]

                for b in range(batch_size):
                    name = img_names[b]
                    save_name = os.path.splitext(name)[0] + "_res.png"
                    save_path = os.path.join(save_dir, save_name)

                    save_prediction(
                        images[b],
                        preds_np[b, 0],
                        targets_np[b, 0],
                        save_path
                    )

    return metric_calc.compute()


def main():
    parser = argparse.ArgumentParser(description='Test RPN')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--data_root', type=str, required=True, help='Path to Dataset root')
    parser.add_argument('--model_path', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--save_dir', type=str, default='./results', help='Dir to save test results')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()
    cfg = get_config(args)

    # 1. 设备与目录
    device = torch.device(cfg.device if cfg.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # 结果保存路径: ./results/MA
    result_dir = os.path.join(args.save_dir, cfg.lesion_type)
    os.makedirs(result_dir, exist_ok=True)

    print(f"=== Testing Task: {cfg.lesion_type} ===")
    print(f"Model: {args.model_path}")

    # 2. 数据准备 (Test Set)
    # 测试集只需要 Resize + Normalize，不需要翻转等增强
    test_transforms = Compose([
        Resize((cfg.input_size[0], cfg.input_size[1])),
        HistogramEqualization(clahe=True),
        ToTensor()
    ])

    # 使用 IDRiDDataset (mode='test')
    test_set = IDRiDDataset(
        root_dir=cfg.data_root,
        mode='test',  # 关键: 切换到测试集模式
        lesion_type=cfg.lesion_type,
        transforms=test_transforms,
        rsm_kernel_size=cfg.rsm_kernel,
        pfm_kernel_size=cfg.pfm_kernel
    )

    test_loader = DataLoader(
        test_set,
        batch_size=1,  # 测试时通常一张张测，方便保存图片
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # 3. 模型加载
    model = RPN(
        num_classes=1,
        backbone_base_channels=64,
        pvb_layer_indices=cfg.pvb_layer_indices,
        cvb_fusion_layers=cfg.cvb_fusion_layers
    ).to(device)

    # 加载权重
    print(f"Loading weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    # 兼容性处理: 如果保存的是 'model_state_dict' 或者是直接的 state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print("Model loaded successfully.")

    # 4. 执行测试
    metric_calc = MetricCalculator()
    metrics = test(model, test_loader, metric_calc, device, save_dir=result_dir)

    # 5. 输出结果
    print("\n" + "=" * 40)
    print(f"   Final Test Results ({cfg.lesion_type})")
    print("=" * 40)
    print(f"AUC-PR   : {metrics['auc_pr_list']:.4f}")
    print(f"Dice (F1): {metrics['dice_list']:.4f}")
    print(f"IoU      : {metrics['iou_list']:.4f}")
    print(f"Precision: {metrics['precision_list']:.4f}")
    print(f"Recall   : {metrics['recall_list']:.4f}")
    print("=" * 40)
    print(f"Visual results saved to: {result_dir}")


if __name__ == "__main__":
    main()