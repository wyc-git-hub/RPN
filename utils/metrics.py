"""评价指标占位: AUC-PR, F1, IoU 等"""

import torch
import numpy as np

"""
优化后的评价指标计算器 (完全基于 GPU)
解决 CPU 瓶颈，加速验证过程
"""


class MetricCalculator:
    """
    医学图像分割评价指标计算器 (GPU 加速版)
    对应论文 Section 4.2: AUC-PR, Dice, IoU
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.results = {
            "auc_pr_list": [],
            "dice_list": [],
            "iou_list": [],
            "precision_list": [],
            "recall_list": []
        }

    def compute_auc_pr_gpu(self, prob, target):
        """
        在 GPU 上计算单张图片的 AUC-PR (Average Precision)
        Args:
            prob: (N,) Tensor, 0-1 概率
            target: (N,) Tensor, 0/1 标签
        """
        # 1. 过滤掉忽略区域 (如果 target 中有非 0/1 的值，这里假设输入已经是 binary)
        # 排序 (降序)
        indices = torch.argsort(prob, descending=True)
        prob = prob[indices]
        target = target[indices]

        # 2. 计算累计 TP 和 FP
        # distinct_value_indices 用于处理概率相同的情况 (Ties)
        # 但为了性能，通常直接计算即可，误差极小

        # 累计 True Positives
        tps = torch.cumsum(target, dim=0)
        # 总 Positives
        total_pos = tps[-1]

        # 如果没有正样本，AUC-PR 未定义 (或视任务定义为 0/1)
        if total_pos == 0:
            return None

            # 3. 计算 Precision 和 Recall
        # fps = (index + 1) - tps
        fps = torch.arange(1, len(target) + 1, device=target.device) - tps

        precision = tps / (tps + fps + 1e-8)
        recall = tps / (total_pos + 1e-8)

        # 4. 计算 Average Precision (Sum(R_n - R_n-1) * P_n)
        # 添加 R=0, P=1 的起始点 (可选，sklearn 的实现方式略有不同，这里采用标准积分近似)

        # 简化的积分计算 (Right-Riemann sum，近似 sklearn 的 average_precision)
        recall_diff = torch.cat([recall[0:1], recall[1:] - recall[:-1]])
        ap = torch.sum(precision * recall_diff)

        return ap.item()

    def update(self, pred_prob, target_binary):
        """
        更新 Batch 指标 (全部在 GPU 上进行)
        Args:
            pred_prob: [B, 1, H, W] (Float, 0-1)
            target_binary: [B, 1, H, W] (Float, 0/1)
        """
        # 确保数据在 GPU
        # Flatten: [B, N_pixels]
        B = pred_prob.size(0)
        preds = pred_prob.view(B, -1)
        targets = target_binary.view(B, -1)

        for i in range(B):
            p = preds[i]
            t = targets[i]

            # 1. 检查是否有正样本
            if torch.sum(t) == 0:
                # 对应论文/竞赛常见做法：
                # 如果 Ground Truth 全黑，预测也全黑则满分，否则扣分
                # 但 AUC-PR 在此情况未定义，通常跳过不计入 Mean
                continue

            # 2. 计算 AUC-PR (GPU)
            auc_pr = self.compute_auc_pr_gpu(p, t)
            if auc_pr is not None:
                self.results["auc_pr_list"].append(auc_pr)

            # 3. 计算 Dice, IoU, Precision, Recall (GPU)
            # 二值化
            p_bin = (p > self.threshold).float()

            tp = (p_bin * t).sum()
            fp = (p_bin * (1 - t)).sum()
            fn = ((1 - p_bin) * t).sum()

            eps = 1e-6
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            dice = (2 * tp) / (2 * tp + fp + fn + eps)
            iou = tp / (tp + fp + fn + eps)

            self.results["precision_list"].append(precision.item())
            self.results["recall_list"].append(recall.item())
            self.results["dice_list"].append(dice.item())
            self.results["iou_list"].append(iou.item())

    def compute(self):
        metrics = {}
        for key, value_list in self.results.items():
            if len(value_list) > 0:
                metrics[key] = np.mean(value_list)
            else:
                metrics[key] = 0.0
        return metrics

# --- 单元测试代码 ---
if __name__ == "__main__":
    # 模拟数据
    B, H, W = 2, 100, 100

    # 1. 模拟预测 (概率图)
    # 假设模型对病灶预测得不错，但也有些噪声
    pred = torch.zeros(B, 1, H, W).float()
    pred[:, :, 40:60, 40:60] = 0.9  # 预测的高置信度区域
    pred[:, :, 10:20, 10:20] = 0.3  # 噪声

    # 2. 模拟真值 (原始 Binary GT)
    target = torch.zeros(B, 1, H, W).float()
    target[:, :, 45:55, 45:55] = 1  # 真实的病灶 (比预测的小一点)

    # 初始化计算器
    metric_engine = MetricCalculator(threshold=0.5)

    # 更新
    metric_engine.update(pred, target)

    # 获取结果
    final_metrics = metric_engine.compute()

    print("=== Metrics Test ===")
    print(f"AUC-PR:    {final_metrics['auc_pr_list']:.4f} (Paper primary metric)")
    print(f"Dice (F1): {final_metrics['dice_list']:.4f}")
    print(f"IoU:       {final_metrics['iou_list']:.4f}")
    print(f"Precision: {final_metrics['precision_list']:.4f}")
    print(f"Recall:    {final_metrics['recall_list']:.4f}")

    # 简单验证逻辑
    # 预测区域 20x20=400像素，真值 10x10=100像素
    # TP=100, FP=300 (预测大了), FN=0
    # Precision = 100/400 = 0.25
    # Recall = 100/100 = 1.0
    # IoU = 100 / (100+300) = 0.25
    # Dice = 200 / (200+300) = 0.4
    # 看看程序输出是否接近