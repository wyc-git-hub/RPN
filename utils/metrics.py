"""评价指标占位: AUC-PR, F1, IoU 等"""

import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix


class MetricCalculator:
    """
    医学图像分割评价指标计算器
    对应论文 Section 4.2 [cite: 371-385]

    核心指标:
    1. AUC-PR (Area Under Precision-Recall Curve): 论文主指标，适合极度不平衡数据
    2. Dice Coefficient (F1 Score)
    3. IoU (Intersection over Union)
    4. AUC-ROC (Supplementary metric)
    """

    def __init__(self, threshold=0.5):
        """
        Args:
            threshold (float): 将概率转为二值 mask 的阈值，默认 0.5
        """
        self.threshold = threshold
        self.reset()

    def reset(self):
        """重置内部累加器"""
        self.results = {
            "auc_pr_list": [],
            "auc_roc_list": [],
            "dice_list": [],
            "iou_list": [],
            "precision_list": [],
            "recall_list": []
        }

    def update(self, pred_prob, target_binary):
        """
        更新一个 Batch 的指标

        Args:
            pred_prob (Tensor): 模型预测的概率图 (经过 Sigmoid), Shape [B, 1, H, W], 值域 [0, 1]
            target_binary (Tensor): 原始 GT, Shape [B, 1, H, W] 或 [B, H, W], 值域 {0, 1}
                                  注意：验证时请使用原始标签，不要使用带忽略区域(2)的 PFM
        """
        # 1. 数据预处理: 转 Numpy, 拉平, 确保是 CPU 浮点数
        # 这种计算通常不可导，所以 detach
        preds = pred_prob.detach().cpu().numpy()
        targets = target_binary.detach().cpu().numpy()

        # 兼容 [B, H, W] 的 target
        if targets.ndim == 3:
            targets = np.expand_dims(targets, axis=1)  # [B, 1, H, W]

        batch_size = preds.shape[0]

        # 2. 逐样本计算 (Per-image evaluation)
        # 医学图像通常是按张计算指标，然后求平均，而不是把所有像素混在一起算
        for i in range(batch_size):
            p = preds[i].flatten()  # 拉成一维向量 [N_pixels]
            t = targets[i].flatten().astype(int)  # 确保是整数 0/1

            # 极特殊情况处理：如果一张图全是背景 (没有病灶)，AUC-PR 未定义
            # 这种情况下，如果模型预测也全为0则满分，否则低分
            if np.sum(t) == 0:
                # 这里简单跳过或记为 NaN，视具体比赛规则而定
                # 为了训练稳定，我们暂且跳过无病灶图片的 AUC 计算，只算 False Positive 相关的
                continue

            # --- 核心指标: AUC-PR ---
            # sklearn 的 average_precision_score 计算的就是 AUC-PR
            auc_pr = average_precision_score(t, p)
            self.results["auc_pr_list"].append(auc_pr)

            # --- 辅助指标: AUC-ROC ---
            try:
                auc_roc = roc_auc_score(t, p)
                self.results["auc_roc_list"].append(auc_roc)
            except ValueError:
                pass  # 同样处理全0或全1的情况

            # --- 基于阈值的指标 (Dice, IoU, Precision, Recall) ---
            # 二值化
            p_bin = (p > self.threshold).astype(int)

            # 计算 TP, FP, FN
            # 技巧: TP = sum(p_bin * t)
            tp = np.sum(p_bin * t)
            fp = np.sum(p_bin * (1 - t))
            fn = np.sum((1 - p_bin) * t)

            # 加极小值 epsilon 防止除零
            eps = 1e-6

            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)

            # Dice (F1) = 2*TP / (2*TP + FP + FN)
            dice = (2 * tp) / (2 * tp + fp + fn + eps)

            # IoU = TP / (TP + FP + FN)
            iou = tp / (tp + fp + fn + eps)

            self.results["precision_list"].append(precision)
            self.results["recall_list"].append(recall)
            self.results["dice_list"].append(dice)
            self.results["iou_list"].append(iou)

    def compute(self):
        """
        返回当前累计的平均指标
        """
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