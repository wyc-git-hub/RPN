"""损失函数占位: L_per + L_ctr（带 mask 的 BCE）"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RPNLoss(nn.Module):
    """
    RPN 联合损失函数
    对应论文公式 (11): L = Sum(L_per) + L_ctr

    包含:
    1. Peripheral Loss: 针对 RSM 的 BCE Loss (支持多尺度/多分支)
    2. Central Loss: 针对 PFM 的 Masked BCE Loss (忽略 label=2 的区域)
    """

    def __init__(self):
        super(RPNLoss, self).__init__()
        # 普通的 BCE Loss，用于周边视觉分支
        # 注意: 如果输入是 Logits (未经过 Sigmoid)，需改用 BCEWithLogitsLoss
        # 但我们在 PeripheralVisionBranch 结尾已经加了 Sigmoid，所以这里用 BCELoss
        self.bce_loss = nn.BCELoss(reduction='mean')

        # 中央视觉分支输出的是 Logits，为了数值稳定性，推荐使用 BCEWithLogitsLoss
        # 我们会在 forward 中手动实现 masked 逻辑
        self.bce_logits_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, rsm_pred_list, rsm_gt, pfm_logits, pfm_gt):
        """
        Args:
            rsm_pred_list (list): 来自 PVB 的预测列表 [Tensor(B,1,H1,W1), Tensor(B,1,H2,W2)...]
                                  值范围 [0, 1] (经过 Sigmoid)
            rsm_gt (Tensor): RSM 真值 [B, 1, H, W]。由 LabelGenerator 生成
            pfm_logits (Tensor): 来自 CVB 的预测 [B, 1, H, W]。值范围 (-inf, inf) (未经过 Sigmoid)
            pfm_gt (Tensor): PFM 真值 [B, H, W]。值域 {0, 1, 2}
                             0: ROI背景, 1: 病灶, 2: 忽略

        Returns:
            total_loss: 总 Loss
            loss_dict: 各分项 Loss (用于日志记录)
        """

        # --- 1. 计算周边视觉损失 (L_per) ---
        # 对应论文: "L_per is binary cross-entropy loss"
        # 公式 (11) 中的求和部分

        # --- 修复核心: 维度检查与对齐 ---
        # 如果 rsm_gt 只有 3 维 [Batch, Height, Width]，
        # 我们手动给它加上 Channel 维，变成 [Batch, 1, Height, Width]
        if rsm_gt.dim() == 3:
            rsm_gt = rsm_gt.unsqueeze(1)
        l_per_total = 0.0

        # 遍历每一个周边分支的预测结果 (多尺度深度监督)
        for rsm_pred in rsm_pred_list:
            # 尺寸对齐: 网络的预测通常尺寸较小，需要上采样到 GT 的大小
            if rsm_pred.shape[2:] != rsm_gt.shape[2:]:
                rsm_pred_up = F.interpolate(
                    rsm_pred,
                    size=rsm_gt.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )
            else:
                rsm_pred_up = rsm_pred

            # 计算当前分支的 loss
            # rsm_gt 是 float 类型的概率密度图，BCE 依然适用 (相当于软标签)
            l_per = self.bce_loss(rsm_pred_up, rsm_gt)
            l_per_total += l_per

        # --- 2. 计算中央视觉损失 (L_ctr) ---
        # 对应论文: "L_ctr... pixels far from the lesion are ignored"

        # 2.1 创建掩码: 找出所有不等于 2 (忽略背景) 的像素
        # pfm_gt shape: [B, H, W] -> valid_mask shape: [B, H, W]
        valid_mask = (pfm_gt != 2)

        # 2.2 筛选有效像素
        # 注意 pfm_logits 是 [B, 1, H, W]，需要 squeeze 掉 channel 维才能和 pfm_gt 对齐
        pfm_logits_sq = pfm_logits.squeeze(1)  # [B, H, W]

        # 仅选取 mask 为 True 的位置
        # pred_valid shape: [N_valid_pixels]
        pred_valid = pfm_logits_sq[valid_mask]

        # target_valid shape: [N_valid_pixels]
        # 取出对应的标签，并转为 float 用于 BCE 计算 (0.0 或 1.0)
        target_valid = pfm_gt[valid_mask].float()

        # 2.3 计算 Masked BCE
        if len(target_valid) > 0:
            l_ctr = self.bce_logits_loss(pred_valid, target_valid)
        else:
            # 极少数情况：如果整张图都是 2 (理论上 LabelGenerator 不会产生这种情况)
            l_ctr = torch.tensor(0.0, device=pfm_logits.device, requires_grad=True)

        # --- 3. 总损失 ---
        # 对应公式 (11): 直接相加，论文中提到 "Since peripheral and central vision are equally important, weights... are not required"
        total_loss = l_per_total + l_ctr

        return total_loss, {
            "loss_per": l_per_total.item(),
            "loss_ctr": l_ctr.item(),
            "loss_total": total_loss.item()
        }


# --- 单元测试代码 ---
if __name__ == "__main__":
    # 模拟数据
    B, H, W = 2, 64, 64

    # 1. 模拟 PVB 输出 (两个分支，尺寸不同)
    pred_rsm_1 = torch.rand(B, 1, H // 4, W // 4)  # 深层特征，尺寸小
    pred_rsm_2 = torch.rand(B, 1, H // 2, W // 2)  # 浅层特征，尺寸中
    rsm_list = [pred_rsm_1, pred_rsm_2]

    # 2. 模拟 RSM GT (全尺寸)
    gt_rsm = torch.rand(B, 1, H, W)

    # 3. 模拟 CVB 输出 (Logits)
    pred_pfm = torch.randn(B, 1, H, W)

    # 4. 模拟 PFM GT (0, 1, 2)
    gt_pfm = torch.zeros(B, H, W).long()
    gt_pfm[:, 10:20, 10:20] = 1  # 病灶
    gt_pfm[:, 0:5, 0:5] = 2  # 忽略区域

    # 初始化 Loss
    criterion = RPNLoss()

    # 计算
    loss, loss_items = criterion(rsm_list, gt_rsm, pred_pfm, gt_pfm)

    print("=== Loss Module Test ===")
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Components: {loss_items}")

    # 反向传播测试
    loss.backward()
    print("Backward pass successful!")