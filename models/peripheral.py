"""周边视觉分支（PVB）占位"""

import torch
import torch.nn as nn
from models.blocks import SEBlock  # 假设你把之前的 SEBlock 放在了 models/blocks.py


class PeripheralVisionBranch(nn.Module):
    """
    周边视觉分支 (Peripheral Vision Branch)
    对应论文 Section 3.1.2 及公式 (1)-(4)

    功能:
    接收 Backbone 的中间层特征 F_di，通过 SE Block 增强特征，
    最后压缩成单通道的概率图 (RSM Prediction)。
    """

    def __init__(self, in_channels, reduction=16):
        """
        Args:
            in_channels (int): 输入特征图的通道数 (即 Backbone 对应层的输出通道)
            reduction (int): SE Block 的降维系数
        """
        super(PeripheralVisionBranch, self).__init__()

        # --- 1. 特征重校准 (Feature Recalibration) ---
        # 对应论文: "F^o_d2 passes through a squeeze-and-excitation (SE) block"
        # 实现公式 (1), (2), (3)
        self.se_block = SEBlock(in_channels, reduction=reduction)

        # --- 2. 信息聚合与预测 (Aggregation & Prediction) ---
        # 对应论文公式 (4): O_ri = Sigma(W_Ri * F_tilde) [cite: 174]
        # 使用 1x1 卷积将多通道特征聚合为 1 个通道
        self.conv_out = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

        # --- 3. 概率激活 (Probability Activation) ---
        # 对应论文公式 (4) 中的 Sigma 符号
        # RSM 是一个概率分布图 (0~1)，所以必须用 Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: 来自 Backbone 解码器的特征图 F_di^o
           Shape: [Batch, C, H, W]
        """
        # Step 1: SE Block 增强特征 (公式 3)
        # 重要的病灶特征通道被放大，背景通道被抑制
        x_se = self.se_block(x)

        # Step 2: 1x1 卷积聚合 (公式 4 内部)
        # Shape: [Batch, C, H, W] -> [Batch, 1, H, W]
        x_conv = self.conv_out(x_se)

        # Step 3: 生成概率图 (公式 4 外部)
        # 输出范围限制在 [0, 1]
        rsm_pred = self.sigmoid(x_conv)

        return rsm_pred


# --- 简单的集成测试 ---
if __name__ == "__main__":
    # 模拟 Backbone 输出的特征图 (例如 F_d2, 128通道)
    feature_map = torch.randn(2, 128, 64, 64)

    # 初始化 PVB
    pvb = PeripheralVisionBranch(in_channels=128)

    # 前向传播
    rsm = pvb(feature_map)

    print(f"Input Feature Shape: {feature_map.shape}")
    print(f"Output RSM Shape:    {rsm.shape}")  # 应该是 [2, 1, 64, 64]

    # 验证输出值范围
    print(f"Max value: {rsm.max().item()}")
    print(f"Min value: {rsm.min().item()}")
    assert rsm.shape == (2, 1, 64, 64)
    assert 0 <= rsm.min() and rsm.max() <= 1
    print("PVB module test passed!")

