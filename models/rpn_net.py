"""RPN 模型组装占位"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入之前实现的模块
from models.backbone import Backbone
from models.peripheral import PeripheralVisionBranch
from models.central import CentralVisionBranch


class RPN(nn.Module):
    """
    RPN: A region-to-pixel-mask-based convolutional network
    对应论文 Figure 1 的整体架构
    """

    def __init__(self,
                 num_classes=1,
                 backbone_base_channels=64,
                 pvb_layer_indices=['f_d4', 'f_d3', 'f_d2', 'f_d1'],  # 默认挂载在 decoder 第2、3层
                 cvb_fusion_layers=['f_d4', 'f_d3', 'f_d2', 'f_d1'],  # CVB 融合第1、2层特征
                 cvb_internal_channels=32):
        """
        Args:
            num_classes: 输出类别数 (二分类通常为1)
            backbone_base_channels: Backbone 基础通道数
            pvb_layer_indices: 列表，指定在 Backbone 的哪些层挂载周边视觉分支
                               可选值: 'f_d1', 'f_d2', 'f_d3', 'f_d4'
            cvb_fusion_layers: 列表，指定中央视觉分支融合哪些层的 Backbone 特征
        """
        super(RPN, self).__init__()

        # --- 1. 初始化 Backbone ---
        # 对应图左侧的 Encoder-Decoder
        self.backbone = Backbone(in_channels=3, base_channels=backbone_base_channels)

        # 定义 Backbone 各层的通道数 (基于 U-Net 结构: 64, 128, 256, 512)
        # f_d1: 64, f_d2: 128, f_d3: 256, f_d4: 512
        self.feature_channels = {
            'f_d1': backbone_base_channels,
            'f_d2': backbone_base_channels * 2,
            'f_d3': backbone_base_channels * 4,
            'f_d4': backbone_base_channels * 8
        }

        # --- 2. 初始化周边视觉分支 (PVBs) ---
        # 对应图中间的蓝色区域
        # 使用 ModuleDict 可以通过名字 ('f_d2') 方便地管理多个分支
        self.pvb_layer_indices = pvb_layer_indices
        self.pvbs = nn.ModuleDict()

        for layer_name in pvb_layer_indices:
            in_c = self.feature_channels[layer_name]
            # 实例化一个 PVB 并存入字典
            self.pvbs[layer_name] = PeripheralVisionBranch(in_channels=in_c)

        # --- 3. 初始化中央视觉分支 (CVB) ---
        # 对应图右侧的橙色区域
        self.cvb_fusion_layers = cvb_fusion_layers

        # 准备传给 CVB 的通道列表 (告诉 CVB 每一层进来多少通道，方便它做投影)
        fusion_channels_list = [self.feature_channels[name] for name in cvb_fusion_layers]

        self.cvb = CentralVisionBranch(
            backbone_channels_list=fusion_channels_list,
            cvb_channels=cvb_internal_channels
        )

    def forward(self, x):
        """
        x: [Batch, 3, H, W] 原始眼底图像
        """
        input_size = x.size()[2:]

        # --- Step 1: Backbone 特征提取 ---
        # 返回字典: {'f_d1': ..., 'f_d2': ..., ...}
        feats = self.backbone(x)

        # --- Step 2: 周边视觉分支预测 (Multi-scale RSM) ---
        rsm_outputs = []  # 存储所有 PVB 的预测结果，用于计算 Loss

        # 用于中央视觉的“聚光灯”RSM
        # 策略：通常取最深层或者指定的某一层的 RSM 作为 Attention Map
        # 这里默认取 pvb_layer_indices 列表里最后一个层级的输出 (例如 f_d3)
        attention_rsm = None

        for layer_name in self.pvb_layer_indices:
            # 拿到对应层的特征
            feat = feats[layer_name]
            # 通过 PVB
            rsm_pred = self.pvbs[layer_name](feat)
            rsm_outputs.append(rsm_pred)

            # 更新 attention_rsm (取循环的最后一个)
            attention_rsm = rsm_pred

        # --- Step 3: 中央视觉分支预测 (PFM) ---
        # 准备融合所需的 Backbone 特征列表
        fusion_feats = [feats[name] for name in self.cvb_fusion_layers]

        # 注意: attention_rsm 需要是 [B, 1, H, W] 大小才能和原图相乘
        # PVB 输出的可能是小尺寸，CVB 内部会处理上采样，这里直接传进去
        pfm_logits = self.cvb(x, attention_rsm, fusion_feats)

        # --- Step 4: 返回结果 ---
        # 训练模式：返回 (rsm_list, pfm_logits) 以便计算 Loss
        # 推理模式：通常只需要 pfm_logits (或者经过 Sigmoid 的结果)
        if self.training:
            return rsm_outputs, pfm_logits
        else:
            # 推理时，为了方便，可以顺便把 mask 归一化并上采样到原图尺寸
            pfm_prob = torch.sigmoid(pfm_logits)
            # 如果尺寸不对，插值回原图
            if pfm_prob.size()[2:] != input_size:
                pfm_prob = F.interpolate(pfm_prob, size=input_size, mode='bilinear', align_corners=True)
            return pfm_prob


# --- 完整性自测 ---
if __name__ == "__main__":
    # 模拟 DDR 输入
    x = torch.randn(2, 3, 640, 640)  # 为了快速测试用小尺寸，实际DDR用 960x1000

    # 按照论文常见配置初始化
    # PVB 挂在 d2, d3; CVB 融合 d1, d2
    model = RPN(
        backbone_base_channels=64,
        pvb_layer_indices=['f_d2', 'f_d3'],
        cvb_fusion_layers=['f_d1', 'f_d2']
    )

    # 前向传播 (默认 training=True)
    rsm_list, pfm_out = model(x)

    print("=== RPN Architecture Test ===")
    print(f"Input: {x.shape}")
    print(f"Num PVB branches: {len(rsm_list)}")
    for i, rsm in enumerate(rsm_list):
        print(f"  RSM_{i} output shape: {rsm.shape}")

    print(f"PFM output shape: {pfm_out.shape}")

    # 简单验证
    assert len(rsm_list) == 2
    # pfm_out 是 logits，形状应为 [B, 1, 640, 640]
    assert pfm_out.shape == (2, 1, 640, 640)
    print("RPN Integration Successful!")