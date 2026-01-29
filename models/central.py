import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import SEBlock

class CentralConvBlock(nn.Module):
    """
    中央视觉分支专用的卷积块
    对应论文: "alternately uses convolution blocks (two 3x3 convolutions...)" 
    注意：论文文中提到 stride=2，但同时也强调 "without pooling and downsampling" 
    且目的是 "maintain high resolution" [cite: 182]。
    基于上下文逻辑冲突，这里采用 stride=1 以保持高分辨率，符合 PFM 精细分割的初衷。
    """
    def __init__(self, in_channels, out_channels):
        super(CentralConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import SEBlock


class CentralConvBlock(nn.Module):
    """
    中央视觉分支专用的卷积块
    """

    def __init__(self, in_channels, out_channels):
        super(CentralConvBlock, self).__init__()
        # 注意：这里 stride=1 保持高分辨率
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class LKA(nn.Module):
    """
    Large Kernel Attention 模块
    用来替代普通的 3x3 卷积，提供超大感受野和类似 Transformer 的注意力机制
    """

    def __init__(self, dim):
        super().__init__()
        # 1. 局部特征捕获 (5x5 Depth-wise)
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # 2. 长距离特征捕获 (7x7 Depth-wise, Dilation=3)
        # 感受野相当于 7 + (7-1)*(3-1) + ... 非常大
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)

        # 3. 通道融合 (1x1 Point-wise)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        # 这里的注意力是 "Gate" 机制：原特征 * 注意力图
        return u * attn


class AdvancedCentralBlock(nn.Module):
    """
    升级版的 CVB Block
    结构: Conv1x1(降维) -> LKA(大核注意) -> Conv1x1(升维/融合) -> DropPath
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, out_channels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(out_channels)
        self.proj_2 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)  # 医学图像推荐用 InstanceNorm 或 BN

    def forward(self, x):
        shorcut = x.clone()
        # 如果输入输出通道不一致，Shortcut 需要投影
        if x.shape[1] != self.proj_2.out_channels:
            shorcut = self.proj_1(x)

        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)  # LKA 处理
        x = self.proj_2(x)
        x = self.bn(x)
        return x + shorcut  # 残差连接

class CentralVisionBranch(nn.Module):
    """
    中央视觉分支 (Central Vision Branch)
    [cite_start]修改说明: 将特征融合方式从 "相加" 改为 "拼接" (Concat) [cite: 1681]
    """

    def __init__(self, backbone_channels_list, cvb_channels=64):
        """
        Args:
            backbone_channels_list: Backbone Decoder 层输出通道列表 [64, 128...]
            cvb_channels: 中央分支内部维持的基础通道数
        """
        super(CentralVisionBranch, self).__init__()

        # 1. 初始处理层
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, cvb_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cvb_channels),
            nn.ReLU(inplace=True)
        )

        # 2. 动态构建融合阶段
        self.stages = nn.ModuleList()
        self.projectors = nn.ModuleList()
        self.se_blocks = nn.ModuleList()

        for i, bb_dim in enumerate(backbone_channels_list):
            # [修改关键点 A] 调整卷积块输入通道
            # 第1个 Stage 输入来自 init_conv (64)
            # 后续 Stage 输入来自上一个 SE Block (由于拼接，它是 64+64=128)
            in_c = cvb_channels if i == 0 else cvb_channels * 2

            # ConvBlock 负责特征提取，并将通道数从 128 降回 64 (Bottleneck)
            # self.stages.append(CentralConvBlock(in_c, cvb_channels))

            # [修改点]: 使用 LKA 模块替代普通卷积
            # 这种结构能更好地处理拼接进来的深层语义特征
            self.stages.append(AdvancedCentralBlock(in_c, cvb_channels))

            # 投影层: 将 Backbone 特征统一投影到 cvb_channels (64)
            self.projectors.append(nn.Conv2d(bb_dim, cvb_channels, kernel_size=1))

            # [修改关键点 B] 调整 SE Block 输入通道
            # 输入是 (Stage输出 64) 拼接 (Projector输出 64) = 128
            self.se_blocks.append(SEBlock(cvb_channels * 2))

        # 3. 最终预测层
        # 输入是最后一个 SE Block 的输出 (128)
        # self.final_conv = nn.Conv2d(cvb_channels * 2, 1, kernel_size=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(cvb_channels * 2, cvb_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cvb_channels, 1, kernel_size=1)
        )
    def forward(self, img, rsm_pred, backbone_features_list):
        # ... (RSM 加权部分保持不变) ...
        if rsm_pred.shape[2:] != img.shape[2:]:
            rsm_up = F.interpolate(rsm_pred, size=img.shape[2:], mode='bilinear', align_corners=True)
        else:
            rsm_up = rsm_pred
        weighted_img = img * rsm_up

        x = self.init_conv(weighted_img)

        # 级联融合循环
        for i, (stage_conv, projector, se) in enumerate(zip(self.stages, self.projectors, self.se_blocks)):
            # 1. 卷积处理 (如果不是第一层，这里会先将通道压缩回 64)
            x = stage_conv(x)

            # 2. 融合 Backbone 特征
            if i < len(backbone_features_list):
                bb_feat = backbone_features_list[i]
                if bb_feat.shape[2:] != x.shape[2:]:
                    bb_feat = F.interpolate(bb_feat, size=x.shape[2:], mode='bilinear', align_corners=True)

                # 投影到 64 通道
                feat_proj = projector(bb_feat)

                # [修改关键点 C] 使用拼接 (Concat) 代替相加
                # x (64) + feat_proj (64) -> (128)
                x = torch.cat([x, feat_proj], dim=1)

            # 3. SE Block 增强 (处理 128 通道的特征)
            x = se(x)

        out = self.final_conv(x)
        return out
# --- 简单的集成测试 ---
if __name__ == "__main__":
    # 模拟输入
    batch_size = 2
    H, W = 640, 640
    img = torch.randn(batch_size, 3, H, W)
    
    # 模拟 RSM (假设来自 PVB，尺寸较小)
    rsm_pred = torch.rand(batch_size, 1, H//8, W//8)
    
    # 模拟 Backbone 特征 (假设融合 f_d1 和 f_d2)
    # f_d1 通常和原图一样大 (或 H/2)，f_d2 是 H/2
    feat_d1 = torch.randn(batch_size, 64, H, W)
    feat_d2 = torch.randn(batch_size, 128, H//2, W//2)
    
    backbone_feats = [feat_d1, feat_d2]
    backbone_channels = [64, 128]
    
    # 初始化 CVB
    cvb = CentralVisionBranch(backbone_channels_list=backbone_channels, cvb_channels=32)
    
    # 前向传播
    pfm_logits = cvb(img, rsm_pred, backbone_feats)
    
    print(f"Weighted Input Created internally using RSM.")
    print(f"Output PFM Logits Shape: {pfm_logits.shape}")
    
    # 验证尺寸是否保持高分辨率=-068778878788
    assert pfm_logits.shape == (batch_size, 1, H, W)
    print("CVB module test passed!")