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

class CentralVisionBranch(nn.Module):
    """
    中央视觉分支 (Central Vision Branch)
    对应论文 Section 3.1.3 及 Figure 1 右侧部分
    """
    def __init__(self, backbone_channels_list, cvb_channels=64):
        """
        Args:
            backbone_channels_list (list): Backbone 各个 Decoder 层输出的通道数列表
                                           例如 [64, 128] (对应 f_d1, f_d2)
            cvb_channels (int): 中央视觉分支内部维持的通道数 (通常较小以节省显存)
        """
        super(CentralVisionBranch, self).__init__()
        
        # --- 1. 初始处理层 ---
        # 输入是 3通道原图 (Weighted Image)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, cvb_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cvb_channels),
            nn.ReLU(inplace=True)
        )
        
        # --- 2. 动态构建融合阶段 (Fusion Stages) ---
        # 对应 Figure 1 右侧的级联结构：ConvBlock -> Add(Backbone) -> SEBlock
        self.stages = nn.ModuleList()
        self.projectors = nn.ModuleList() # 用于匹配 Backbone 通道数到 CVB 通道数
        self.se_blocks = nn.ModuleList()
        
        # 假设我们融合 Backbone 的前 N 层特征 (由 list 长度决定)
        for bb_dim in backbone_channels_list:
            # 卷积块 (提取特征)
            self.stages.append(CentralConvBlock(cvb_channels, cvb_channels))
            
            # 投影层 (Backbone Channel -> CVB Channel)
            # 对应 Figure 1 中的 (+) 操作前的通道对齐
            self.projectors.append(nn.Conv2d(bb_dim, cvb_channels, kernel_size=1))
            
            # SE Block (特征增强)
            self.se_blocks.append(SEBlock(cvb_channels))
            
        # --- 3. 最终预测层 ---
        # 输出像素级 PFM (Pixel-level Focal Mask)
        # 对应公式 (5) 后描述: "A pixel-level response map O_c is output" [cite: 184]
        self.final_conv = nn.Conv2d(cvb_channels, 1, kernel_size=1)

    def forward(self, img, rsm_pred, backbone_features_list):
        """
        Args:
            img: 原始眼底图像 [B, 3, H, W]
            rsm_pred: 周边分支预测的 RSM [B, 1, h, w] (尺寸可能较小)
            backbone_features_list: Backbone 解码器特征列表 [f_d1, f_d2...]
                                    顺序应与 init 中的 backbone_channels_list 对应
        """
        # --- Step 1: 视觉聚焦 (Attention Weighting) ---
        # 对应公式 (5): I' = I * RSM 
        # 必须先把 RSM 上采样到原图大小
        if rsm_pred.shape[2:] != img.shape[2:]:
            rsm_up = F.interpolate(rsm_pred, size=img.shape[2:], mode='bilinear', align_corners=True)
        else:
            rsm_up = rsm_pred
            
        # 生成加权图像 (New Input Image)
        weighted_img = img * rsm_up
        
        # --- Step 2: 初始特征提取 ---
        x = self.init_conv(weighted_img)
        
        # --- Step 3: 级联融合 (Iterative Fusion) ---
        # 对应 Figure 1 右侧的流程: 
        # Conv -> Add(Backbone) -> SE -> Conv ...
        for i, (stage_conv, projector, se) in enumerate(zip(self.stages, self.projectors, self.se_blocks)):
            
            # 3.1 卷积处理
            x = stage_conv(x)
            
            # 3.2 融合 Backbone 特征 (Skip Connection) 
            if i < len(backbone_features_list):
                bb_feat = backbone_features_list[i]
                
                # 调整尺寸 (Backbone 特征可能尺寸较小，需要上采样)
                if bb_feat.shape[2:] != x.shape[2:]:
                    bb_feat = F.interpolate(bb_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
                
                # 调整通道 (Project) 并相加
                # 对应 Figure 1 中的 (+)
                feat_proj = projector(bb_feat)
                x = x + feat_proj
            
            # 3.3 SE Block 增强
            x = se(x)
            
        # --- Step 4: 输出 PFM ---
        # 注意：这里输出的是 Logits，不需要 Sigmoid
        # 因为在 Loss 计算时通常使用 BCEWithLogitsLoss 或者手动 Sigmoid
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
    
    # 验证尺寸是否保持高分辨率
    assert pfm_logits.shape == (batch_size, 1, H, W)
    print("CVB module test passed!")