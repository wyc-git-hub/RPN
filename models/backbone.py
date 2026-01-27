"""主干网络占位（U-Net Encoder-Decoder）"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    对应论文: "Each encoder block consists of two 3x3 convolutions" [cite: 158]
    我们通常加上 BN 和 ReLU 以加速收敛，并使用 padding=1 保持尺寸不变。
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Backbone(nn.Module):
    """
    对应论文 Section 3.1.1: Backbone
    结构: Encoder-Decoder with 5 encoder blocks and 4 decoder blocks.
    """

    def __init__(self, in_channels=3, base_channels=64):
        super(Backbone, self).__init__()

        # --- Encoder (下采样路径) [cite: 157-158] ---
        # 论文提到有 5 个 Encoder block
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        # 第5层是 Bottleneck (最底层)，没有 pooling 了
        self.enc5 = DoubleConv(base_channels * 8, base_channels * 16)

        # --- Decoder (上采样路径) [cite: 159-160] ---
        # 论文提到有 4 个 Decoder block，并且有 Skip Connections

        # Decoder 4 (对应 F_d4)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = DoubleConv(base_channels * 16 + base_channels * 8, base_channels * 8)

        # Decoder 3 (对应 F_d3)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = DoubleConv(base_channels * 8 + base_channels * 4, base_channels * 4)

        # Decoder 2 (对应 F_d2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DoubleConv(base_channels * 4 + base_channels * 2, base_channels * 2)

        # Decoder 1 (对应 F_d1)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(base_channels * 2 + base_channels, base_channels)

    def forward(self, x):
        # --- Encoding ---
        # x: [B, 3, H, W]
        e1 = self.enc1(x)  # F_e1: [B, 64, H, W]
        p1 = self.pool1(e1)  # [B, 64, H/2, W/2]

        e2 = self.enc2(p1)  # F_e2: [B, 128, H/2, W/2]
        p2 = self.pool2(e2)  # [B, 128, H/4, W/4]

        e3 = self.enc3(p2)  # F_e3: [B, 256, H/4, W/4]
        p3 = self.pool3(e3)  # [B, 256, H/8, W/8]

        e4 = self.enc4(p3)  # F_e4: [B, 512, H/8, W/8]
        p4 = self.pool4(e4)  # [B, 512, H/16, W/16]

        e5 = self.enc5(p4)  # F_e5 (Bottleneck): [B, 1024, H/16, W/16]

        # --- Decoding ---
        # 论文强调: "Use skip connections... fusing local and global features" [cite: 160]

        # Block 4
        d4 = self.up4(e5)
        # 遇到 DDR 数据集的尺寸问题 (1000不是32的倍数)，这里需要处理尺寸对齐
        if d4.size()[2:] != e4.size()[2:]:
            d4 = F.interpolate(d4, size=e4.size()[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)  # Skip Connection
        f_d4 = self.dec4(d4)  # Output F^o_d4

        # Block 3
        d3 = self.up3(f_d4)
        if d3.size()[2:] != e3.size()[2:]:
            d3 = F.interpolate(d3, size=e3.size()[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        f_d3 = self.dec3(d3)  # Output F^o_d3

        # Block 2
        d2 = self.up2(f_d3)
        if d2.size()[2:] != e2.size()[2:]:
            d2 = F.interpolate(d2, size=e2.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        f_d2 = self.dec2(d2)  # Output F^o_d2

        # Block 1
        d1 = self.up1(f_d2)
        if d1.size()[2:] != e1.size()[2:]:
            d1 = F.interpolate(d1, size=e1.size()[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        f_d1 = self.dec1(d1)  # Output F^o_d1

        # 关键点[cite: 161]:
        # 论文中提到需要获得 "two sets of features, F and F^o_di"
        # 这里的 f_d1, f_d2, f_d3, f_d4 就是 F^o_di，后续要喂给周边视觉分支

        # 返回字典，方便后续调用
        return {
            "f_d1": f_d1,  # [B, 64, H, W]
            "f_d2": f_d2,  # [B, 128, H/2, W/2]
            "f_d3": f_d3,  # [B, 256, H/4, W/4]
            "f_d4": f_d4,  # [B, 512, H/8, W/8]
            "bottleneck": e5
        }


# --- 测试代码 ---
if __name__ == "__main__":
    # DDR 数据集输入尺寸: 960x1000
    input_tensor = torch.randn(2, 3, 1000, 960)
    model = Backbone()
    outputs = model(input_tensor)

    print("DDR Dataset Input: (2, 3, 1000, 960)")
    for key, value in outputs.items():
        print(f"Output {key}: {value.shape}")