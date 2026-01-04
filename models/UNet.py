import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 基础卷积块
# =========================================================
def conv_block(in_ch, out_ch):
    """
    标准 3x3 卷积块
    用于 encoder / decoder
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


# =========================================================
# Bottleneck：带空洞卷积的全局几何感知模块
# =========================================================
class DilatedBottleneck(nn.Module):
    """
    通过多尺度空洞卷积扩大感受野
    让模型在最低分辨率下感知“整体几何是否存在”
    """
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch, ch, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch, ch, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# =========================================================
# 结构增强版 UNet（适配剖面线 SDF 插值）
# =========================================================
class UNet(nn.Module):
    """
    
    1. depth = 5（更强全局感受野）
    2. base_channels = 64（几何表达能力）
    3. bottleneck 使用 dilated conv（全局几何）
    4. skip connection channel reduction（防止复制输入）
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_channels=64,
    ):
        super().__init__()

        # -----------------------------
        # Encoder（下采样路径）
        # -----------------------------
        self.enc1 = conv_block(in_channels, base_channels)          # 512 x 512
        self.enc2 = conv_block(base_channels, base_channels * 2)    # 256 x 256
        self.enc3 = conv_block(base_channels * 2, base_channels * 4)# 128 x 128
        self.enc4 = conv_block(base_channels * 4, base_channels * 8)# 64  x 64
        self.enc5 = conv_block(base_channels * 8, base_channels * 8)# 32  x 32

        self.pool = nn.MaxPool2d(2)

        # -----------------------------
        # Bottleneck（最低分辨率，全局几何）
        # -----------------------------
        self.bottleneck = DilatedBottleneck(base_channels * 8)

        # -----------------------------
        # Skip connection channel reduction
        # 防止 decoder 直接“抄输入结构”
        # -----------------------------
        self.skip5_reduce = nn.Conv2d(base_channels * 8, base_channels * 4, 1)
        self.skip4_reduce = nn.Conv2d(base_channels * 8, base_channels * 4, 1)
        self.skip3_reduce = nn.Conv2d(base_channels * 4, base_channels * 2, 1)
        self.skip2_reduce = nn.Conv2d(base_channels * 2, base_channels, 1)
        self.skip1_reduce = nn.Conv2d(base_channels, base_channels // 2, 1)

        # -----------------------------
        # Decoder（上采样路径）
        # -----------------------------
        self.up5 = nn.ConvTranspose2d(base_channels * 8, base_channels * 8, 2, stride=2)
        self.dec5 = conv_block(base_channels * 12, base_channels * 8)

        self.up4 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec4 = conv_block(base_channels * 8, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec3 = conv_block(base_channels * 4, base_channels * 2)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec2 = conv_block(base_channels * 2, base_channels)

        self.up1 = nn.ConvTranspose2d(base_channels, base_channels // 2, 2, stride=2)
        self.dec1 = conv_block(base_channels, base_channels // 2)

        # -----------------------------
        # 输出层（不加激活，直接回归 SDF）
        # -----------------------------
        self.out_conv = nn.Conv2d(base_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        # -------- Encoder --------
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        # -------- Bottleneck --------
        b = self.bottleneck(self.pool(e5))

        # -------- Decoder --------
        d5 = self.up5(b)
        s5 = self.skip5_reduce(e5)
        d5 = self.dec5(torch.cat([d5, s5], dim=1))

        d4 = self.up4(d5)
        s4 = self.skip4_reduce(e4)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        d3 = self.up3(d4)
        s3 = self.skip3_reduce(e3)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = self.up2(d3)
        s2 = self.skip2_reduce(e2)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self.up1(d2)
        s1 = self.skip1_reduce(e1)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        # -------- Output --------
        # 输出为 SDF（正负值均允许）
        return self.out_conv(d1)
