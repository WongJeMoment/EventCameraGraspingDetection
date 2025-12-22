# LAG_Block.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Ghost Convolution
# -------------------------
class GhostConv(nn.Module):
    """
    Ghost Convolution (简化实现)
    先用主卷积生成 intrinsic feature，再用 depthwise 生成 ghost feature，最后 concat。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        ratio: int = 2,
        dw_kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        use_bn: bool = True,
        act: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        init_channels = int(math.ceil(out_channels / ratio))
        new_channels = out_channels - init_channels

        self.primary = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(init_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True) if act else nn.Identity(),
        )

        self.cheap = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_kernel_size, 1, dw_kernel_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True) if act else nn.Identity(),
        ) if new_channels > 0 else nn.Identity()

        self.out_channels = out_channels
        self.init_channels = init_channels
        self.new_channels = new_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary(x)
        if self.new_channels > 0:
            x2 = self.cheap(x1)
            out = torch.cat([x1, x2], dim=1)
        else:
            out = x1
        return out[:, : self.out_channels, :, :]


# -------------------------
# ECA Attention
# -------------------------
class ECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA) 简化实现：
    GAP -> 1D conv -> sigmoid -> channel scale
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        y = self.avg_pool(x)  # (B,C,1,1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B,1,C)
        y = self.conv1d(y)                   # (B,1,C)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B,C,1,1)
        return x * y


# -------------------------
# Coordinate Attention
# -------------------------
class CoordAtt(nn.Module):
    """
    Coordinate Attention (简化实现)
    参考：沿 H/W 分别池化，拼接后降维，再生成 a_h / a_w。
    """
    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mip = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # Pool along width -> (B,C,H,1)
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        # Pool along height -> (B,C,1,W)
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)  # (B,C,W,1)

        y = torch.cat([x_h, x_w], dim=2)  # (B,C,H+W,1)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (B,mip,1,W)

        a_h = self.sigmoid(self.conv_h(x_h))  # (B,C,H,1)
        a_w = self.sigmoid(self.conv_w(x_w))  # (B,C,1,W)

        return x * a_h * a_w


# -------------------------
# LAG-Block = Ghost + Ghost + ECA + CoordAtt
# -------------------------
class LAGBlock(nn.Module):
    """
    按图：GhostConv -> GhostConv -> ECA -> Coordinate Attention
    输入输出通道数保持不变（也可以改成可变）
    """
    def __init__(self, channels: int):
        super().__init__()
        self.g1 = GhostConv(channels, channels, kernel_size=3, ratio=2)
        self.g2 = GhostConv(channels, channels, kernel_size=3, ratio=2)
        self.eca = ECABlock(channels, k_size=3)
        self.ca = CoordAtt(channels, reduction=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.g1(x)
        y = self.g2(y)
        y = self.eca(y)
        y = self.ca(y)
        return y
