# model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from LAGNetwork.LAGBlock import LAGBlock


def conv_bn_relu(in_ch: int, out_ch: int, k: int = 3, s: int = 2, p: int | None = None) -> nn.Sequential:
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def deconv_bn_relu(in_ch: int, out_ch: int, k: int = 3, s: int = 2, p: int = 1, op: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, output_padding=op, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class GraspNetLAG(nn.Module):
    """
    Encoder(3 downs) -> LAG -> Decoder(3 ups) with skip-concat
    Heads:
      quality(1),
      angle(2: cos/sin),
      width(1)  -> long side length (你现在监督 long_len)
      thick(1)  -> short side / height (新增)
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        use_sigmoid_quality: bool = True,
        use_sigmoid_width: bool = True,
        use_sigmoid_thick: bool = True,   # <-- 新增
        normalize_angle: bool = True,
    ):
        super().__init__()
        c1 = base_channels        # 32
        c2 = base_channels * 2    # 64
        c3 = base_channels * 4    # 128
        cm = base_channels * 8    # 256

        self.stem = conv_bn_relu(in_channels, c1, k=3, s=1)

        self.enc1 = conv_bn_relu(in_channels, c1, k=3, s=2)  # /2
        self.enc2 = conv_bn_relu(c1, c2, k=3, s=2)           # /4
        self.enc3 = conv_bn_relu(c2, c3, k=3, s=2)           # /8

        self.to_mid = conv_bn_relu(c3, cm, k=3, s=1)         # keep /8
        self.lag = LAGBlock(cm)

        self.up3 = deconv_bn_relu(cm, c3)                    # /8 -> /4
        self.fuse3 = conv_bn_relu(c3 + c2, c3, k=3, s=1)

        self.up2 = deconv_bn_relu(c3, c2)                    # /4 -> /2
        self.fuse2 = conv_bn_relu(c2 + c1, c2, k=3, s=1)

        self.up1 = deconv_bn_relu(c2, c1)                    # /2 -> /1
        self.fuse1 = conv_bn_relu(c1 + c1, c1, k=3, s=1)

        # Heads (1x1)
        self.head_quality = nn.Conv2d(c1, 1, kernel_size=1)
        self.head_angle   = nn.Conv2d(c1, 2, kernel_size=1)   # cos, sin
        self.head_width   = nn.Conv2d(c1, 1, kernel_size=1)   # long
        self.head_thick   = nn.Conv2d(c1, 1, kernel_size=1)   # short/height  <-- 新增

        self.use_sigmoid_quality = use_sigmoid_quality
        self.use_sigmoid_width = use_sigmoid_width
        self.use_sigmoid_thick = use_sigmoid_thick
        self.normalize_angle = normalize_angle
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        s0 = self.stem(x)     # (B,c1,H,W)

        e1 = self.enc1(x)     # (B,c1,H/2,W/2)
        e2 = self.enc2(e1)    # (B,c2,H/4,W/4)
        e3 = self.enc3(e2)    # (B,c3,H/8,W/8)

        mid = self.lag(self.to_mid(e3))  # (B,cm,H/8,W/8)

        d3 = self.up3(mid)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.fuse3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.fuse2(d2)

        d1 = self.up1(d2)
        if d1.shape[-2:] != s0.shape[-2:]:
            d1 = F.interpolate(d1, size=s0.shape[-2:], mode="bilinear", align_corners=False)

        d1 = torch.cat([d1, s0], dim=1)
        feat = self.fuse1(d1)

        q   = self.head_quality(feat)
        ang = self.head_angle(feat)
        w   = self.head_width(feat)   # long
        t   = self.head_thick(feat)   # short  <-- 新增

        if self.use_sigmoid_quality:
            q = self.sigmoid(q)
        if self.use_sigmoid_width:
            w = self.sigmoid(w)
        if self.use_sigmoid_thick:
            t = self.sigmoid(t)

        if self.normalize_angle:
            eps = 1e-6
            norm = torch.sqrt(ang[:, 0:1] ** 2 + ang[:, 1:2] ** 2 + eps)
            ang = ang / norm

        return {"quality": q, "angle": ang, "width": w, "thick": t}


if __name__ == "__main__":
    model = GraspNetLAG(in_channels=3, base_channels=32)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("quality:", y["quality"].shape, "angle:", y["angle"].shape,
          "width:", y["width"].shape, "thick:", y["thick"].shape)
