# heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConv(nn.Module):
    """Depthwise separable conv: dw -> pw, lightweight and fast."""
    def __init__(self, ch, k=3, s=1):
        super().__init__()
        p = (k - 1) // 2
        self.dw = nn.Conv2d(ch, ch, k, s, p, groups=ch, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.pw = nn.Conv2d(ch, ch, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class GraspHead(nn.Module):
    """
    Grasp head that outputs:
      Q_logits: (B,1,H,W)
      reg:      (B,5,H,W)  channels: dx, dy, logw, sin, cos

    Options:
      - normalize_sincos: normalize sin/cos to unit length for stable angle
      - dxdy_sigmoid: constrain dx,dy to [0,1) if your target is offset within cell
      - logw_clamp: clamp logw to a reasonable range to avoid instability
    """
    def __init__(
        self,
        in_ch: int,
        head_ch: int = 96,
        num_dw: int = 1,
        normalize_sincos: bool = True,
        dxdy_sigmoid: bool = True,
        logw_clamp: tuple = (-5.0, 6.0),   # exp(-5)=0.0067px, exp(6)=403px (adjust if needed)
    ):
        super().__init__()
        self.normalize_sincos = normalize_sincos
        self.dxdy_sigmoid = dxdy_sigmoid
        self.logw_clamp = logw_clamp

        # light stem
        stem = []
        stem.append(nn.Conv2d(in_ch, head_ch, kernel_size=1, stride=1, padding=0, bias=False))
        stem.append(nn.BatchNorm2d(head_ch))
        stem.append(nn.ReLU(inplace=True))
        for _ in range(num_dw):
            stem.append(DWConv(head_ch))
        self.stem = nn.Sequential(*stem)

        # two heads: Q and reg
        self.q_conv = nn.Conv2d(head_ch, 1, kernel_size=1, stride=1, padding=0)
        self.reg_conv = nn.Conv2d(head_ch, 5, kernel_size=1, stride=1, padding=0)

        # init: small weights for stable start
        nn.init.normal_(self.q_conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.q_conv.bias, 0.0)
        nn.init.normal_(self.reg_conv.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.reg_conv.bias, 0.0)

    def forward(self, feat):
        """
        feat: (B, C, H, W)
        returns:
          Q_logits: (B,1,H,W)
          reg: (B,5,H,W): dx,dy,logw,sin,cos
        """
        x = self.stem(feat)

        Q_logits = self.q_conv(x)
        reg = self.reg_conv(x)

        # ----- post constraints (optional but helpful) -----
        # dx,dy in [0,1)
        if self.dxdy_sigmoid:
            dx = torch.sigmoid(reg[:, 0:1])
            dy = torch.sigmoid(reg[:, 1:2])
        else:
            dx = reg[:, 0:1]
            dy = reg[:, 1:2]

        # logw clamp
        logw = reg[:, 2:3]
        if self.logw_clamp is not None:
            lo, hi = float(self.logw_clamp[0]), float(self.logw_clamp[1])
            logw = torch.clamp(logw, lo, hi)

        sin = reg[:, 3:4]
        cos = reg[:, 4:5]

        # normalize sin/cos to unit circle for stable theta
        if self.normalize_sincos:
            norm = torch.sqrt(sin * sin + cos * cos + 1e-8)
            sin = sin / norm
            cos = cos / norm

        reg_out = torch.cat([dx, dy, logw, sin, cos], dim=1)
        return Q_logits, reg_out


if __name__ == "__main__":
    # quick sanity
    B = 2
    feat = torch.randn(B, 96, 60, 80)  # e.g., stride-4 feature
    head = GraspHead(in_ch=96, head_ch=96, num_dw=1)
    Q, reg = head(feat)
    print("Q:", Q.shape, "reg:", reg.shape)  # (2,1,60,80) (2,5,60,80)
