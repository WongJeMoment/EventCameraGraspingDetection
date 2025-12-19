# decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_act(in_ch, out_ch, k=1, s=1, p=0, act=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
              nn.BatchNorm2d(out_ch)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DWConv(nn.Module):
    """Depthwise separable conv block: dw -> pw"""
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


class LiteFPNDecoder(nn.Module):
    """
    Lightweight decoder/neck for grasp:
    input feats dict:
      c2: stride 4
      c3: stride 8
      c4: stride 16

    output:
      Q:   (B,1,H/4,W/4)   (logits; you can sigmoid outside)
      reg: (B,5,H/4,W/4)   channels: dx, dy, logw, sin, cos
    """
    def __init__(
        self,
        c2_ch, c3_ch, c4_ch,
        out_ch=96,
        num_dw=1,
        reg_norm=True,
    ):
        super().__init__()
        self.reg_norm = reg_norm

        # lateral 1x1 to unify channels
        self.lat2 = conv_bn_act(c2_ch, out_ch, k=1, s=1, p=0, act=True)
        self.lat3 = conv_bn_act(c3_ch, out_ch, k=1, s=1, p=0, act=True)
        self.lat4 = conv_bn_act(c4_ch, out_ch, k=1, s=1, p=0, act=True)

        # fusion refinement (depthwise for speed)
        self.refine2 = nn.Sequential(*[DWConv(out_ch) for _ in range(num_dw)])
        self.refine3 = nn.Sequential(*[DWConv(out_ch) for _ in range(num_dw)])
        self.refine4 = nn.Sequential(*[DWConv(out_ch) for _ in range(num_dw)])

        # heads (very light)
        # Q head: 1 channel
        self.q_head = nn.Sequential(
            DWConv(out_ch),
            nn.Conv2d(out_ch, 1, kernel_size=1, stride=1, padding=0)
        )

        # reg head: 5 channels (dx,dy,logw,sin,cos)
        self.reg_head = nn.Sequential(
            DWConv(out_ch),
            nn.Conv2d(out_ch, 5, kernel_size=1, stride=1, padding=0)
        )

        # init last conv small for stability
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, feats):
        """
        feats: dict with keys c2,c3,c4
        """
        c2 = feats["c2"]
        c3 = feats["c3"]
        c4 = feats["c4"]

        p4 = self.refine4(self.lat4(c4))  # stride 16
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.refine3(p3)             # stride 8
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2 = self.refine2(p2)             # stride 4 (final)

        Q = self.q_head(p2)               # logits
        reg = self.reg_head(p2)           # dx,dy,logw,sin,cos

        if self.reg_norm:
            # normalize sin/cos to unit length for stable angle
            sin = reg[:, 3:4, :, :]
            cos = reg[:, 4:5, :, :]
            norm = torch.sqrt(sin * sin + cos * cos + 1e-8)
            reg = torch.cat([reg[:, 0:3, :, :], sin / norm, cos / norm], dim=1)

        return Q, reg


if __name__ == "__main__":
    # quick sanity check
    B = 1
    x2 = torch.randn(B, 24, 60, 80)   # c2 stride4 (if input 240x320)
    x3 = torch.randn(B, 116, 30, 40)  # c3 stride8
    x4 = torch.randn(B, 232, 15, 20)  # c4 stride16

    dec = LiteFPNDecoder(c2_ch=24, c3_ch=116, c4_ch=232, out_ch=96, num_dw=1)
    Q, reg = dec({"c2": x2, "c3": x3, "c4": x4})
    print("Q:", Q.shape, "reg:", reg.shape)  # (1,1,60,80) and (1,5,60,80)
