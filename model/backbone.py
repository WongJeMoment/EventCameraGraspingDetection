# backbone.py
import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utils
# -------------------------
def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    def forward(self, x):
        return x * (F.relu6(x + 3.0, inplace=True) / 6.0)


def conv_bn_act(in_ch, out_ch, k=3, s=1, p=None, groups=1, act: Optional[nn.Module] = None):
    if p is None:
        p = (k - 1) // 2
    layers = [
        nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if act is not None:
        layers.append(act)
    return nn.Sequential(*layers)


class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        mid = max(8, ch // reduction)
        self.fc1 = nn.Conv2d(ch, mid, 1)
        self.fc2 = nn.Conv2d(mid, ch, 1)
        self.act = nn.ReLU(inplace=True)
        self.gate = HSigmoid()

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.act(self.fc1(s))
        s = self.gate(self.fc2(s))
        return x * s


# -------------------------
# MobileNetV3 Small
# -------------------------
class InvertedResidualV3(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, exp_ch, use_se, nl):
        super().__init__()
        assert s in (1, 2)
        self.use_res = (s == 1 and in_ch == out_ch)

        if nl == "RE":
            act = nn.ReLU(inplace=True)
        elif nl == "HS":
            act = HSwish()
        else:
            raise ValueError("nl must be RE or HS")

        layers = []

        # pw expand
        if exp_ch != in_ch:
            layers.append(conv_bn_act(in_ch, exp_ch, k=1, s=1, p=0, act=act))

        # dw
        layers.append(conv_bn_act(exp_ch, exp_ch, k=k, s=s, groups=exp_ch, act=act))

        # ✅ SE 放在 expand 通道
        if use_se:
            layers.append(SEBlock(exp_ch))

        # pw linear
        layers.append(conv_bn_act(exp_ch, out_ch, k=1, s=1, p=0, act=None))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = out + x
        return out


class MobileNetV3Small(nn.Module):
    """
    Returns dict features:
      c2: stride 4
      c3: stride 8
      c4: stride 16
    """
    def __init__(self, width_mult: float = 1.0, out_indices=(2, 3, 4)):
        super().__init__()
        self.out_indices = out_indices

        # stem
        in_ch = make_divisible(16 * width_mult)
        self.stem = conv_bn_act(3, in_ch, k=3, s=2, act=HSwish())  # stride 2

        # cfg: (k, exp, out, se, nl, s)
        cfg = [
            (3, 16, 16, True,  "RE", 2),  # -> s4
            (3, 72, 24, False, "RE", 2),  # -> s8
            (3, 88, 24, False, "RE", 1),  # -> s8
            (5, 96, 40, True,  "HS", 2),  # -> s16
            (5, 240, 40, True, "HS", 1),  # -> s16
            (5, 240, 40, True, "HS", 1),  # -> s16
            (5, 120, 48, True, "HS", 1),  # -> s16
            (5, 144, 48, True, "HS", 1),  # -> s16
            (5, 288, 96, True, "HS", 2),  # -> s32 (可选不用)
            (5, 576, 96, True, "HS", 1),  # -> s32
        ]

        blocks = []
        cur = in_ch
        self.stage_out = []  # channels after each block
        for (k, exp, out, se, nl, s) in cfg:
            exp_ch = make_divisible(exp * width_mult)
            out_ch = make_divisible(out * width_mult)
            blocks.append(InvertedResidualV3(cur, out_ch, k=k, s=s, exp_ch=exp_ch, use_se=se, nl=nl))
            cur = out_ch
            self.stage_out.append(cur)
        self.blocks = nn.ModuleList(blocks)

        # record strides at certain blocks for c2/c3/c4
        # After stem: stride 2.
        # Block0 has s=2 -> stride 4
        # Block1 has s=2 -> stride 8
        # Block3 has s=2 -> stride 16
        self._c2_idx = 0
        self._c3_idx = 1
        self._c4_idx = 3

    def forward(self, x) -> Dict[str, torch.Tensor]:
        feats = {}
        x = self.stem(x)  # s2
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self._c2_idx:
                feats["c2"] = x  # stride 4
            if i == self._c3_idx:
                feats["c3"] = x  # stride 8
            if i == self._c4_idx:
                feats["c4"] = x  # stride 16
        return feats


# -------------------------
# ShuffleNetV2
# -------------------------
def channel_shuffle(x, groups: int):
    b, c, h, w = x.size()
    assert c % groups == 0
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride: int):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride
        branch_ch = out_ch // 2

        if stride == 1:
            assert in_ch == out_ch
            inp = branch_ch
        else:
            inp = in_ch

        # branch1 only when stride=2
        if stride == 2:
            self.branch1 = nn.Sequential(
                conv_bn_act(inp, inp, k=3, s=2, groups=inp, act=None),
                conv_bn_act(inp, branch_ch, k=1, s=1, p=0, act=nn.ReLU(inplace=True)),
            )
        else:
            self.branch1 = None

        # branch2
        self.branch2 = nn.Sequential(
            conv_bn_act(inp, branch_ch, k=1, s=1, p=0, act=nn.ReLU(inplace=True)),
            conv_bn_act(branch_ch, branch_ch, k=3, s=stride, groups=branch_ch, act=None),
            conv_bn_act(branch_ch, branch_ch, k=1, s=1, p=0, act=nn.ReLU(inplace=True)),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = torch.chunk(x, 2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    """
    Returns dict features:
      c2: stride 4
      c3: stride 8
      c4: stride 16
    """
    def __init__(self, width_mult: float = 1.0):
        super().__init__()
        # channels for 0.5/1.0/1.5/2.0, we use 1.0 default
        if abs(width_mult - 1.0) < 1e-6:
            stage_out_channels = [24, 116, 232, 464]   # stem, stage2, stage3, stage4
        elif abs(width_mult - 0.5) < 1e-6:
            stage_out_channels = [24, 48, 96, 192]
        elif abs(width_mult - 1.5) < 1e-6:
            stage_out_channels = [24, 176, 352, 704]
        elif abs(width_mult - 2.0) < 1e-6:
            stage_out_channels = [24, 244, 488, 976]
        else:
            # fallback: scale 1.0 roughly
            stage_out_channels = [24,
                                  make_divisible(116 * width_mult),
                                  make_divisible(232 * width_mult),
                                  make_divisible(464 * width_mult)]

        self.stem = nn.Sequential(
            conv_bn_act(3, stage_out_channels[0], k=3, s=2, act=nn.ReLU(inplace=True)),  # s2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # s4
        )

        # stage repeats for shufflenetv2 1.0x
        stage_repeats = [4, 8, 4]  # stage2, stage3, stage4
        input_channels = stage_out_channels[0]

        self.stage2 = self._make_stage(input_channels, stage_out_channels[1], stage_repeats[0])  # s4 -> s8? actually first block stride2: s8
        self.stage3 = self._make_stage(stage_out_channels[1], stage_out_channels[2], stage_repeats[1])  # -> s16
        self.stage4 = self._make_stage(stage_out_channels[2], stage_out_channels[3], stage_repeats[2])  # -> s32

        # For our features:
        # after stem: stride 4
        # after stage2: stride 8
        # after stage3: stride 16
        # stage4: stride 32 (可选不用)

        self.c2_ch = stage_out_channels[0]
        self.c3_ch = stage_out_channels[1]
        self.c4_ch = stage_out_channels[2]

    def _make_stage(self, in_ch, out_ch, repeat: int):
        layers = [ShuffleV2Block(in_ch, out_ch, stride=2)]
        for _ in range(repeat - 1):
            layers.append(ShuffleV2Block(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        feats = {}
        x = self.stem(x)        # stride 4
        feats["c2"] = x
        x = self.stage2(x)      # stride 8
        feats["c3"] = x
        x = self.stage3(x)      # stride 16
        feats["c4"] = x
        return feats


# -------------------------
# Builder
# -------------------------
def build_backbone(name: str = "mobilenetv3_small", width_mult: float = 1.0):
    """
    name:
      - "mobilenetv3_small"
      - "shufflenetv2"
    """
    name = name.lower()
    if name in ("mobilenetv3_small", "mnetv3s", "mbv3s"):
        return MobileNetV3Small(width_mult=width_mult)
    if name in ("shufflenetv2", "shufflev2", "shufflenet"):
        return ShuffleNetV2(width_mult=width_mult)
    raise ValueError("Unknown backbone: {}".format(name))


if __name__ == "__main__":
    # quick sanity
    x = torch.randn(1, 3, 240, 320)
    for n in ["mobilenetv3_small", "shufflenetv2"]:
        bb = build_backbone(n, width_mult=1.0)
        feats = bb(x)
        print(n, {k: tuple(v.shape) for k, v in feats.items()})
