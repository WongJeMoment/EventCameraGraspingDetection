# live_viz.py
import math
from typing import Optional, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt


def _to_numpy_img(x: torch.Tensor) -> np.ndarray:
    """
    x: (3,H,W) in [0,1]
    return: (H,W,3) uint8
    """
    x = x.detach().cpu().clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _angle_to_rgb(cos_map: np.ndarray, sin_map: np.ndarray) -> np.ndarray:
    """
    将 cos/sin 映射到颜色（HSV->RGB），角度=色相
    cos_map/sin_map: (H,W)
    return: (H,W,3) float in [0,1]
    """
    ang = np.arctan2(sin_map, cos_map)  # [-pi, pi]
    hue = (ang + math.pi) / (2.0 * math.pi)  # [0,1]

    sat = np.ones_like(hue)
    val = np.ones_like(hue)

    h = hue * 6.0
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)

    p = val * (1.0 - sat)
    q = val * (1.0 - f * sat)
    t = val * (1.0 - (1.0 - f) * sat)

    r = np.zeros_like(hue)
    g = np.zeros_like(hue)
    b = np.zeros_like(hue)

    m = (i == 0); r[m], g[m], b[m] = val[m], t[m], p[m]
    m = (i == 1); r[m], g[m], b[m] = q[m], val[m], p[m]
    m = (i == 2); r[m], g[m], b[m] = p[m], val[m], t[m]
    m = (i == 3); r[m], g[m], b[m] = p[m], q[m], val[m]
    m = (i == 4); r[m], g[m], b[m] = t[m], p[m], val[m]
    m = (i == 5); r[m], g[m], b[m] = val[m], p[m], q[m]

    return np.stack([r, g, b], axis=-1)


class LiveVisualizer:
    """
    实时可视化窗口（不保存）
    show(x, pred, gt)
    """
    def __init__(self):
        # 交互模式
        plt.ion()

        self.fig = plt.figure(figsize=(12, 8))
        self.fig.suptitle("Live Grasp Visualization", fontsize=14)

        self.axes = [self.fig.add_subplot(2, 4, i + 1) for i in range(8)]
        titles = [
            "Image",
            "GT Quality",
            "Pred Quality",
            "Overlay (Pred Q)",
            "GT Angle (color)",
            "Pred Angle (color)",
            "GT Width",
            "Pred Width",
        ]
        for ax, t in zip(self.axes, titles):
            ax.set_title(t)
            ax.axis("off")

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    @torch.no_grad()
    def show(
        self,
        x: torch.Tensor,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor],
        item_index: int = 0,
        epoch: Optional[int] = None,
    ):
        """
        x:    (B,3,H,W)
        pred: dict {'quality':(B,1,H,W), 'angle':(B,2,H,W), 'width':(B,1,H,W)}
        gt:   同上
        """
        i = int(item_index)

        img = _to_numpy_img(x[i])

        gt_q = gt["quality"][i, 0].detach().cpu().numpy()
        pr_q = pred["quality"][i, 0].detach().cpu().numpy()

        gt_w = gt["width"][i, 0].detach().cpu().numpy()
        pr_w = pred["width"][i, 0].detach().cpu().numpy()

        gt_cos = gt["angle"][i, 0].detach().cpu().numpy()
        gt_sin = gt["angle"][i, 1].detach().cpu().numpy()
        pr_cos = pred["angle"][i, 0].detach().cpu().numpy()
        pr_sin = pred["angle"][i, 1].detach().cpu().numpy()

        gt_ang_rgb = _angle_to_rgb(gt_cos, gt_sin)
        pr_ang_rgb = _angle_to_rgb(pr_cos, pr_sin)

        # 1) Image
        self._imshow(0, img)

        # 2) GT quality
        self._imshow(1, gt_q, vmin=0, vmax=1)

        # 3) Pred quality
        self._imshow(2, pr_q, vmin=0, vmax=1)

        # 4) overlay
        ax = self.axes[3]
        ax.clear()
        ax.set_title("Overlay (Pred Q)" + (f" | epoch {epoch}" if epoch is not None else ""))
        ax.axis("off")
        ax.imshow(img)
        ax.imshow(pr_q, alpha=0.45, vmin=0, vmax=1)

        # 5/6) angle
        self._imshow(4, gt_ang_rgb)
        self._imshow(5, pr_ang_rgb)

        # 7/8) width
        self._imshow(6, gt_w, vmin=0, vmax=1)
        self._imshow(7, pr_w, vmin=0, vmax=1)

        # 刷新窗口
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _imshow(self, idx: int, data, vmin=None, vmax=None):
        ax = self.axes[idx]
        ax.clear()
        ax.axis("off")
        # 重新写标题，避免 clear 后标题丢失
        titles = [
            "Image",
            "GT Quality",
            "Pred Quality",
            "Overlay (Pred Q)",
            "GT Angle (color)",
            "Pred Angle (color)",
            "GT Width",
            "Pred Width",
        ]
        ax.set_title(titles[idx])

        if data.ndim == 2:
            ax.imshow(data, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(data)
