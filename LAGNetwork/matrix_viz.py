# matrix_viz.py
import numpy as np
import torch
import matplotlib.pyplot as plt


def _to_hw(x: torch.Tensor) -> np.ndarray:
    """
    把 tensor 统一变成 (H,W) numpy
    支持:
      (H,W)
      (1,H,W)
      (1,1,H,W)
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.detach().cpu()

    if x.dim() == 2:
        return x.numpy()
    if x.dim() == 3:
        # (C,H,W) -> 取第0通道
        return x[0].numpy()
    if x.dim() == 4:
        # (B,C,H,W) -> 取第0个样本第0通道
        return x[0, 0].numpy()

    raise ValueError(f"Unsupported tensor shape for _to_hw: {tuple(x.shape)}")


def _angle_cos_to_hw(angle: torch.Tensor) -> np.ndarray:
    """
    angle 支持:
      (2,H,W)
      (1,2,H,W)
    返回 cos(theta) 的 (H,W)
    """
    if not torch.is_tensor(angle):
        angle = torch.as_tensor(angle)
    angle = angle.detach().cpu()

    if angle.dim() == 3:
        # (2,H,W)
        return angle[0].numpy()
    if angle.dim() == 4:
        # (1,2,H,W)
        return angle[0, 0].numpy()

    raise ValueError(f"Unsupported angle shape: {tuple(angle.shape)}")


class MatrixVisualizer:
    """
    显示 GT / Pred 的矩阵图（不显示RGB）
    """

    def __init__(self, pause_time: float = 0.2):
        plt.ion()
        self.pause_time = float(pause_time)

        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 7))
        self.fig.suptitle("GT vs Pred (Matrix)", fontsize=14)

        titles = [
            "GT Quality",
            "Pred Quality",
            "|GT - Pred|",
            "GT Angle (cos)",
            "Pred Angle (cos)",
            "Width: GT | Pred",
        ]
        for ax, t in zip(self.axes.flatten(), titles):
            ax.set_title(t)
            ax.axis("off")

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    @torch.no_grad()
    def show(self, gt: dict, pred: dict, epoch: int, sample_idx: int):
        # quality -> (H,W)
        gt_q = _to_hw(gt["quality"])
        pr_q = _to_hw(pred["quality"])
        diff_q = np.abs(gt_q - pr_q)

        # angle -> (H,W) cos(theta)
        gt_ang = _angle_cos_to_hw(gt["angle"])
        pr_ang = _angle_cos_to_hw(pred["angle"])

        # width -> (H,W)
        gt_w = _to_hw(gt["width"])
        pr_w = _to_hw(pred["width"])

        panels = [
            ("GT Quality", gt_q, 0, 1),
            ("Pred Quality", pr_q, 0, 1),
            ("|GT - Pred|", diff_q, None, None),
            ("GT Angle (cos)", gt_ang, None, None),
            ("Pred Angle (cos)", pr_ang, None, None),
        ]

        for ax, (title, mat, vmin, vmax) in zip(self.axes.flatten()[:5], panels):
            ax.clear()
            ax.set_title(title)
            ax.axis("off")
            if vmin is None:
                ax.imshow(mat, cmap="jet")
            else:
                ax.imshow(mat, cmap="jet", vmin=vmin, vmax=vmax)

        # width 拼接: GT | Pred
        ax = self.axes.flatten()[5]
        ax.clear()
        ax.set_title("Width: GT | Pred")
        ax.axis("off")
        w_cat = np.concatenate([gt_w, pr_w], axis=1)
        ax.imshow(w_cat, cmap="jet", vmin=0, vmax=1)

        self.fig.suptitle(f"Epoch {epoch} | Sample {sample_idx}", fontsize=14)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(self.pause_time)
