# matrix_viz.py
import numpy as np
import torch
import matplotlib.pyplot as plt


def _to_hw(x: torch.Tensor) -> np.ndarray:
    """
    把 tensor 统一变成 (H,W) numpy
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.detach().cpu()

    if x.dim() == 2:
        return x.numpy()
    if x.dim() == 3:
        return x[0].numpy()
    if x.dim() == 4:
        return x[0, 0].numpy()

    raise ValueError(f"Unsupported tensor shape for _to_hw: {tuple(x.shape)}")


def _angle_cos_to_hw(angle: torch.Tensor) -> np.ndarray:
    """
    返回 cos(theta) 的 (H,W)
    """
    if not torch.is_tensor(angle):
        angle = torch.as_tensor(angle)
    angle = angle.detach().cpu()

    if angle.dim() == 3:
        return angle[0].numpy()
    if angle.dim() == 4:
        return angle[0, 0].numpy()

    raise ValueError(f"Unsupported angle shape: {tuple(angle.shape)}")


class MatrixVisualizer:
    """
    无闪烁优化版：显示 GT / Pred 的矩阵图
    """

    def __init__(self, pause_time: float = 0.2):
        plt.ion()
        self.pause_time = float(pause_time)

        # 初始化画布
        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 7))
        self.fig.suptitle("GT vs Pred (Matrix)", fontsize=14)

        # 预定义标题
        self.titles = [
            "GT Quality",
            "Pred Quality",
            "|GT - Pred|",
            "GT Angle (cos)",
            "Pred Angle (cos)",
            "Width: GT | Pred",
        ]

        # 展平 axes 方便遍历
        self.ax_flat = self.axes.flatten()

        for ax, t in zip(self.ax_flat, self.titles):
            ax.set_title(t)
            ax.axis("off")

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # ✅ 关键点：用于存储 imshow 返回的图像对象
        # 初始为 None，表示还没画过
        self.artists = [None] * 6

    @torch.no_grad()
    def show(self, gt: dict, pred: dict, epoch: int, sample_idx: int):
        # 1. 准备数据
        gt_q = _to_hw(gt["quality"])
        pr_q = _to_hw(pred["quality"])
        diff_q = np.abs(gt_q - pr_q)

        gt_ang = _angle_cos_to_hw(gt["angle"])
        pr_ang = _angle_cos_to_hw(pred["angle"])

        gt_w = _to_hw(gt["width"])
        pr_w = _to_hw(pred["width"])
        w_cat = np.concatenate([gt_w, pr_w], axis=1)

        # 将6个图的数据和配置打包
        # 格式: (数据, vmin, vmax)
        # diff_q 的 vmax 设为 None 让它自动缩放，或者你可以固定为 1.0
        data_list = [
            (gt_q, 0, 1),
            (pr_q, 0, 1),
            (diff_q, 0, None),  # Diff 自动范围
            (gt_ang, -1, 1),
            (pr_ang, -1, 1),
            (w_cat, 0, 1)
        ]

        # 2. 绘图或更新
        for i, (mat, vmin, vmax) in enumerate(data_list):
            if self.artists[i] is None:
                # --- 第一次运行：创建图像 ---
                # 只有第一次调用 imshow，后续只更新数据
                if vmin is None:
                    self.artists[i] = self.ax_flat[i].imshow(mat, cmap="jet")
                else:
                    self.artists[i] = self.ax_flat[i].imshow(mat, cmap="jet", vmin=vmin, vmax=vmax)
            else:
                # --- 后续运行：原地更新数据 (无闪烁) ---
                self.artists[i].set_data(mat)

                # 如果是 Diff 图 (vmin=None)，可能需要更新颜色范围
                if vmin is None:
                    self.artists[i].autoscale()

                    # 3. 更新总标题
        self.fig.suptitle(f"Epoch {epoch} | Sample {sample_idx}", fontsize=14)

        # 4. 刷新显示 (去掉 draw，只用 flush 和 pause 提高性能)
        # self.fig.canvas.draw_idle() # 也可以加上这句确保重绘
        self.fig.canvas.flush_events()
        plt.pause(self.pause_time)