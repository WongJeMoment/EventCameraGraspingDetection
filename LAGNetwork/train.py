# train.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

import os
import random
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from config import TrainConfig
from model import GraspNetLAG
from utils.grasp_dataset_txt import GraspTxtDataset
from matrix_viz import MatrixVisualizer


# ============================================================
# 0) Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 1) boxes -> GT maps (baseline)
# ============================================================
def _order_points_clockwise(pts: torch.Tensor) -> torch.Tensor:
    center = pts.mean(dim=0, keepdim=True)
    v = pts - center
    ang = torch.atan2(v[:, 1], v[:, 0])
    idx = torch.argsort(ang)
    return pts[idx]


def _estimate_angle_width(pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    p = _order_points_clockwise(pts)
    e0 = p[1] - p[0]
    e1 = p[2] - p[1]
    len0 = torch.norm(e0)
    len1 = torch.norm(e1)

    if len0 >= len1:
        direction = e0
        width = len1
    else:
        direction = e1
        width = len0

    theta = torch.atan2(direction[1], direction[0])
    return theta, width


def _bbox_from_quad(pts: torch.Tensor) -> Tuple[int, int, int, int]:
    x0 = int(torch.min(pts[:, 0]).item())
    y0 = int(torch.min(pts[:, 1]).item())
    x1 = int(torch.max(pts[:, 0]).item())
    y1 = int(torch.max(pts[:, 1]).item())
    return x0, y0, x1, y1


def boxes_to_gt_maps(
    boxes_norm: torch.Tensor,
    out_hw: Tuple[int, int],
    max_boxes_used: int = 20,
) -> Dict[str, torch.Tensor]:
    """
    boxes_norm: (K,8) normalized quad points (x,y)*4 in [0,1]
    out_hw: (H,W) of output maps
    """
    H, W = out_hw
    device = boxes_norm.device

    quality = torch.zeros((1, H, W), device=device)
    angle = torch.zeros((2, H, W), device=device)
    width = torch.zeros((1, H, W), device=device)

    if boxes_norm.numel() == 0:
        return {"quality": quality, "angle": angle, "width": width}

    K = min(int(boxes_norm.shape[0]), max_boxes_used)

    for i in range(K):
        b = boxes_norm[i].view(4, 2)
        pts = torch.stack([b[:, 0] * (W - 1), b[:, 1] * (H - 1)], dim=1)

        theta, wlen = _estimate_angle_width(pts)

        x0, y0, x1, y1 = _bbox_from_quad(pts)
        x0 = max(0, min(W - 1, x0))
        x1 = max(0, min(W - 1, x1))
        y0 = max(0, min(H - 1, y0))
        y1 = max(0, min(H - 1, y1))
        if x1 <= x0 or y1 <= y0:
            continue

        # NOTE: this is a "bbox fill" baseline (coarse)
        quality[:, y0:y1 + 1, x0:x1 + 1] = 1.0

        c = torch.cos(theta)
        s = torch.sin(theta)
        angle[0, y0:y1 + 1, x0:x1 + 1] = c
        angle[1, y0:y1 + 1, x0:x1 + 1] = s

        w_norm = torch.clamp(wlen / float(max(H, W)), 0.0, 1.0)
        width[:, y0:y1 + 1, x0:x1 + 1] = w_norm

    return {"quality": quality, "angle": angle, "width": width}


# ============================================================
# 2) pred maps -> boxes (visualization)
# ============================================================
def _quad_from_center(
    cx: torch.Tensor,
    cy: torch.Tensor,
    theta: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Build rotated quad (4,2) in pixel coords from center (cx,cy), angle theta, width w, height h.
    """
    c = torch.cos(theta)
    s = torch.sin(theta)

    dx = w / 2.0
    dy = h / 2.0

    local = torch.tensor(
        [[-1.0, -1.0],
         [ 1.0, -1.0],
         [ 1.0,  1.0],
         [-1.0,  1.0]],
        device=device,
        dtype=torch.float32,
    )
    local[:, 0] *= dx
    local[:, 1] *= dy

    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])  # (2,2)
    pts = local @ R.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


def decode_maps_to_boxes_norm(
    pred_one: Dict[str, torch.Tensor],
    topk: int = 5,
    q_thresh: float = 0.5,
    grasp_h_ratio: float = 0.5,
    width_scale: float = 2.0,
) -> torch.Tensor:
    """
    pred_one:
      quality: (1,Hm,Wm)
      angle:   (2,Hm,Wm)
      width:   (1,Hm,Wm)
    Return:
      boxes_norm: (K,8) with normalized quad points (x,y)*4 in [0,1]
    """
    q = pred_one["quality"][0]                 # (Hm,Wm)
    cos_t = pred_one["angle"][0]               # (Hm,Wm)
    sin_t = pred_one["angle"][1]               # (Hm,Wm)
    w_norm = pred_one["width"][0].clamp(0, 1)  # (Hm,Wm)

    Hm, Wm = q.shape
    device = q.device

    flat_q = q.reshape(-1)
    k = min(topk, flat_q.numel())
    vals, idxs = torch.topk(flat_q, k=k, largest=True)

    boxes: List[torch.Tensor] = []
    for v, idx in zip(vals, idxs):
        if float(v.item()) < q_thresh:
            continue

        y = (idx // Wm).float()
        x = (idx % Wm).float()

        yy = int(y.item())
        xx = int(x.item())

        theta = torch.atan2(sin_t[yy, xx], cos_t[yy, xx])

        # recover pixels ~ w_norm * max(Hm,Wm) * width_scale
        w_px = (w_norm[yy, xx] * float(max(Hm, Wm)) * width_scale).clamp(min=2.0)
        h_px = (w_px * grasp_h_ratio).clamp(min=2.0)

        quad = _quad_from_center(x, y, theta, w_px, h_px, device=device)  # (4,2) in map coords

        quad_norm = torch.stack(
            [quad[:, 0] / (Wm - 1), quad[:, 1] / (Hm - 1)],
            dim=1
        ).clamp(0.0, 1.0)
        boxes.append(quad_norm.reshape(-1))  # (8,)

    if len(boxes) == 0:
        return torch.zeros((0, 8), device=device)

    return torch.stack(boxes, dim=0)  # (K,8)


def _draw_quads_on_ax(
    ax,
    quads_norm: torch.Tensor,
    H_img: int,
    W_img: int,
    color: str,
    lw: float = 2.0,
):
    if not isinstance(quads_norm, torch.Tensor) or quads_norm.numel() == 0:
        return

    quads = quads_norm.view(-1, 4, 2).detach().cpu().numpy()
    for q in quads:
        xs = q[:, 0] * (W_img - 1)
        ys = q[:, 1] * (H_img - 1)
        xs = list(xs) + [xs[0]]
        ys = list(ys) + [ys[0]]
        ax.plot(xs, ys, color=color, linewidth=lw)


# ============================================================
# 3) Train visualization (only GT vs Pred boxes; NO Matrix white windows)
# ============================================================
@torch.no_grad()
def visualize_train_batch_boxes(
    cfg: TrainConfig,
    x: torch.Tensor,
    boxes_list: List[torch.Tensor],
    pred: Dict[str, torch.Tensor],
    epoch: int,
    step: int,
    fig=None,
    ax=None,
):
    """
    ✅ Fix white alternating frames:
      - reuse a single fig/ax
      - do NOT close('all')
      - force draw/flush/pause
    ✅ Also: NO MatrixVisualizer here.
    """
    B, _, H_img, W_img = x.shape
    x_cpu = x.detach().cpu()
    Hm, Wm = pred["quality"].shape[-2], pred["quality"].shape[-1]

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(B):
        ax.clear()

        img = x_cpu[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        gt_boxes = boxes_list[i]
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.to(cfg.device)

        pred_one = {
            "quality": pred["quality"][i],
            "angle": pred["angle"][i],
            "width": pred["width"][i],
        }

        pred_boxes_img_norm = decode_maps_to_boxes_norm(
            pred_one,
            topk=getattr(cfg, "viz_topk", 5),
            q_thresh=getattr(cfg, "viz_q_thresh", 0.5),
            grasp_h_ratio=getattr(cfg, "viz_h_ratio", 0.5),
            width_scale=getattr(cfg, "viz_width_scale", 2.0),
        )

        ax.imshow(img)
        ax.set_title(f"Train | Epoch {epoch} Step {step} | GT(green) vs Pred(red) | {i} | map {Hm}x{Wm}")
        ax.axis("off")

        _draw_quads_on_ax(ax, gt_boxes, H_img, W_img, color="lime", lw=2.0)
        _draw_quads_on_ax(ax, pred_boxes_img_norm, H_img, W_img, color="red", lw=2.0)

        # force render
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(getattr(cfg, "viz_pause", 0.05))

        if getattr(cfg, "viz_save", False):
            save_dir = os.path.join(cfg.ckpt_dir, "viz_train")
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, f"e{epoch:03d}_s{step:06d}_i{i}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig, ax


# ============================================================
# 4) Loss
# ============================================================
class GraspLoss(nn.Module):
    def __init__(self, w_q=1.0, w_a=1.0, w_w=1.0):
        super().__init__()
        self.w_q = w_q
        self.w_a = w_a
        self.w_w = w_w
        self.bce = nn.BCELoss()

    def forward(self, pred: Dict[str, torch.Tensor], gt: Dict[str, torch.Tensor]):
        loss_q = self.bce(pred["quality"], gt["quality"])

        mask = (gt["quality"] > 0.5).float()
        mask2 = mask.repeat(1, 2, 1, 1)

        if mask2.sum() > 0:
            loss_a = (torch.abs(pred["angle"] - gt["angle"]) * mask2).sum() / (mask2.sum() + 1e-6)
        else:
            loss_a = torch.tensor(0.0, device=mask.device)

        if mask.sum() > 0:
            loss_w = (torch.abs(pred["width"] - gt["width"]) * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss_w = torch.tensor(0.0, device=mask.device)

        loss = self.w_q * loss_q + self.w_a * loss_a + self.w_w * loss_w
        info = {
            "loss": float(loss.item()),
            "q": float(loss_q.item()),
            "a": float(loss_a.item()),
            "w": float(loss_w.item()),
        }
        return loss, info


# ============================================================
# 5) Collate
# ============================================================
def collate_keep_boxes(batch):
    xs, bs = zip(*batch)
    xs = torch.stack(xs, 0)
    return xs, bs


# ============================================================
# 6) Train / Eval
# ============================================================
def train_one_epoch(cfg: TrainConfig, epoch: int, model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0

    viz_every = getattr(cfg, "viz_every", 0)          # 0 disables
    viz_max_steps = getattr(cfg, "viz_max_steps", 0)  # 0 means no limit per epoch
    step_idx = 0

    # reuse one window for the whole epoch
    viz_fig, viz_ax = (None, None)
    if viz_every > 0:
        viz_fig, viz_ax = plt.subplots(figsize=(6, 6))

    for x, boxes_list in loader:
        x = x.to(cfg.device, non_blocking=True)

        pred = model(x)
        Hm, Wm = pred["quality"].shape[-2:]

        gt_q, gt_a, gt_w = [], [], []
        for b in boxes_list:
            b = b.to(cfg.device)
            gt = boxes_to_gt_maps(b, (Hm, Wm), max_boxes_used=cfg.max_boxes_used)
            gt_q.append(gt["quality"])
            gt_a.append(gt["angle"])
            gt_w.append(gt["width"])

        gt_batch = {
            "quality": torch.stack(gt_q, 0),
            "angle": torch.stack(gt_a, 0),
            "width": torch.stack(gt_w, 0),
        }

        loss, _ = criterion(pred, gt_batch)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

        if viz_every > 0 and (step_idx % viz_every == 0):
            if viz_max_steps <= 0 or step_idx < viz_max_steps:
                viz_fig, viz_ax = visualize_train_batch_boxes(
                    cfg, x, list(boxes_list), pred,
                    epoch=epoch, step=step_idx,
                    fig=viz_fig, ax=viz_ax
                )

        step_idx += 1

    if viz_fig is not None and getattr(cfg, "viz_close_epoch_fig", False):
        plt.close(viz_fig)

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_one_epoch(cfg: TrainConfig, model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n = 0

    for x, boxes_list in loader:
        x = x.to(cfg.device, non_blocking=True)

        pred = model(x)
        H, W = pred["quality"].shape[-2:]

        gt_q, gt_a, gt_w = [], [], []
        for b in boxes_list:
            b = b.to(cfg.device)
            gt = boxes_to_gt_maps(b, (H, W), max_boxes_used=cfg.max_boxes_used)
            gt_q.append(gt["quality"])
            gt_a.append(gt["angle"])
            gt_w.append(gt["width"])

        gt_batch = {
            "quality": torch.stack(gt_q, 0),
            "angle": torch.stack(gt_a, 0),
            "width": torch.stack(gt_w, 0),
        }

        loss, _ = criterion(pred, gt_batch)

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


# ============================================================
# 7) OPTIONAL: Visualize val set (MatrixVisualizer) - DISABLED by default
#    This is where your "white matrix" windows come from.
# ============================================================
@torch.no_grad()
def visualize_val_set(cfg: TrainConfig, model, val_set: GraspTxtDataset, viz: MatrixVisualizer, epoch: int):
    model.eval()

    for sample_idx in range(len(val_set)):
        x_one, boxes_one = val_set[sample_idx]
        x_one = x_one.unsqueeze(0).to(cfg.device)

        pred_one = model(x_one)
        Hm, Wm = pred_one["quality"].shape[-2:]

        gt_one = boxes_to_gt_maps(
            boxes_one.to(cfg.device),
            (Hm, Wm),
            max_boxes_used=cfg.max_boxes_used
        )

        viz.show(gt=gt_one, pred=pred_one, epoch=epoch, sample_idx=sample_idx)


# ============================================================
# 8) Main
# ============================================================
def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ---------- safe defaults ----------
    if not hasattr(cfg, "viz_every"):
        cfg.viz_every = 50
    if not hasattr(cfg, "viz_max_steps"):
        cfg.viz_max_steps = 10
    if not hasattr(cfg, "viz_topk"):
        cfg.viz_topk = 5
    if not hasattr(cfg, "viz_q_thresh"):
        cfg.viz_q_thresh = 0.5
    if not hasattr(cfg, "viz_h_ratio"):
        cfg.viz_h_ratio = 0.5
    if not hasattr(cfg, "viz_pause"):
        cfg.viz_pause = 0.05
    if not hasattr(cfg, "viz_width_scale"):
        cfg.viz_width_scale = 2.0

    # save train viz?
    if not hasattr(cfg, "viz_save"):
        cfg.viz_save = False
    if not hasattr(cfg, "viz_close_epoch_fig"):
        cfg.viz_close_epoch_fig = False

    # ✅ IMPORTANT: disable MatrixVisualizer windows by default (removes the white matrix)
    if not hasattr(cfg, "val_viz_enable"):
        cfg.val_viz_enable = False
    if not hasattr(cfg, "val_viz_every_epoch"):
        cfg.val_viz_every_epoch = 1  # if enabled, show every N epochs

    # set interactive mode ONCE
    plt.ion()

    train_set = GraspTxtDataset(cfg.train_images, cfg.train_labels, image_size=cfg.image_size)
    val_set = GraspTxtDataset(cfg.val_images, cfg.val_labels, image_size=cfg.image_size)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_keep_boxes,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_keep_boxes,
    )

    model = GraspNetLAG(in_channels=3, base_channels=cfg.base_channels).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = GraspLoss(w_q=cfg.w_q, w_a=cfg.w_a, w_w=cfg.w_w).to(cfg.device)

    # only used if val_viz_enable=True
    viz = MatrixVisualizer(pause_time=0.2)

    best = 1e9
    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(cfg, epoch, model, train_loader, optimizer, criterion)
        va_loss = eval_one_epoch(cfg, model, val_loader, criterion)

        print(f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")

        # ✅ This is where the "white matrix" comes from -> now controlled by a flag
        if cfg.val_viz_enable and (epoch % cfg.val_viz_every_epoch == 0):
            visualize_val_set(cfg, model, val_set, viz, epoch)

        if va_loss < best:
            best = va_loss
            torch.save(
                {"model": model.state_dict(), "cfg": cfg.__dict__},
                os.path.join(cfg.ckpt_dir, "best.pt"),
            )

    print("Done. Best val loss:", best)


if __name__ == "__main__":
    main()
