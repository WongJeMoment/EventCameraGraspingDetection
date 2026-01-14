# train.py (Final: long+short supervision with thick head + endpoint loss + correct decode)
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from LAGNetwork.config import TrainConfig
from LAGNetwork.model import GraspNetLAG
from utils.grasp_dataset_txt import GraspTxtDataset


# -------------------------
# GT map builder
# -------------------------
def boxes_to_gt_maps(boxes, out_hw, sigma=6.0):
    """
    boxes: [N, 8] or [N, 4, 2] (normalized [0,1])
    out_hw: (H, W)

    Output:
      quality: [1,H,W]
      angle:   [2,H,W] = (cos2t, sin2t) aligned to LONG side
      width:   [1,H,W] = normalized LONG length (0~1)
      thick:   [1,H,W] = normalized SHORT length (0~1)
      aw_mask: [1,H,W]
    """
    H, W = out_hw
    device = boxes.device

    quality = torch.zeros((1, H, W), device=device)
    angle   = torch.zeros((2, H, W), device=device)
    width   = torch.zeros((1, H, W), device=device)   # long
    thick   = torch.zeros((1, H, W), device=device)   # short
    aw_mask = torch.zeros((1, H, W), device=device)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    xx = xx.float()
    yy = yy.float()

    # normalize box shape
    if boxes.numel() == 0:
        return {"quality": quality, "angle": angle, "width": width, "thick": thick, "aw_mask": aw_mask}

    if boxes.dim() == 2 and boxes.size(-1) == 8:
        boxes_ = boxes.view(-1, 4, 2)
    elif boxes.dim() == 3 and boxes.size(1) == 4 and boxes.size(2) == 2:
        boxes_ = boxes
    else:
        raise ValueError(f"Unexpected boxes shape: {boxes.shape}")

    scale = torch.tensor([W - 1, H - 1], device=device).float()

    for pts in boxes_:
        p = pts.float() * scale  # [4,2] pixels

        center = p.mean(dim=0)
        cx_f, cy_f = center[0], center[1]

        # adjacent edges
        e01 = torch.norm(p[1] - p[0])
        e12 = torch.norm(p[2] - p[1])
        long_len  = torch.maximum(e01, e12)
        short_len = torch.minimum(e01, e12)

        # angle aligned to long edge
        vec = (p[1] - p[0]) if (e01 >= e12) else (p[2] - p[1])
        theta = torch.atan2(vec[1], vec[0])
        cos_2t = torch.cos(2 * theta)
        sin_2t = torch.sin(2 * theta)

        norm_long  = torch.clamp(long_len / max(H, W), 0.0, 1.0)
        norm_short = torch.clamp(short_len / max(H, W), 0.0, 1.0)

        # gaussian quality
        dist_sq = (xx - cx_f) ** 2 + (yy - cy_f) ** 2
        g = torch.exp(-dist_sq / (2 * sigma ** 2))
        quality[0] = torch.maximum(quality[0], g)

        mask_region = g > 0.1
        if mask_region.any():
            angle[0][mask_region] = cos_2t
            angle[1][mask_region] = sin_2t
            width[0][mask_region] = norm_long
            thick[0][mask_region] = norm_short
            aw_mask[0][mask_region] = 1.0

    return {"quality": quality, "angle": angle, "width": width, "thick": thick, "aw_mask": aw_mask}


# -------------------------
# Visualization helpers
# -------------------------
def draw_grasp_box(ax, box, color="red"):
    box = box.detach().cpu().numpy()
    xs = list(box[:, 0]) + [box[0, 0]]
    ys = list(box[:, 1]) + [box[0, 1]]
    ax.plot(xs, ys, color=color, linewidth=2)


def decode_grasp(pred, topk=1, q_thresh=0.2, min_side_px=6.0):
    """
    Use predicted thick for short side.
    pred:
      quality: [1,H,W] (should already be in [0,1])
      angle:   [2,H,W]
      width:   [1,H,W] normalized LONG
      thick:   [1,H,W] normalized SHORT
    """
    q = pred["quality"][0]  # [H,W]
    a = pred["angle"]       # [2,H,W]
    w = pred["width"][0]    # [H,W]
    t = pred["thick"][0]    # [H,W]

    H, W = q.shape
    device = q.device

    padding = 2
    local_max = F.max_pool2d(q.unsqueeze(0), kernel_size=5, stride=1, padding=padding)[0]
    is_local_max = (q == local_max) & (q > q_thresh)
    idxs = torch.nonzero(is_local_max, as_tuple=False)

    if idxs.shape[0] == 0:
        return []

    if idxs.shape[0] > topk:
        q_vals = q[idxs[:, 0], idxs[:, 1]]
        _, top_k = torch.topk(q_vals, k=topk)
        idxs = idxs[top_k]

    boxes = []
    for (y_t, x_t) in idxs:
        y = int(y_t.item())
        x = int(x_t.item())
        cx, cy = float(x), float(y)

        sin2t = float(a[1, y, x].item())
        cos2t = float(a[0, y, x].item())
        theta = 0.5 * math.atan2(sin2t, cos2t)

        long_px  = float(w[y, x].item()) * max(H, W)
        short_px = float(t[y, x].item()) * max(H, W)

        long_px  = max(long_px,  min_side_px)
        short_px = max(short_px, min_side_px)

        dx = long_px / 2.0
        dy = short_px / 2.0

        R = torch.tensor([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)]
        ], device=device, dtype=torch.float32)

        corners = torch.tensor([
            [-dx, -dy],
            [ dx, -dy],
            [ dx,  dy],
            [-dx,  dy]
        ], device=device, dtype=torch.float32)

        rotated = corners @ R.T
        rotated[:, 0] += cx
        rotated[:, 1] += cy
        boxes.append(rotated)

    return boxes


def visualize_training(imgs, preds, boxes_list, step):
    os.makedirs("vis", exist_ok=True)
    imgs_cpu = imgs.detach().cpu()
    B = imgs_cpu.shape[0]

    for i in range(B):
        img = imgs_cpu[i].permute(1, 2, 0).numpy()

        pred_one = {
            "quality": preds["quality"][i],
            "angle":   preds["angle"][i],
            "width":   preds["width"][i],
            "thick":   preds["thick"][i],
        }

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img)
        ax.set_title(f"Step {step} | Image {i}")
        ax.set_xlim([0, img.shape[1]])
        ax.set_ylim([img.shape[0], 0])

        gt_boxes = boxes_list[i]
        if isinstance(gt_boxes, torch.Tensor):
            if gt_boxes.dim() == 1:
                gt_boxes = gt_boxes.unsqueeze(0)
            gt_boxes = gt_boxes.to(preds["quality"].device)

        for b in gt_boxes:
            quad = b.view(4, 2).clone()
            quad[:, 0] *= img.shape[1]
            quad[:, 1] *= img.shape[0]
            draw_grasp_box(ax, quad, color="lime")

        pred_boxes = decode_grasp(pred_one, topk=1, q_thresh=0.2, min_side_px=6.0)
        for pb in pred_boxes:
            draw_grasp_box(ax, pb, color="red")

        fig.savefig(f"vis/step{step:04d}_img{i}.png", dpi=150)
        plt.close(fig)


# -------------------------
# Loss helpers
# -------------------------
def endpoints_from_maps(cx, cy, theta, long_px):
    ux = math.cos(theta)
    uy = math.sin(theta)
    half = 0.5 * long_px
    top = (cx - half * ux, cy - half * uy)
    bot = (cx + half * ux, cy + half * uy)
    return top, bot


def maybe_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    If model already applies sigmoid (values mostly in [0,1]), keep it.
    Else apply sigmoid.
    """
    # cheap heuristic: if max <= 1.2 and min >= -0.2, assume already sigmoid
    with torch.no_grad():
        mx = float(x.max().item())
        mn = float(x.min().item())
    if mx <= 1.2 and mn >= -0.2:
        return x
    return torch.sigmoid(x)


class GraspLoss(nn.Module):
    def __init__(self, w_q=10.0, w_a=1.0, w_w=1.0, w_t=1.0, w_end=0.3):
        super().__init__()
        self.w_q = w_q
        self.w_a = w_a
        self.w_w = w_w
        self.w_t = w_t
        self.w_end = w_end

    def forward(self, pred, gt):
        q_pred = pred["quality"]
        q_gt   = gt["quality"]

        # weighted quality mse
        pos = (q_gt > 0.1).float()
        w_map = 1.0 + 5.0 * pos
        q_loss = ((q_pred - q_gt) ** 2 * w_map).mean()

        mask = gt["aw_mask"]              # [B,1,H,W]
        mask2 = mask.repeat(1, 2, 1, 1)   # [B,2,H,W]

        a_loss = F.smooth_l1_loss(pred["angle"] * mask2, gt["angle"] * mask2,
                                  reduction="sum") / (mask2.sum() + 1e-6)

        w_loss = F.smooth_l1_loss(pred["width"] * mask, gt["width"] * mask,
                                  reduction="sum") / (mask.sum() + 1e-6)

        t_loss = F.smooth_l1_loss(pred["thick"] * mask, gt["thick"] * mask,
                                  reduction="sum") / (mask.sum() + 1e-6)

        # endpoint loss at GT peak (one point per sample)
        B, _, H, W = q_gt.shape
        end_losses = []

        for b in range(B):
            flat_idx = torch.argmax(q_gt[b, 0]).item()
            gy = flat_idx // W
            gx = flat_idx % W

            # GT theta + long
            gt_cos2 = float(gt["angle"][b, 0, gy, gx].item())
            gt_sin2 = float(gt["angle"][b, 1, gy, gx].item())
            gt_theta = 0.5 * math.atan2(gt_sin2, gt_cos2)
            gt_long = float(gt["width"][b, 0, gy, gx].item()) * max(H, W)

            # Pred theta + long (at same location)
            pr_cos2 = float(pred["angle"][b, 0, gy, gx].item())
            pr_sin2 = float(pred["angle"][b, 1, gy, gx].item())
            pr_theta = 0.5 * math.atan2(pr_sin2, pr_cos2)
            pr_long = float(pred["width"][b, 0, gy, gx].item()) * max(H, W)

            gt_top, gt_bot = endpoints_from_maps(gx, gy, gt_theta, gt_long)
            pr_top, pr_bot = endpoints_from_maps(gx, gy, pr_theta, pr_long)

            gt_pts = torch.tensor([gt_top[0], gt_top[1], gt_bot[0], gt_bot[1]],
                                  device=q_gt.device, dtype=torch.float32)
            pr_pts = torch.tensor([pr_top[0], pr_top[1], pr_bot[0], pr_bot[1]],
                                  device=q_gt.device, dtype=torch.float32)
            end_losses.append(F.smooth_l1_loss(pr_pts, gt_pts))

        end_loss = torch.stack(end_losses).mean() if len(end_losses) else torch.tensor(0.0, device=q_gt.device)

        return (self.w_q * q_loss +
                self.w_a * a_loss +
                self.w_w * w_loss +
                self.w_t * t_loss +
                self.w_end * end_loss)


# -------------------------
# Train loop
# -------------------------
def train():
    cfg = TrainConfig()
    model = GraspNetLAG().to(cfg.device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()

    criterion = GraspLoss(w_q=10.0, w_a=1.0, w_w=1.0, w_t=1.0, w_end=0.3)

    train_set = GraspTxtDataset(cfg.train_images, cfg.train_labels, image_size=cfg.image_size)
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    best_loss = float("inf")

    print("Start Training (Long+Short supervision with thick head)...")

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")

        for step, (imgs, boxes_list) in enumerate(pbar):
            imgs = imgs.to(cfg.device)

            with torch.no_grad():
                gts = [boxes_to_gt_maps(b.to(cfg.device), (cfg.image_size, cfg.image_size)) for b in boxes_list]
                gt_batch = {k: torch.stack([g[k] for g in gts], dim=0) for k in gts[0]}

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                preds = model(imgs)

                # model.py 里可能已经 sigmoid，这里做兼容
                preds["quality"] = maybe_sigmoid(preds["quality"])
                preds["width"]   = maybe_sigmoid(preds["width"])
                preds["thick"]   = maybe_sigmoid(preds["thick"])
                # angle 不 sigmoid

                loss = criterion(preds, gt_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.5f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

            if step % 50 == 0:
                visualize_training(imgs, preds, boxes_list, step + epoch * len(loader))

        scheduler.step()
        avg_loss = total_loss / max(1, len(loader))

        torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, "latest.pt"))
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, "best.pt"))

        print(f"[Epoch {epoch + 1}] Avg Loss: {avg_loss:.5f} (Best: {best_loss:.5f})")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train()
