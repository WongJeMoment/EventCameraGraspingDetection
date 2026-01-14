import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from LAGNetwork.model import GraspNetLAG
from LAGNetwork.config import TrainConfig
from LAGNetwork.train2 import decode_grasp  # thick 版 decode

# =========================
# 配置参数
# =========================
video_path  = '/home/wangzhe/ICME2026/MyDataset/Video/b1.avi'
output_path = '/home/wangzhe/ICME2026/MyDataset/Video/b122.mp4'  # mp4v 建议用 mp4 后缀
ckpt_path   = '/home/wangzhe/ICME2026/LAGNetwork/ckpt_lag/best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DRAW_BOX = True  # 是否画 grasp 框（红色）

# =========================
# 工具函数
# =========================
def maybe_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """避免重复 sigmoid（如果看起来已在 [0,1] 附近则不处理）"""
    with torch.no_grad():
        mx = float(x.max().item())
        mn = float(x.min().item())
    if mx <= 1.2 and mn >= -0.2:
        return x
    return torch.sigmoid(x)

def peak_center_from_quality(qmap_2d: torch.Tensor, subpixel=True):
    """
    qmap_2d: [H,W] in [0,1]
    return: (cx, cy, score) in network coords
    """
    H, W = qmap_2d.shape
    flat_idx = int(torch.argmax(qmap_2d).item())
    y = flat_idx // W
    x = flat_idx % W
    score = float(qmap_2d[y, x].item())

    if not subpixel:
        return float(x), float(y), score

    # 3x3 质心（亚像素）
    x0 = max(0, x - 1); x1 = min(W - 1, x + 1)
    y0 = max(0, y - 1); y1 = min(H - 1, y + 1)
    patch = qmap_2d[y0:y1+1, x0:x1+1].float()
    if patch.numel() == 0:
        return float(x), float(y), score

    ys, xs = torch.meshgrid(
        torch.arange(y0, y1 + 1, device=qmap_2d.device, dtype=torch.float32),
        torch.arange(x0, x1 + 1, device=qmap_2d.device, dtype=torch.float32),
        indexing='ij'
    )
    w = patch.clamp_min(1e-6)
    denom = float(w.sum().item())
    cx = float((xs * w).sum().item() / denom)
    cy = float((ys * w).sum().item() / denom)
    return cx, cy, score

class PointTrackerEMA:
    """中心点 EMA + 防跳 + 低分冻结"""
    def __init__(self, ema_alpha=0.15, max_jump=25.0, min_update_q=0.15):
        self.ema_alpha = float(ema_alpha)
        self.max_jump = float(max_jump)
        self.min_update_q = float(min_update_q)
        self.prev = None  # (x,y) network coords
        self.traj = []    # store original coords (we overwrite outside)

    def update(self, cx, cy, score):
        if self.prev is None:
            self.prev = (cx, cy)
            self.traj.append((cx, cy))
            return cx, cy

        px, py = self.prev

        if score < self.min_update_q:
            sx, sy = px, py
        else:
            dist = float(np.hypot(cx - px, cy - py))
            if dist > self.max_jump:
                sx, sy = px, py
            else:
                a = self.ema_alpha
                sx = (1 - a) * px + a * cx
                sy = (1 - a) * py + a * cy

        self.prev = (sx, sy)
        self.traj.append((sx, sy))
        return sx, sy

def box_center(pts_np: np.ndarray):
    return float(pts_np[:, 0].mean()), float(pts_np[:, 1].mean())

class BoxTracker:
    """
    稳定 grasp 矩形：
    - topk 候选里选最接近上一帧中心的
    - 没候选 / 大跳：沿用上一帧若干帧（不闪）
    - EMA 平滑中心并平移整框（简单但非常稳）
    """
    def __init__(self, max_jump=30.0, hold_frames=25, ema_alpha=0.25):
        self.max_jump = float(max_jump)
        self.hold_frames = int(hold_frames)
        self.ema_alpha = float(ema_alpha)

        self.prev_box = None
        self.prev_c = None
        self.miss = 0

    def update(self, cand_boxes):
        if len(cand_boxes) == 0:
            self.miss += 1
            if self.prev_box is not None and self.miss <= self.hold_frames:
                return self.prev_box
            return None

        self.miss = 0

        if self.prev_c is None:
            self.prev_box = cand_boxes[0]
            pts = cand_boxes[0].detach().cpu().numpy()
            self.prev_c = box_center(pts)
            return self.prev_box

        px, py = self.prev_c

        best, best_d = None, 1e9
        for b in cand_boxes:
            pts = b.detach().cpu().numpy()
            cx, cy = box_center(pts)
            d = float(np.hypot(cx - px, cy - py))
            if d < best_d:
                best_d = d
                best = b

        if best is None or best_d > self.max_jump:
            self.miss = min(self.miss + 1, self.hold_frames)
            return self.prev_box

        pts = best.detach().cpu().numpy()
        cx, cy = box_center(pts)

        a = self.ema_alpha
        sx = (1 - a) * px + a * cx
        sy = (1 - a) * py + a * cy

        dx, dy = sx - cx, sy - cy
        best_smooth = best.clone()
        best_smooth[:, 0] += dx
        best_smooth[:, 1] += dy

        self.prev_box = best_smooth
        self.prev_c = (sx, sy)
        return self.prev_box

def draw_traj(frame, traj_xy, color=(0,255,0), thickness=2, max_len=250):
    if len(traj_xy) < 2:
        return
    pts = np.array(traj_xy[-max_len:], dtype=np.float32)
    pts = np.round(pts).astype(np.int32)
    for i in range(1, len(pts)):
        cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]), color, thickness)

def estimate_fps(cap: cv2.VideoCapture, fallback=30.0, probe_frames=60):
    """尽量拿到靠谱 fps：先读 CAP_PROP_FPS，不靠谱就用时间戳估计"""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is not None and np.isfinite(fps) and fps > 1.0:
        return float(fps)

    # 兜底：用 POS_MSEC 估计
    ts = []
    pos0 = cap.get(cv2.CAP_PROP_POS_FRAMES)
    for _ in range(probe_frames):
        ok, _ = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        if np.isfinite(t) and t > 0:
            ts.append(float(t))

    # 回退到原位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)

    if len(ts) >= 2:
        dt = (ts[-1] - ts[0]) / 1000.0
        fps_est = (len(ts) - 1) / max(dt, 1e-6)
        if np.isfinite(fps_est) and fps_est > 1.0:
            return float(fps_est)

    return float(fallback)

# =========================
# 模型初始化
# =========================
cfg = TrainConfig()
model = GraspNetLAG().to(device)
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state, strict=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.ToTensor(),
])

# =========================
# 视频读取 & 写入初始化
# =========================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {video_path}")

fps = estimate_fps(cap, fallback=30.0, probe_frames=60)
W_out = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H_out = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] input fps = {fps:.3f}, size = {W_out}x{H_out}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_path, fourcc, fps, (W_out, H_out))
if not out_video.isOpened():
    raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

wait_time = max(1, int(round(1000.0 / fps)))  # 只影响显示，不影响写入fps

scale_x = W_out / cfg.image_size
scale_y = H_out / cfg.image_size

# 跟踪器
pt_tracker = PointTrackerEMA(ema_alpha=0.15, max_jump=25.0, min_update_q=0.15)
box_tracker = BoxTracker(max_jump=30.0, hold_frames=25, ema_alpha=0.05)

processing_started = False
print("🚦 按 'v' 开始检测视频帧，按 'q' 退出")

# =========================
# 主循环
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- 等待按键 ----------
    if not processing_started:
        show = frame.copy()
        cv2.putText(show, "Press 'v' to start detection", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(show, "Press 'q' to quit", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Waiting", show)
        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('v'):
            processing_started = True
            print("✅ 开始处理视频")
        elif key == ord('q'):
            print("🚪 提前退出")
            break
        continue

    # ---------- 推理 ----------
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(img_tensor)

        preds['quality'] = maybe_sigmoid(preds['quality'])
        preds['width']   = maybe_sigmoid(preds['width'])
        preds['thick']   = maybe_sigmoid(preds['thick'])

        pred_map = {
            'quality': preds['quality'][0],  # [1,H,W]
            'angle':   preds['angle'][0],    # [2,H,W]
            'width':   preds['width'][0],    # [1,H,W]
            'thick':   preds['thick'][0],    # [1,H,W]
        }

        # 1) quality peak 中心点
        q2d = pred_map['quality'][0]
        cx, cy, score = peak_center_from_quality(q2d, subpixel=True)

        # 2) 中心点平滑
        sx, sy = pt_tracker.update(cx, cy, score)

        # 3) decode 多候选框，减少空帧 + 减少跳
        cand_boxes = []
        if DRAW_BOX:
            cand_boxes = decode_grasp(
                pred_map,
                topk=10,
                q_thresh=0.10,
                min_side_px=6.0
            )

        # 4) 稳定输出一个框（可能沿用上一帧）
        stable_box = box_tracker.update(cand_boxes) if DRAW_BOX else None

    # ---------- 映射到原图坐标 ----------
    sx_o = sx * scale_x
    sy_o = sy * scale_y

    # 轨迹点（原图坐标存储）
    pt_tracker.traj[-1] = (sx_o, sy_o)

    # ---------- 可视化 ----------
    draw_traj(frame, pt_tracker.traj, color=(0, 255, 0), thickness=2, max_len=250)

    cv2.circle(frame, (int(round(sx_o)), int(round(sy_o))), 5, (0, 255, 0), -1)
    cv2.putText(frame, f"q={score:.3f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 稳定 grasp 框（红色）
    if DRAW_BOX and (stable_box is not None):
        pts = stable_box.detach().cpu().numpy()
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        pts = np.round(pts).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W_out - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H_out - 1)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

    # 写入输出视频（fps 与输入一致）
    out_video.write(frame)

    # 显示 & 按键
    cv2.imshow("Grasp Center Tracking", frame)
    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord('q'):
        print("🚪 提前退出")
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()
print(f"✅ 视频已保存到 {output_path}")
