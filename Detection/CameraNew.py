import sys
import time
import argparse
import numpy as np
import cv2
import torch
from typing import Tuple, Optional

# ======== 你的 LAG 相关 ========
from LAGNetwork.model import GraspNetLAG
from LAGNetwork.train import decode_maps_to_boxes_norm

# ======== 你的事件发布器 ========
from event_pub import B2Z1CmdPublisher

# ======== Metavision 相关 ========
sys.path.append("/usr/local/local/lib/python3.8/dist-packages/")

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIKeyEvent


# =========================
# 工具函数
# =========================
def _strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def load_checkpoint_robust(ckpt_path: str, map_location="cpu") -> dict:
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
        state_dict = checkpoint["model"]
        cfg = checkpoint.get("cfg", {})
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        state_dict = checkpoint
        cfg = {}
    else:
        raise ValueError(f"Unsupported checkpoint format: type={type(checkpoint)}")
    state_dict = _strip_module_prefix(state_dict)
    return {"raw": checkpoint, "state_dict": state_dict, "cfg": cfg}


def quad_to_aabb(pts: np.ndarray) -> np.ndarray:
    x1, y1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    x2, y2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou_aabb(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (areaA + areaB - inter + 1e-6))


def preprocess_frame(frame, input_size, device):
    frame_resized = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = frame_resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)).copy()
    return torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)


def quad_norm_to_aabb_px(boxes_norm: torch.Tensor, w: int, h: int):
    """
    boxes_norm: (1, 8) tensor, quad in normalized coords [0,1]
    return: (x1,y1,x2,y2,cx,cy) in pixel coords (int)
    """
    pts = boxes_norm[0].reshape(4, 2).detach().cpu().numpy()
    pts[:, 0] *= (w - 1)
    pts[:, 1] *= (h - 1)

    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0]))
    y2 = int(np.max(pts[:, 1]))

    cx = int((x1 + x2) * 0.5)
    cy = int((y1 + y2) * 0.5)
    return x1, y1, x2, y2, cx, cy


def draw_rect_center_and_corners(
    frame: np.ndarray,
    boxes_norm: Optional[torch.Tensor],
    status: str
) -> Tuple[np.ndarray, Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    画 AABB 矩形框 + 中心点，并返回:
      - center: (cx, cy)
      - tl: (x1, y1)  左上角
      - br: (x2, y2)  右下角
    """
    if boxes_norm is None:
        return frame, None, None, None

    h, w = frame.shape[:2]
    x1, y1, x2, y2, cx, cy = quad_norm_to_aabb_px(boxes_norm, w, h)

    color = (0, 255, 0) if "Tracking" in status else (0, 0, 255)
    if "Rejected" in status or "JumpBlocked" in status:
        color = (0, 255, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

    return frame, (cx, cy), (x1, y1), (x2, y2)


# =========================
# Kalman
# =========================
class SimpleCenterKalman:
    """State: [cx, cy, vx, vy], Measurement: [cx, cy]"""

    def __init__(self):
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 100.0
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)

        self.Q = np.eye(4, dtype=np.float32) * 2.0
        self.R = np.eye(2, dtype=np.float32) * 3.0

    def init(self, cx: float, cy: float):
        self.x[:] = 0
        self.x[0, 0] = cx
        self.x[1, 0] = cy
        self.P = np.eye(4, dtype=np.float32) * 10.0

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, cx: float, cy: float) -> np.ndarray:
        z = np.array([[cx], [cy]], dtype=np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P
        return self.x.copy()


# =========================
# Tracker
# =========================
class RobustGraspTracker:
    def __init__(self, model, device, input_size=320, conf_thresh=0.15):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.smooth_factor = 0.35

        self.is_tracking = False
        self.prev_gray = None
        self.prev_pts = None
        self.current_box_pts = None
        self.base_shape = None

        self.last_detect_time = 0.0
        self.detect_interval = 0.6
        self.gate_iou = 0.45
        self.gate_center_ratio = 0.18

        self.fb_thr = 1.2
        self.min_inlier_ratio = 0.40
        self.max_med_fb = 2.0

        self.min_pts_reseed = 40
        self.min_pts_restart = 12
        self.max_pts_keep = 120

        self.lk_win = (61, 61)
        self.lk_level = 4
        self.lk_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)

        self.kf = SimpleCenterKalman()
        self.kf_inited = False

        self.max_bad_count = 25
        self.bad_count = 0

        self.track_center = None
        self.target_center = None
        self.max_speed_ratio = 0.06

        self.pending_click = None
        self.pending_hard_reset_kf = True
        self.click_flash_until = 0.0

        self.last_gray = None
        self.last_hw = None

    def set_manual_target(self, x: float, y: float, hard_reset_kf: bool = True):
        self.pending_click = np.array([float(x), float(y)], dtype=np.float32)
        self.pending_hard_reset_kf = hard_reset_kf
        self.bad_count = 0
        self.click_flash_until = time.time() + 0.8

    def clear_manual_target(self):
        self.pending_click = None

    def _box_diag(self, box_pts: np.ndarray) -> float:
        return float(np.linalg.norm(box_pts[0] - box_pts[2]))

    def _adaptive_jump_thr(self, box_pts: np.ndarray) -> float:
        diag = self._box_diag(box_pts)
        return max(10.0, 0.10 * diag)

    def _max_step(self, box_pts: np.ndarray) -> float:
        diag = self._box_diag(box_pts)
        base = max(6.0, self.max_speed_ratio * diag)
        base *= float(np.clip(self.smooth_factor, 0.05, 1.0))
        return max(1.0, base)

    def _slew_to_target(self, current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
        d = target - current
        dist = float(np.linalg.norm(d))
        if dist <= max_step or dist < 1e-6:
            return target
        return current + d / dist * max_step

    def _get_center(self, box_pts: np.ndarray) -> np.ndarray:
        return np.mean(box_pts, axis=0).astype(np.float32)

    def _reseed_points(self, frame_gray: np.ndarray, box_pts: np.ndarray) -> Optional[np.ndarray]:
        mask = np.zeros_like(frame_gray)
        cv2.fillConvexPoly(mask, box_pts.astype(np.int32), 255)
        pts = cv2.goodFeaturesToTrack(
            frame_gray, mask=mask,
            maxCorners=self.max_pts_keep,
            qualityLevel=0.02,
            minDistance=6,
            blockSize=7
        )
        if pts is None or len(pts) == 0:
            return None
        return pts.astype(np.float32)

    def _pts_to_norm_tensor(self, pts, W, H):
        pts_copy = pts.copy()
        pts_copy[:, 0] /= (W - 1)
        pts_copy[:, 1] /= (H - 1)
        return torch.from_numpy(pts_copy.flatten()).unsqueeze(0).to(self.device).float()

    def _apply_click_recenter(self, frame_gray: np.ndarray, H: int, W: int) -> bool:
        if self.pending_click is None:
            return False

        c = self.pending_click.copy()
        hard_reset = self.pending_hard_reset_kf
        self.pending_click = None

        if self.base_shape is None:
            side = 80.0
            half = side * 0.5
            box = np.array([
                [c[0] - half, c[1] - half],
                [c[0] + half, c[1] - half],
                [c[0] + half, c[1] + half],
                [c[0] - half, c[1] + half],
            ], dtype=np.float32)
            box[:, 0] = np.clip(box[:, 0], 0, W - 1)
            box[:, 1] = np.clip(box[:, 1], 0, H - 1)
            self.current_box_pts = box
            cc = np.mean(self.current_box_pts, axis=0).astype(np.float32)
            self.base_shape = self.current_box_pts - cc

        self.track_center = c.copy()
        self.target_center = c.copy()
        self.current_box_pts = (self.base_shape + self.track_center).astype(np.float32)

        if hard_reset or (not self.kf_inited):
            self.kf.init(float(c[0]), float(c[1]))
            self.kf_inited = True
        else:
            self.kf.update(float(c[0]), float(c[1]))

        pts = self._reseed_points(frame_gray, self.current_box_pts)
        self.prev_gray = frame_gray
        self.last_detect_time = time.time()
        self.bad_count = 0

        if pts is not None and len(pts) >= self.min_pts_restart:
            self.prev_pts = pts.astype(np.float32)
            self.is_tracking = True
        else:
            self.prev_pts = None
            self.is_tracking = False

        return True

    def get_grasp(self, frame_orig: np.ndarray) -> Tuple[Optional[torch.Tensor], str]:
        curr_time = time.time()
        frame_gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        H, W = frame_orig.shape[:2]

        self.last_gray = frame_gray
        self.last_hw = (H, W)

        if self._apply_click_recenter(frame_gray, H, W):
            box_norm = self._pts_to_norm_tensor(self.current_box_pts, W, H)
            return box_norm, "Tracking (Recentered by Click)"

        if (
            self.is_tracking
            and self.prev_pts is not None
            and self.prev_gray is not None
            and self.current_box_pts is not None
            and self.base_shape is not None
        ):
            if curr_time - self.last_detect_time > self.detect_interval:
                det_box_norm, det_status = self._run_detection(frame_orig, frame_gray, allow_reject=True)
                return det_box_norm, det_status

            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, frame_gray, self.prev_pts, None,
                winSize=self.lk_win, maxLevel=self.lk_level, criteria=self.lk_crit
            )
            if next_pts is None or status is None:
                return self._run_detection(frame_orig, frame_gray, allow_reject=False)

            back_pts, _, _ = cv2.calcOpticalFlowPyrLK(
                frame_gray, self.prev_gray, next_pts, None,
                winSize=self.lk_win, maxLevel=self.lk_level, criteria=self.lk_crit
            )
            if back_pts is None:
                return self._run_detection(frame_orig, frame_gray, allow_reject=False)

            fb_err = np.linalg.norm(self.prev_pts - back_pts, axis=2).ravel()
            valid_mask = (status.ravel() == 1) & (fb_err < self.fb_thr)

            inlier_ratio = float(np.mean(valid_mask)) if len(valid_mask) > 0 else 0.0
            med_fb = float(np.median(fb_err[valid_mask])) if np.any(valid_mask) else 1e9

            if inlier_ratio < self.min_inlier_ratio or med_fb > self.max_med_fb:
                return self._run_detection(frame_orig, frame_gray, allow_reject=False)

            good_prev = self.prev_pts[valid_mask]
            good_next = next_pts[valid_mask]

            if len(good_next) < self.min_pts_restart:
                return self._run_detection(frame_orig, frame_gray, allow_reject=False)

            disp = (good_next - good_prev).reshape(-1, 2)
            dx, dy = np.median(disp, axis=0)

            prev_center = self._get_center(self.current_box_pts)
            obs_center = prev_center + np.array([dx, dy], dtype=np.float32)

            dist = float(np.linalg.norm(obs_center - prev_center))
            jump_thr = self._adaptive_jump_thr(self.current_box_pts)
            obs_ok = (dist < jump_thr)

            if not self.kf_inited:
                self.kf.init(float(prev_center[0]), float(prev_center[1]))
                self.kf_inited = True

            kf_pred = self.kf.predict()
            kf_cx, kf_cy = float(kf_pred[0, 0]), float(kf_pred[1, 0])

            if obs_ok:
                kf_upd = self.kf.update(float(obs_center[0]), float(obs_center[1]))
                kf_cx, kf_cy = float(kf_upd[0, 0]), float(kf_upd[1, 0])

            kf_center = np.array([kf_cx, kf_cy], dtype=np.float32)

            if self.track_center is None:
                self.track_center = prev_center.copy()

            self.target_center = kf_center

            step = self._max_step(self.current_box_pts)
            self.track_center = self._slew_to_target(self.track_center, self.target_center, step)
            self.current_box_pts = (self.base_shape + self.track_center).astype(np.float32)

            if obs_ok:
                self.bad_count = 0
                status_str = f"Tracking (TranslateOnly Slew, inlier={inlier_ratio:.2f}, fb={med_fb:.2f})"
            else:
                self.bad_count += 1
                status_str = f"Tracking (JumpBlocked {self.bad_count}/{self.max_bad_count}, thr={jump_thr:.1f})"
                if self.bad_count >= self.max_bad_count:
                    return self._run_detection(frame_orig, frame_gray, allow_reject=False)

            if len(good_next) < self.min_pts_reseed:
                new_pts = self._reseed_points(frame_gray, self.current_box_pts)
                if new_pts is not None:
                    merged = np.vstack([good_next.reshape(-1, 1, 2), new_pts])
                    self.prev_pts = merged[:self.max_pts_keep].astype(np.float32)
                else:
                    self.prev_pts = good_next.reshape(-1, 1, 2).astype(np.float32)
            else:
                self.prev_pts = good_next.reshape(-1, 1, 2).astype(np.float32)

            self.prev_gray = frame_gray
            box_norm = self._pts_to_norm_tensor(self.current_box_pts, W, H)
            return box_norm, status_str

        return self._run_detection(frame_orig, frame_gray, allow_reject=False)

    def _run_detection(self, frame: np.ndarray, frame_gray: np.ndarray, allow_reject: bool = False) -> Tuple[Optional[torch.Tensor], str]:
        tensor = preprocess_frame(frame, self.input_size, self.device)
        with torch.no_grad():
            pred = self.model(tensor)

        pred_one = {"quality": pred["quality"][0], "angle": pred["angle"][0], "width": pred["width"][0]}
        boxes_norm = decode_maps_to_boxes_norm(pred_one, topk=1, q_thresh=self.conf_thresh)

        H, W = frame.shape[:2]
        if boxes_norm is None or boxes_norm.numel() == 0:
            if self.is_tracking and allow_reject and self.current_box_pts is not None:
                self.last_detect_time = time.time()
                return self._pts_to_norm_tensor(self.current_box_pts, W, H), "Tracking (Detect None)"
            else:
                self._reset()
                return None, "Searching"

        det_box = boxes_norm[0].reshape(4, 2).detach().cpu().numpy()
        det_box[:, 0] *= (W - 1)
        det_box[:, 1] *= (H - 1)
        det_box = det_box.astype(np.float32)
        det_center = np.mean(det_box, axis=0).astype(np.float32)

        if allow_reject and self.is_tracking and self.current_box_pts is not None:
            prev = self.current_box_pts
            prev_center = np.mean(prev, axis=0).astype(np.float32)
            center_dist = float(np.linalg.norm(prev_center - det_center))

            prev_aabb = quad_to_aabb(prev)
            det_aabb = quad_to_aabb(det_box)
            iou = iou_aabb(prev_aabb, det_aabb)

            diag = float(np.linalg.norm(prev[0] - prev[2]))
            ok = (center_dist < self.gate_center_ratio * max(1.0, diag)) or (iou > self.gate_iou)

            if not ok:
                self.last_detect_time = time.time()
                box_norm = self._pts_to_norm_tensor(self.current_box_pts, W, H)
                return box_norm, f"Tracking (Detect Rejected, dist={center_dist:.1f}, iou={iou:.2f})"

        if self.base_shape is None:
            self.current_box_pts = det_box
            c = np.mean(self.current_box_pts, axis=0).astype(np.float32)
            self.base_shape = self.current_box_pts - c
            self.track_center = c.copy()
            self.target_center = c.copy()
            fused_center = c
        else:
            if self.is_tracking and self.current_box_pts is not None:
                prev_center = np.mean(self.current_box_pts, axis=0).astype(np.float32)
                detect_blend = 0.75
                fused_center = detect_blend * prev_center + (1 - detect_blend) * det_center
            else:
                fused_center = det_center

            if self.track_center is None:
                self.track_center = np.mean(self.current_box_pts, axis=0).astype(np.float32)

            self.target_center = fused_center.copy()
            step = self._max_step(self.current_box_pts if self.current_box_pts is not None else det_box)
            self.track_center = self._slew_to_target(self.track_center, self.target_center, step)
            self.current_box_pts = (self.base_shape + self.track_center).astype(np.float32)

        if not self.kf_inited:
            self.kf.init(float(fused_center[0]), float(fused_center[1]))
            self.kf_inited = True
        else:
            self.kf.update(float(fused_center[0]), float(fused_center[1]))

        pts = self._reseed_points(frame_gray, self.current_box_pts)
        self.last_detect_time = time.time()

        if pts is not None and len(pts) >= self.min_pts_restart:
            self.prev_pts = pts.astype(np.float32)
            self.prev_gray = frame_gray
            self.is_tracking = True
            self.bad_count = 0
            return self._pts_to_norm_tensor(self.current_box_pts, W, H), "Model Detect (Slew TranslateOnly)"

        self.is_tracking = False
        self.prev_pts = None
        self.prev_gray = None
        self.bad_count = 0
        return self._pts_to_norm_tensor(self.current_box_pts, W, H), "Model Detect (No Points)"

    def _reset(self):
        self.is_tracking = False
        self.prev_pts = None
        self.prev_gray = None
        self.current_box_pts = None
        self.base_shape = None
        self.bad_count = 0
        self.kf_inited = False
        self.track_center = None
        self.target_center = None


# =========================
# 主程序
# =========================
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-i", "--input-event-file", dest="event_file_path", default="",
        help="RAW/DAT/HDF5 path. If empty -> live camera. If serial -> open that camera."
    )
    parser.add_argument("--ckpt", type=str, default="/home/wangzhe/ICME2026/ckpt_lag/l/last.pt")
    parser.add_argument("--out", type=str, default="output_overlay.avi")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--speed-ratio", type=float, default=0.06, help="slew max speed ratio")
    parser.add_argument("--smooth", type=float, default=0.35,
                        help="center move smoothing factor in (0,1], smaller=slower")
    parser.add_argument("--pub-hz", type=float, default=20.0, help="publish bbox rate (Hz)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- device & model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint_robust(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    base_channels = cfg.get("base_channels", 32)

    model = GraspNetLAG(in_channels=3, base_channels=base_channels)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if len(missing) > 0:
        print("[Warning] Missing keys:", missing[:20], "..." if len(missing) > 20 else "")
    if len(unexpected) > 0:
        print("[Warning] Unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")

    model.to(device).eval()

    tracker = RobustGraspTracker(model, device, input_size=args.input_size, conf_thresh=args.conf)
    tracker.max_speed_ratio = args.speed_ratio
    tracker.smooth_factor = args.smooth

    # ✅ Publisher：只初始化一次，后面回调里复用
    publisher = B2Z1CmdPublisher("/event/detection")
    last_pub_t = 0.0
    pub_interval = 1.0 / max(0.1, float(args.pub_hz))

    # ---- Metavision iterator ----
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=1000)
    height, width = mv_iterator.get_size()

    if not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    # ---- OpenCV window for mouse click ----
    cv2_win_name = "Click To Recenter (L-click recenter / R-click clear pending)"
    cv2.namedWindow(cv2_win_name, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            tracker.set_manual_target(x, y, hard_reset_kf=True)
            print(f"\n[Click Recenter] pending to x={x}, y={y}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            tracker.clear_manual_target()
            print("\n[Click Recenter] pending cleared")

    cv2.setMouseCallback(cv2_win_name, on_mouse)

    # ---- Video writer ----
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = None
    is_recording = False

    # ---- UI flags ----
    enable_detect_track = False
    debounce_sec = 0.25
    last_key_time = {}

    with MTWindow(
        title="Metavision + LAG Grasp Tracking (Press V to Start/Stop)",
        width=width,
        height=height,
        mode=BaseWindow.RenderMode.BGR
    ) as window:

        def _start_detect_and_record():
            nonlocal enable_detect_track, is_recording, video_writer
            enable_detect_track = True
            tracker._reset()
            print("▶️ V: Detect/Track ON + Recording ON")

            if not is_recording:
                video_writer = cv2.VideoWriter(args.out, fourcc, float(args.fps), (width, height))
                if not video_writer.isOpened():
                    video_writer = None
                    is_recording = False
                    print(f"[Error] Cannot open VideoWriter: {args.out}")
                    return
                is_recording = True
                print(f"🎥 Recording started... -> {args.out}")

        def _stop_detect_and_record():
            nonlocal enable_detect_track, is_recording, video_writer
            enable_detect_track = False
            tracker._reset()
            print("⏹️ V: Detect/Track OFF + Recording OFF")

            if is_recording:
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                is_recording = False
                print(f"✅ Video saved to {args.out}")

        def keyboard_cb(key, scancode, action, mods):
            nonlocal enable_detect_track, last_key_time
            now = time.time()

            def debounced(k, dt=debounce_sec):
                t = last_key_time.get(k, 0.0)
                if now - t < dt:
                    return True
                last_key_time[k] = now
                return False

            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                if debounced(key, dt=0.15):
                    return
                if enable_detect_track or is_recording:
                    _stop_detect_and_record()
                window.set_close_flag()
                return

            if key == UIKeyEvent.KEY_V:
                if debounced(UIKeyEvent.KEY_V, dt=debounce_sec):
                    return
                if not enable_detect_track:
                    _start_detect_and_record()
                else:
                    _stop_detect_and_record()

        window.set_keyboard_callback(keyboard_cb)

        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width,
            sensor_height=height,
            fps=float(args.fps),
            palette=ColorPalette.CoolWarm
        )

        def on_cd_frame_cb(ts, cd_frame):
            nonlocal is_recording, video_writer, enable_detect_track, last_pub_t

            frame = cd_frame  # BGR uint8

            if enable_detect_track:
                box, status = tracker.get_grasp(frame)
                frame, center, tl, br = draw_rect_center_and_corners(frame, box, status)

                cv2.putText(frame, status, (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if center is not None:
                    cx, cy = center
                    cv2.putText(frame, f"Center: ({cx}, {cy})", (15, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if tl is not None and br is not None:
                    x1, y1 = tl
                    x2, y2 = br

                    cv2.putText(frame, f"TL: ({x1}, {y1})  BR: ({x2}, {y2})", (15, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # 终端输出
                    if center is not None:
                        cx, cy = center
                        print(f"\r[AABB] TL=({x1},{y1}) BR=({x2},{y2}) Center=({cx},{cy})   ",
                              end="", flush=True)
                    else:
                        print(f"\r[AABB] TL=({x1},{y1}) BR=({x2},{y2})   ",
                              end="", flush=True)

                    # ✅ 节流发送（避免每帧都发导致不“实时”或阻塞）
                    now = time.time()
                    if now - last_pub_t >= pub_interval:
                        publisher.publish(

                            bbox_x1=int(x1), bbox_y1=int(y1),
                            bbox_x2=int(x2), bbox_y2=int(y2)
                        )
                        last_pub_t = now

            else:
                cv2.putText(frame, "Detect/Track OFF (press V)", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # show
            window.show_async(frame)

            # show for mouse click
            cv2.imshow(cv2_win_name, frame)
            cv2.waitKey(1)

            # record overlay
            if is_recording and video_writer is not None:
                video_writer.write(frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)
            if window.should_close():
                break

    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
