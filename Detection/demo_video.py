import cv2
import torch
import numpy as np
import time
from typing import Tuple, Optional

from utils.WidthAdaption import shrink_grasp_width, shrink_grasp_length
from LAGNetwork.model import GraspNetLAG
from LAGNetwork.train import decode_maps_to_boxes_norm


# =========================
# Utils
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


def draw_grasps(frame, boxes_norm, status):
    if boxes_norm is None:
        return frame
    h, w = frame.shape[:2]
    pts = boxes_norm[0].reshape(4, 2).detach().cpu().numpy()
    pts[:, 0] *= (w - 1)
    pts[:, 1] *= (h - 1)

    # 你要缩短长度就保留这一行
    pts = shrink_grasp_length(pts, ratio=0.5).astype(np.int32)

    color = (0, 255, 0) if "Tracking" in status else (0, 0, 255)
    if "Rejected" in status or "JumpBlocked" in status:
        color = (0, 255, 255)

    cv2.polylines(frame, [pts], True, color, 2)
    cv2.circle(frame, tuple(pts[0]), 4, (255, 255, 255), -1)
    return frame


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
# Tracker (保持你原逻辑)
# =========================
class RobustGraspTracker:
    def __init__(self, model, device, input_size=320, conf_thresh=0.15):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.conf_thresh = conf_thresh

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

        self.max_rot_deg = 6.0
        self.min_scale = 0.97
        self.max_scale = 1.03
        self.max_trans_ratio = 0.10

    def _box_diag(self, box_pts: np.ndarray) -> float:
        return float(np.linalg.norm(box_pts[0] - box_pts[2]))

    def _adaptive_jump_thr(self, box_pts: np.ndarray) -> float:
        diag = self._box_diag(box_pts)
        return max(10.0, 0.10 * diag)

    def _max_trans_step(self, box_pts: np.ndarray) -> float:
        diag = self._box_diag(box_pts)
        return max(6.0, self.max_trans_ratio * diag)

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

    def _limit_similarity(self, A: np.ndarray, ref_box: np.ndarray) -> Optional[np.ndarray]:
        if A is None:
            return None
        a, b, tx = A[0]
        c, d, ty = A[1]

        scale = float(np.sqrt(a * a + c * c))
        theta = float(np.arctan2(c, a))

        max_theta = np.deg2rad(self.max_rot_deg)
        theta = float(np.clip(theta, -max_theta, max_theta))
        scale = float(np.clip(scale, self.min_scale, self.max_scale))

        max_t = self._max_trans_step(ref_box)
        t = np.array([tx, ty], dtype=np.float32)
        t_norm = float(np.linalg.norm(t))
        if t_norm > max_t:
            t = t / (t_norm + 1e-6) * max_t

        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], dtype=np.float32) * scale
        A_limited = np.zeros((2, 3), dtype=np.float32)
        A_limited[:, :2] = R
        A_limited[:, 2] = t
        return A_limited

    def get_grasp(self, frame_orig: np.ndarray) -> Tuple[Optional[torch.Tensor], str]:
        curr_time = time.time()
        frame_gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        H, W = frame_orig.shape[:2]

        if self.is_tracking and self.prev_pts is not None and self.prev_gray is not None and self.current_box_pts is not None:
            if curr_time - self.last_detect_time > self.detect_interval:
                return self._run_detection(frame_orig, frame_gray, allow_reject=True)

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

            good_prev = self.prev_pts[valid_mask].reshape(-1, 2)
            good_next = next_pts[valid_mask].reshape(-1, 2)

            if len(good_next) < self.min_pts_restart:
                return self._run_detection(frame_orig, frame_gray, allow_reject=False)

            A, inliers = cv2.estimateAffinePartial2D(
                good_prev, good_next,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
                confidence=0.99
            )
            if A is None:
                return self._run_detection(frame_orig, frame_gray, allow_reject=False)

            A = self._limit_similarity(A.astype(np.float32), self.current_box_pts)
            if A is None:
                return self._run_detection(frame_orig, frame_gray, allow_reject=False)

            ones = np.ones((self.current_box_pts.shape[0], 1), dtype=np.float32)
            pts_h = np.hstack([self.current_box_pts.astype(np.float32), ones])
            new_box = (pts_h @ A.T).astype(np.float32)

            prev_center = self._get_center(self.current_box_pts)
            new_center = self._get_center(new_box)
            dist = float(np.linalg.norm(new_center - prev_center))
            jump_thr = self._adaptive_jump_thr(self.current_box_pts)
            obs_ok = dist < jump_thr

            if not self.kf_inited:
                self.kf.init(float(prev_center[0]), float(prev_center[1]))
                self.kf_inited = True

            kf_pred = self.kf.predict()
            kf_cx, kf_cy = float(kf_pred[0, 0]), float(kf_pred[1, 0])
            if obs_ok:
                kf_upd = self.kf.update(float(new_center[0]), float(new_center[1]))
                kf_cx, kf_cy = float(kf_upd[0, 0]), float(kf_upd[1, 0])

            kf_center = np.array([kf_cx, kf_cy], dtype=np.float32)
            new_box = (new_box + (kf_center - new_center)[None, :]).astype(np.float32)

            if obs_ok:
                self.bad_count = 0
                self.current_box_pts = new_box
                status_str = f"Tracking (Similarity+KF, inlier={inlier_ratio:.2f}, fb={med_fb:.2f})"
            else:
                self.bad_count += 1
                status_str = f"Tracking (JumpBlocked {self.bad_count}/{self.max_bad_count}, thr={jump_thr:.1f})"
                if self.bad_count >= self.max_bad_count:
                    return self._run_detection(frame_orig, frame_gray, allow_reject=False)

            good_next_1x2 = next_pts[valid_mask].reshape(-1, 1, 2).astype(np.float32)
            if len(good_next_1x2) < self.min_pts_reseed:
                new_pts = self._reseed_points(frame_gray, self.current_box_pts)
                if new_pts is not None:
                    merged = np.vstack([good_next_1x2, new_pts])
                    self.prev_pts = merged[:self.max_pts_keep].astype(np.float32)
                else:
                    self.prev_pts = good_next_1x2
            else:
                self.prev_pts = good_next_1x2[:self.max_pts_keep]

            self.prev_gray = frame_gray
            return self._pts_to_norm_tensor(self.current_box_pts, W, H), status_str

        return self._run_detection(frame_orig, frame_gray, allow_reject=False)

    def _run_detection(self, frame: np.ndarray, frame_gray: np.ndarray, allow_reject: bool = False):
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
            self._reset()
            return None, "Searching"

        det_box = boxes_norm[0].reshape(4, 2).detach().cpu().numpy()
        det_box[:, 0] *= (W - 1)
        det_box[:, 1] *= (H - 1)
        det_box = det_box.astype(np.float32)
        det_center = np.mean(det_box, axis=0).astype(np.float32)

        # 这里保持你原来的“角点融合”逻辑（你若还遇到尺寸塌缩，建议改为中心融合）
        if self.current_box_pts is None:
            fused_box = det_box
        else:
            prev = self.current_box_pts
            prev_center = self._get_center(prev)

            iou = iou_aabb(quad_to_aabb(prev), quad_to_aabb(det_box))
            center_dist = float(np.linalg.norm(prev_center - det_center))
            diag = self._box_diag(prev) + 1e-6

            conf_iou = np.clip((iou - 0.2) / 0.6, 0.0, 1.0)
            conf_dist = np.clip(1.0 - center_dist / (0.25 * diag), 0.0, 1.0)
            det_conf = 0.6 * conf_iou + 0.4 * conf_dist

            alpha = float(np.clip(0.9 - 0.8 * det_conf, 0.1, 0.9))
            fused_box = (alpha * prev + (1 - alpha) * det_box).astype(np.float32)

        self.current_box_pts = fused_box
        fused_center = self._get_center(self.current_box_pts)

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
            return self._pts_to_norm_tensor(self.current_box_pts, W, H), "Model Detect"

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
        self.last_detect_time = 0.0


# =========================
# Key debounce helper
# =========================
class KeyDebouncer:
    """
    OpenCV waitKey 可能会因为按住/抖动导致多次返回同一个键值。
    这里做两层保护：
      1) 时间防抖：同一按键在 debounce_sec 内只触发一次
      2) 边沿触发：需要先释放（waitKey 返回 -1）再允许下一次触发
    """
    def __init__(self, debounce_sec: float = 0.25):
        self.debounce_sec = debounce_sec
        self.last_fire_time = {}
        self.armed = True  # 只有在看到 -1（松开）后才重新 armed

    def update(self, key: int) -> Optional[int]:
        now = time.time()

        if key == -1:
            self.armed = True
            return None

        if not self.armed:
            return None

        ch = key & 0xFF
        last_t = self.last_fire_time.get(ch, 0.0)
        if now - last_t < self.debounce_sec:
            return None

        self.last_fire_time[ch] = now
        self.armed = False
        return ch


# =========================
# Main (video)
# =========================
def main():
    ckpt_path = "/home/wangzhe/ICME2026/ckpt_lag/apple/best.pt"
    video_source = "/home/wangzhe/ICME2026/MyDataset/Video/apple1.avi"
    save_path = "output_similarity_adaptive.mp4"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint_robust(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    base_channels = cfg.get("base_channels", 32)

    model = GraspNetLAG(in_channels=3, base_channels=base_channels)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device).eval()

    tracker = RobustGraspTracker(model, device, input_size=320, conf_thresh=0.15)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_source}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, float(orig_fps), (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {save_path}")

    frame_duration_ms = 1000.0 / float(orig_fps)

    # ======= state =======
    detecting = False  # 默认不检测：按 v 才开始
    debouncer = KeyDebouncer(debounce_sec=0.25)

    print("Keys: [v] toggle detect ON/OFF, [q] quit")
    print("Start with detecting OFF. Press v to start.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        if detecting:
            box, status = tracker.get_grasp(frame)
            frame = draw_grasps(frame, box, status)
            cv2.putText(frame, f"{status}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Detecting: ON (press v to stop)", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Detecting: OFF (press v to start)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(frame)
        cv2.imshow("Grasp Tracking", frame)

        elapsed_ms = (time.time() - start_time) * 1000.0
        wait_time = max(1, int(frame_duration_ms - elapsed_ms))

        raw_key = cv2.waitKey(wait_time)
        key = debouncer.update(raw_key)

        if key is None:
            continue

        if key == ord('q'):
            break

        if key == ord('v'):
            detecting = not detecting
            tracker._reset()
            print(f"[Toggle] Detecting -> {detecting}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved -> {save_path}")


if __name__ == "__main__":
    main()
