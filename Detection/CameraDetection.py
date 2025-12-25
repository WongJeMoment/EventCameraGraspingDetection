import sys
import time
import numpy as np
import cv2
import argparse
import os
import torch
from typing import Tuple, Optional

# === Metavision SDK Imports ===
sys.path.append("/usr/local/local/lib/python3.8/dist-packages/")

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent

# === Neural Network Imports ===
from LAGNetwork.model import GraspNetLAG
from LAGNetwork.config import TrainConfig
from LAGNetwork.train import decode_maps_to_boxes_norm


# ==============================================================================
# 1. 终极稳定器 (加入缓冲逻辑，解决闪烁和跳跃)
# ==============================================================================
class GraspStabilizer:
    def __init__(self, alpha=0.25, dead_zone=0.03, patience=50, grace_period=4):
        """
        alpha: 0.25 -> 降低该值以获得极致的平滑度（解决跳跃）。配合高FPS不会觉得卡。
        grace_period: 4 -> "宽限期"。如果检测丢失在4帧以内，依然显示绿色，防止颜色乱闪。
        patience: 50 -> "记忆期"。超过宽限期后，变成黄色记忆框，持续约1秒。
        """
        self.alpha = alpha
        self.dead_zone = dead_zone
        self.patience = patience
        self.grace_period = grace_period  # 新增：防闪烁缓冲

        self.prev_box = None
        self.missed_counter = 0

    def update(self, current_boxes_norm: torch.Tensor) -> Tuple[Optional[torch.Tensor], bool]:
        """
        返回: (Box, is_prediction)
        is_prediction = True (黄色/记忆), False (绿色/实时)
        """
        # --- A. 检测到了物体 ---
        if current_boxes_norm is not None and current_boxes_norm.numel() > 0:
            best_new_box = current_boxes_norm[0]

            # 1. 初始化
            if self.prev_box is None:
                self.prev_box = best_new_box
                self.missed_counter = 0
                return self.prev_box.unsqueeze(0), False

            # 2. 计算距离
            dist = torch.norm(best_new_box - self.prev_box)

            # 3. 防抖 (死区)
            if dist < self.dead_zone:
                # 几乎没动，直接返回旧框，计数器归零
                self.missed_counter = 0
                return self.prev_box.unsqueeze(0), False

            # 4. 突变保护 (距离太远不平滑，直接跳)
            if dist > 0.4:
                self.prev_box = best_new_box
            else:
                # 5. 平滑更新 (EMA)
                # alpha 越小，历史权重越大，框越稳，不跳跃
                self.prev_box = self.alpha * best_new_box + (1 - self.alpha) * self.prev_box

            self.missed_counter = 0  # 重置丢失计数
            return self.prev_box.unsqueeze(0), False

        # --- B. 没检测到物体 (可能只是偶尔丢帧) ---
        else:
            if self.prev_box is None:
                return None, False

            self.missed_counter += 1

            # [核心修改] 宽限期逻辑
            # 如果丢失帧数很短 (比如 1~3 帧)，我们假装它还在 (is_prediction=False)
            # 这样颜色就不会变黄，视觉上就非常连续
            if self.missed_counter <= self.grace_period:
                return self.prev_box.unsqueeze(0), False  # <--- 依然返回 False (绿色)

            # 超过宽限期，进入记忆期 (显示黄色)
            elif self.missed_counter <= self.patience:
                return self.prev_box.unsqueeze(0), True  # <--- 变成 True (黄色)

            # 超时，彻底消失
            else:
                self.prev_box = None
                return None, False


# ==============================================================================
# Helper Functions
# ==============================================================================

def preprocess_simple(frame: np.ndarray, input_size: int, device: torch.device) -> torch.Tensor:
    # 高斯模糊稍微加强一点点 (5->7)，进一步抑制噪点跳动
    frame_blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    frame_resized = cv2.resize(frame_blurred, (input_size, input_size))
    img = frame_resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor


def draw_grasps(frame: np.ndarray, boxes_norm: torch.Tensor, is_prediction: bool):
    h_img, w_img = frame.shape[:2]
    if boxes_norm is None or boxes_norm.numel() == 0:
        return frame

    if is_prediction:
        color = (0, 255, 255)  # 黄色 (记忆/静止)
        thickness = 2
        # 画虚线效果模拟 (可选，这里保持实线简单点)
    else:
        color = (0, 255, 0)  # 绿色 (实时/缓冲中)
        thickness = 2

    boxes_np = boxes_norm.detach().cpu().numpy()
    for box in boxes_np:
        pts = box.reshape(4, 2)
        pts[:, 0] *= (w_img - 1)
        pts[:, 1] *= (h_img - 1)
        pts_int = pts.astype(np.int32)

        cv2.polylines(frame, [pts_int], isClosed=True, color=color, thickness=thickness)

        # 只有在非预测模式下才画方向点，看起来更干净
        if not is_prediction:
            cv2.circle(frame, tuple(pts_int[0]), 4, (0, 0, 255), -1)

    return frame


def parse_args():
    parser = argparse.ArgumentParser(description='Event Camera Grasp Detection')
    parser.add_argument('-i', '--input-event-file', dest='event_file_path', default="",
                        help="Path to input event file.")
    return parser.parse_args()


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    # --- 配置参数 ---
    ckpt_path = "/home/wangzhe/ICME2026/ckpt_lag/b/best.pt"

    conf_thresh = 0.12
    input_size = 320
    top_k = 1

    # 保持高 FPS，利用高刷新率来弥补低平滑系数带来的延迟感
    fps = 50

    # [参数微调]
    # alpha=0.25: 很低，意味着新检测只占25%权重，框会像在水里一样平滑游动，不跳。
    # grace_period=4: 连续丢4帧以内，不切黄色。
    stabilizer = GraspStabilizer(alpha=0.25, dead_zone=0.02, patience=50, grace_period=4)

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Inference Device: {device}")

    if not os.path.exists(ckpt_path):
        print(f"[Error] Checkpoint not found.")
        return

    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        base_channels = checkpoint.get("cfg", {}).get("base_channels", 32)
        model = GraspNetLAG(in_channels=3, base_channels=base_channels)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=1000)
    height, width = mv_iterator.get_size()

    if args.event_file_path and not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    state = {"detecting": False, "fps_avg": 0.0}

    with MTWindow(title="Smooth & Narrow", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.PRESS: return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()
            if key == UIKeyEvent.KEY_V:
                state["detecting"] = not state["detecting"]
                stabilizer.prev_box = None
                print(f"[User] Detection {'STARTED' if state['detecting'] else 'STOPPED'}")

        window.set_keyboard_callback(keyboard_cb)

        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height,
                                                           fps=fps, palette=ColorPalette.CoolWarm)

        def on_cd_frame_cb(ts, cd_frame):
            vis_frame = cd_frame

            if state["detecting"]:
                t0 = time.time()

                input_tensor = preprocess_simple(vis_frame, input_size, device)

                with torch.no_grad():
                    pred = model(input_tensor)

                pred_one = {
                    "quality": pred["quality"][0],
                    "angle": pred["angle"][0],
                    "width": pred["width"][0],
                }

                # 保持 1.0 的宽度缩放 (较窄的框)
                raw_boxes_norm = decode_maps_to_boxes_norm(
                    pred_one, topk=top_k, q_thresh=conf_thresh,
                    grasp_h_ratio=0.5,
                    width_scale=1.5
                )

                # 更新稳定器
                final_box, is_prediction = stabilizer.update(raw_boxes_norm)

                if final_box is not None:
                    draw_grasps(vis_frame, final_box, is_prediction=is_prediction)

                t1 = time.time()
                curr_fps = 1.0 / (t1 - t0 + 1e-6)
                state["fps_avg"] = 0.9 * state["fps_avg"] + 0.1 * curr_fps

                status_text = "Tracking" if not is_prediction else "Memory"
                color_text = (0, 255, 0) if not is_prediction else (0, 255, 255)

                cv2.putText(vis_frame, f"FPS: {state['fps_avg']:.1f} | {status_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_text, 2)
            else:
                cv2.putText(vis_frame, "Press 'v' to Detect", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            window.show_async(vis_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        print("\n=== Application Started ===")
        print("  System configured for Smoothness and Stability.")

        for evs in mv_iterator:
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)
            if window.should_close():
                break


if __name__ == "__main__":
    main()