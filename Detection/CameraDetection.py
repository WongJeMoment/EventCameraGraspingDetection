import sys
import time
import numpy as np
import cv2
import argparse
import os
import torch
from typing import Tuple

# === Metavision SDK Imports ===
# 如果你的安装路径不同，请调整这里的路径
sys.path.append("/usr/local/local/lib/python3.8/dist-packages/")

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent

# === Neural Network Imports ===
# 假设你的目录结构是:
# ├── main_combined.py (本文件)
# └── LAGNetwork/
#     ├── model.py
#     ├── config.py
#     └── train.py
from LAGNetwork.model import GraspNetLAG
from LAGNetwork.config import TrainConfig
from LAGNetwork.train import decode_maps_to_boxes_norm


# ==============================================================================
# Helper Functions (来自 Script A)
# ==============================================================================

def preprocess_frame(frame: np.ndarray, input_size: int, device: torch.device) -> torch.Tensor:
    """预处理：Resize -> Normalize -> ToTensor"""
    # 1. Resize
    frame_resized = cv2.resize(frame, (input_size, input_size))
    # 2. Normalize
    img = frame_resized.astype(np.float32) / 255.0
    # 3. HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    # 4. Batch Dim
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor


def draw_grasps(frame: np.ndarray, boxes_norm: torch.Tensor, color=(0, 255, 0), thickness=2):
    """绘制抓取框"""
    h_img, w_img = frame.shape[:2]
    if boxes_norm.numel() == 0:
        return frame

    boxes_np = boxes_norm.detach().cpu().numpy()
    for box in boxes_np:
        pts = box.reshape(4, 2)
        pts[:, 0] *= (w_img - 1)
        pts[:, 1] *= (h_img - 1)
        pts_int = pts.astype(np.int32)

        cv2.polylines(frame, [pts_int], isClosed=True, color=color, thickness=thickness)
        cv2.circle(frame, tuple(pts_int[0]), 4, (0, 0, 255), -1)  # 头部红点
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description='Event Camera Grasp Detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-event-file', dest='event_file_path', default="",
                        help="Path to input event file (RAW/DAT). Leave empty for Live Camera.")
    return parser.parse_args()


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    # --- 配置参数 ---
    ckpt_path = "/home/wangzhe/ICME2026/ckpt_lag/b/best.pt"
    conf_thresh = 0.15  # 置信度
    input_size = 320  # 模型输入大小
    top_k = 1  # 显示前 K 个抓取
    fps = 25  # 事件转视频的帧率

    args = parse_args()

    # 1. 初始化设备
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Inference Device: {device_name}")
    device = torch.device(device_name)

    # 2. 加载模型
    if not os.path.exists(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        return

    print(f"[Info] Loading Model from {ckpt_path} ...")
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

    # 3. 初始化 Metavision 事件流
    # 如果 args.event_file_path 为空，EventsIterator 会尝试打开第一个可用的相机
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=1000)
    height, width = mv_iterator.get_size()
    print(f"[Info] Sensor Size: {width}x{height}")

    # 如果是文件回放，包装一下以支持类似实时的流
    if args.event_file_path and not is_live_camera(args.event_file_path):
        mv_iterator = LiveReplayEventsIterator(mv_iterator)

    # 4. 状态控制变量
    state = {
        "detecting": False,
        "fps_avg": 0.0,
        "last_time": time.time()
    }

    # 5. 启动 Metavision 窗口和循环
    with MTWindow(title="Event Grasp Detection", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:

        # --- 键盘回调 ---
        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.PRESS:
                return

            # ESC / Q 退出
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

            # V 切换检测
            if key == UIKeyEvent.KEY_V:
                state["detecting"] = not state["detecting"]
                status = "STARTED" if state["detecting"] else "STOPPED"
                print(f"[User] Detection {status}")

        window.set_keyboard_callback(keyboard_cb)

        # --- 帧生成算法 ---
        # 将事件流累积并转换为图像 (Histogram/Edge image by default color palette)
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height,
                                                           fps=fps, palette=ColorPalette.CoolWarm)

        # --- 帧处理回调 (核心集成点) ---
        def on_cd_frame_cb(ts, cd_frame):
            """
            ts: 时间戳
            cd_frame: SDK 生成的 BGR 图像 (numpy array)
            """
            # 注意：cd_frame 是 SDK 内部管理的内存，如果需要大幅修改或保存，建议 copy()
            # 但为了实时绘制，我们直接在上面画图是最高效的（只要不改变尺寸）

            vis_frame = cd_frame  # 引用

            if state["detecting"]:
                t0 = time.time()

                # 1. 预处理
                input_tensor = preprocess_frame(vis_frame, input_size, device)

                # 2. 推理
                with torch.no_grad():
                    pred = model(input_tensor)

                # 3. 解码
                pred_one = {
                    "quality": pred["quality"][0],
                    "angle": pred["angle"][0],
                    "width": pred["width"][0],
                }

                boxes_norm = decode_maps_to_boxes_norm(
                    pred_one,
                    topk=top_k,
                    q_thresh=conf_thresh,
                    grasp_h_ratio=0.5,
                    width_scale=3.0
                )

                # 4. 绘图 (直接修改 vis_frame)
                draw_grasps(vis_frame, boxes_norm, color=(0, 255, 0), thickness=2)

                # 计算推理 FPS
                t1 = time.time()
                curr_fps = 1.0 / (t1 - t0 + 1e-6)
                # 平滑 FPS 显示
                state["fps_avg"] = 0.9 * state["fps_avg"] + 0.1 * curr_fps

                # UI 文字
                cv2.putText(vis_frame, f"Det FPS: {state['fps_avg']:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # 暂停状态提示
                cv2.putText(vis_frame, "Press 'v' to Detect", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 将处理后的帧显示在 Metavision 窗口中
            window.show_async(vis_frame)

        # 设置回调
        event_frame_gen.set_output_callback(on_cd_frame_cb)

        print("\n=== Application Started ===")
        print("  View the window. Press 'v' to toggle detection.")
        print("  Press 'q' or 'ESC' to exit.")

        # --- 主循环 ---
        for evs in mv_iterator:
            # 1. 分发 UI 事件 (鼠标、键盘)
            EventLoop.poll_and_dispatch()

            # 2. 处理事件生成图像 -> 触发回调 on_cd_frame_cb -> 触发神经网络 -> 显示
            event_frame_gen.process_events(evs)

            if window.should_close():
                break


if __name__ == "__main__":
    main()