import cv2
import torch
import numpy as np
import time
import os
from typing import Tuple

# ============================================================
# 导入你的项目模块
# 确保 demo_video.py 和 train.py, model.py 在同一目录
# ============================================================
from LAGNetwork.model import GraspNetLAG
from LAGNetwork.config import TrainConfig
# 我们直接复用 train.py 中的解码函数
from LAGNetwork.train import decode_maps_to_boxes_norm


def preprocess_frame(frame: np.ndarray, input_size: int, device: torch.device) -> Tuple[torch.Tensor, float, float]:
    """
    预处理视频帧：缩放 -> 归一化 -> 转 Tensor
    """
    # 1. 缩放图片到模型训练时的大小
    frame_resized = cv2.resize(frame, (input_size, input_size))

    # 2. 归一化 [0, 255] -> [0.0, 1.0]
    img = frame_resized.astype(np.float32) / 255.0

    # 3. HWC (OpenCV) -> CHW (PyTorch)
    img = np.transpose(img, (2, 0, 1))

    # 4. 增加 Batch 维度 -> (1, 3, H, W)
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    return tensor


def draw_grasps(frame: np.ndarray, boxes_norm: torch.Tensor, color=(0, 255, 0), thickness=2):
    """
    在原始帧上绘制抓取框
    """
    h_img, w_img = frame.shape[:2]

    if boxes_norm.numel() == 0:
        return frame

    # 转为 numpy
    boxes_np = boxes_norm.detach().cpu().numpy()

    for box in boxes_np:
        # box 是 8个浮点数 [x0, y0, x1, y1, x2, y2, x3, y3]
        pts = box.reshape(4, 2)

        # 将归一化坐标 [0,1] 还原回原始图片尺寸
        pts[:, 0] *= (w_img - 1)
        pts[:, 1] *= (h_img - 1)

        # 转为 int 供 cv2 使用
        pts_int = pts.astype(np.int32)

        # 绘制多边形 (闭合)
        cv2.polylines(frame, [pts_int], isClosed=True, color=color, thickness=thickness)

        # 画一个红点表示“头部”，方便看方向
        cv2.circle(frame, tuple(pts_int[0]), 4, (0, 0, 255), -1)

    return frame


def main():
    # ============================================================
    # 👇👇👇 在这里修改路径和参数 👇👇👇
    # ============================================================

    # 1. 模型路径 (.pt 文件)
    ckpt_path = "/home/wangzhe/ICME2026/ckpt_lag/b/best.pt"

    # 2. 视频源 (0 代表摄像头，或者填视频路径 "test.mp4")
    video_source = "/home/wangzhe/ICME2026/MyDataset/Video/b2.avi"

    # 3. 其他参数
    conf_thresh = 0.15  # 置信度阈值
    input_size = 320  # 必须和训练时的 image_size 一致
    top_k = 1  # 画面上最多显示几个抓取框
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # 👆👆👆 修改结束 👆👆👆
    # ============================================================

    print(f"[Info] 设备: {device_name}")
    device = torch.device(device_name)

    # --- 1. 加载模型 ---
    if not os.path.exists(ckpt_path):
        print(f"[Error] 找不到模型文件: {ckpt_path}")
        return

    print(f"[Info] 正在加载模型: {ckpt_path} ...")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)

        base_channels = 32
        if isinstance(checkpoint, dict) and "cfg" in checkpoint:
            base_channels = checkpoint["cfg"].get("base_channels", 32)

        model = GraspNetLAG(in_channels=3, base_channels=base_channels)

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

    except Exception as e:
        print(f"[Error] 模型加载出错: {e}")
        return

    model.to(device)
    model.eval()

    # --- 2. 打开视频 ---
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[Error] 无法打开视频源: {video_source}")
        return

    print(f"\n===以此键控制:===")
    print(f"  'v': 开始/暂停 检测")
    print(f"  'q': 退出程序")

    fps_avg = 0
    detecting = False  # <--- 新增状态标志：默认不检测

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Info] 视频播放结束")
                break

            # 只有当 detecting 为 True 时，才进行模型推理
            if detecting:
                t_start = time.time()

                # --- 预处理 ---
                input_tensor = preprocess_frame(frame, input_size, device)

                # --- 推理 ---
                pred = model(input_tensor)

                # --- 解码 ---
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

                # --- 绘图 ---
                frame = draw_grasps(frame, boxes_norm, color=(0, 255, 0), thickness=2)

                # --- FPS ---
                t_end = time.time()
                fps = 1.0 / (t_end - t_start + 1e-6)
                fps_avg = 0.9 * fps_avg + 0.1 * fps

                # 显示 FPS 和 状态
                status_text = f"RUNNING | FPS: {fps_avg:.1f}"
                color_text = (0, 255, 0)  # 绿色
            else:
                # 不检测时的状态提示
                status_text = "PAUSED (Press 'v' to Start)"
                color_text = (0, 0, 255)  # 红色

            # 在左上角绘制状态文字
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)

            cv2.imshow("Grasp Detection", frame)

            # --- 按键监听 ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('v'):
                detecting = not detecting  # 切换状态
                print(f"[Info] Detection status: {detecting}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()