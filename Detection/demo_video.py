import cv2
import torch
import numpy as np
import time
import os
from typing import Tuple, Optional

# ============================================================
# 导入你的项目模块
# ============================================================
from LAGNetwork.model import GraspNetLAG
from LAGNetwork.config import TrainConfig
from LAGNetwork.train import decode_maps_to_boxes_norm


# ==============================================================================
# 1. 智能跟踪管理器 (Hybrid Tracker)
#    策略: 神经网络检测 -> KCF 跟踪 -> 定期/失败重检测
# ==============================================================================
class GraspTrackerManager:
    def __init__(self, model, device, input_size=320, conf_thresh=0.15):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.conf_thresh = conf_thresh

        # OpenCV 跟踪器 (KCF 速度快效果好，CSRT 精度高但慢)
        # 如果报错 AttributeError，请确保安装了 opencv-contrib-python
        self.tracker = None
        self.is_tracking = False

        # 状态参数
        self.last_detect_time = 0
        self.detect_interval = 1.0  # 每隔 1 秒强制重检测一次，消除累计误差
        self.fail_counter = 0

    def _create_tracker(self):
        # 创建一个新的 KCF 跟踪器
        return cv2.TrackerKCF_create()
        # 如果没有安装 contrib，可以使用: cv2.TrackerMIL_create()

    def get_grasp(self, frame_orig: np.ndarray) -> Tuple[Optional[torch.Tensor], str]:
        """
        主函数：输入当前帧，输出抓取框和当前状态
        返回: (Box_Norm, Status_String)
        """
        curr_time = time.time()
        H, W = frame_orig.shape[:2]

        # --- 策略 A: 正在跟踪中 ---
        if self.is_tracking:
            # 1. 如果超过了重置时间，强制重新检测
            if curr_time - self.last_detect_time > self.detect_interval:
                self.is_tracking = False
                return self._run_detection(frame_orig)

            # 2. 更新跟踪器
            success, bbox = self.tracker.update(frame_orig)

            if success:
                # 跟踪成功，将 xywh 转换为模型输出的格式 (归一化 Quad)
                x, y, w, h = bbox

                # 简单的转换：矩形 -> 4点多边形
                # 注意：KCF 不支持旋转，所以这里只能维持上一帧的角度，或者假设是水平的
                # 为了简单，我们构建一个轴对齐矩形
                box_norm = self._bbox_to_norm_tensor(x, y, w, h, W, H)
                return box_norm, "Tracking (KCF)"
            else:
                # 跟踪失败，立即切回检测模式
                self.is_tracking = False
                self.fail_counter += 1
                return self._run_detection(frame_orig)

        # --- 策略 B: 未跟踪 (或刚失败)，运行神经网络检测 ---
        else:
            return self._run_detection(frame_orig)

    def _run_detection(self, frame: np.ndarray) -> Tuple[Optional[torch.Tensor], str]:
        """运行深度学习模型进行全图检测"""
        # 1. 预处理
        tensor = preprocess_frame(frame, self.input_size, self.device)

        # 2. 推理
        with torch.no_grad():
            pred = self.model(tensor)

        pred_one = {
            "quality": pred["quality"][0],
            "angle": pred["angle"][0],
            "width": pred["width"][0],
        }

        # 3. 解码
        # width_scale=1.0 窄框
        boxes_norm = decode_maps_to_boxes_norm(
            pred_one, topk=1, q_thresh=self.conf_thresh,
            grasp_h_ratio=0.5, width_scale=1.0
        )

        # 4. 如果检测到了
        if boxes_norm is not None and boxes_norm.numel() > 0:
            best_box = boxes_norm[0]  # (8,)

            # 初始化跟踪器
            self.tracker = self._create_tracker()

            # 将归一化 4点坐标 转为 像素 xywh 供 KCF 使用
            H, W = frame.shape[:2]
            x, y, w, h = self._norm_tensor_to_bbox(best_box, W, H)

            # KCF 初始化
            self.tracker.init(frame, (x, y, w, h))

            self.is_tracking = True
            self.last_detect_time = time.time()

            return best_box.unsqueeze(0), "Detection (NN)"

        else:
            self.is_tracking = False
            return None, "Searching..."

    def _norm_tensor_to_bbox(self, box_norm, W, H):
        """辅助：将 (8,) 归一化坐标转为 (x, y, w, h)"""
        pts = box_norm.reshape(4, 2).detach().cpu().numpy()
        pts[:, 0] *= (W - 1)
        pts[:, 1] *= (H - 1)
        rect = cv2.boundingRect(pts.astype(np.int32))  # x, y, w, h
        return rect

    def _bbox_to_norm_tensor(self, x, y, w, h, W, H):
        """辅助：将 (x, y, w, h) 转为 (1, 8) 归一化 Tensor"""
        # 构造矩形4个点
        pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)

        # 归一化
        pts[:, 0] /= (W - 1)
        pts[:, 1] /= (H - 1)

        tensor = torch.from_numpy(pts.flatten()).unsqueeze(0).to(self.device)
        return tensor


# ============================================================
# 辅助函数 (保持不变)
# ============================================================
def preprocess_frame(frame: np.ndarray, input_size: int, device: torch.device) -> torch.Tensor:
    frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    frame_resized = cv2.resize(frame_blurred, (input_size, input_size))
    img = frame_resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor


def draw_grasps(frame: np.ndarray, boxes_norm: torch.Tensor, status: str):
    """根据状态绘制不同颜色的框"""
    h_img, w_img = frame.shape[:2]
    if boxes_norm is None or boxes_norm.numel() == 0: return frame

    # 颜色策略
    if "Tracking" in status:
        color = (255, 191, 0)  # 蓝色/青色 (跟踪中)
        thickness = 2
    elif "Detection" in status:
        color = (0, 0, 255)  # 红色 (刚检测到/校准)
        thickness = 3
    else:
        color = (0, 255, 0)

    boxes_np = boxes_norm.detach().cpu().numpy()
    for box in boxes_np:
        pts = box.reshape(4, 2)
        pts[:, 0] *= (w_img - 1)
        pts[:, 1] *= (h_img - 1)

        # 使用旋转矩形绘制
        rect = cv2.minAreaRect(pts.astype(np.float32))
        box_pts = cv2.boxPoints(rect)
        box_pts = np.int0(box_pts)

        cv2.drawContours(frame, [box_pts], 0, color, thickness)

        # 只有检测帧才画方向点，跟踪帧因为KCF不含旋转信息，画方向点可能不准
        if "Detection" in status:
            for p in box_pts:
                cv2.circle(frame, tuple(p), 2, color, -1)

    return frame


def main():
    # ============================================================
    # 配置
    # ============================================================
    ckpt_path = "/home/wangzhe/ICME2026/ckpt_lag/apple/last.pt"
    video_source = "/home/wangzhe/ICME2026/MyDataset/Video/apple1.avi"
    save_path = "result_tracking.mp4"

    conf_thresh = 0.15
    input_size = 320

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Device: {device_name}")
    device = torch.device(device_name)

    # 1. 加载模型
    if not os.path.exists(ckpt_path):
        print(f"[Error] Model not found: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    base_channels = checkpoint.get("cfg", {}).get("base_channels", 32)
    model = GraspNetLAG(in_channels=3, base_channels=base_channels)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 2. 初始化 跟踪管理器 (代替原来的 Stabilizer)
    # 这里的 detect_interval=2.0 意味着每2秒重新用神经网络校准一次
    tracker_manager = GraspTrackerManager(model, device, input_size, conf_thresh)
    tracker_manager.detect_interval = 2.0

    # 3. 视频输入输出
    cap = cv2.VideoCapture(video_source)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0: original_fps = 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), original_fps, (W, H))

    print(f"[Info] Mode: Hybrid (Detection + KCF Tracking)")
    print(f"       Red Box  = Neural Network Detection")
    print(f"       Blue Box = KCF Tracker")

    detecting = False
    delay_ms = int(1000 / original_fps)

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret: break

        status_text = "PAUSED"
        color_text = (0, 0, 255)

        if detecting:
            # === 核心调用 ===
            # 直接把原图扔给管理器，它自己决定是 跑网络 还是 跑跟踪
            final_box, status_mode = tracker_manager.get_grasp(frame)

            if final_box is not None:
                frame = draw_grasps(frame, final_box, status_mode)

            status_text = status_mode
            color_text = (0, 255, 0)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)
        cv2.imshow("Hybrid Tracking", frame)
        out.write(frame)

        # 帧率同步
        dt = (time.time() - t_start) * 1000
        wait = max(1, int(delay_ms - dt))
        key = cv2.waitKey(wait) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('v'):
            detecting = not detecting
            # 每次重新开始时，强制切回检测模式
            tracker_manager.is_tracking = False
            print(f"[Info] Active: {detecting}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()