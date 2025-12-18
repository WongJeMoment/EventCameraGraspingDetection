import os
import cv2
import numpy as np

from geom import rect_params_from_4pts, grasp_params_from_2pts
from heatmap import build_targets


def parse_txt_to_grasps(txt_path, theta_ref="approach"):
    """
    把你的 txt 标注解析成 grasps 列表：
      grasps = [{"cx","cy","theta","w"}, ...]
    支持：
      - 4点：每行 x y（凑够4个点算一个矩形），或一行8个数
      - 2点：一行5个数 label x1 y1 x2 y2
    """
    lines = open(txt_path, "r", encoding="utf-8").read().splitlines()

    def to_floats(line):
        line = line.replace(",", " ").strip()
        if not line:
            return None
        try:
            return [float(x) for x in line.split()]
        except:
            return None

    grasps = []
    buf_pts = []  # for lines of "x y"

    for ln in lines:
        vals = to_floats(ln)
        if vals is None:
            continue

        # contacts2: label x1 y1 x2 y2
        if len(vals) == 5:
            _, x1, y1, x2, y2 = vals
            cx, cy, theta, w = grasp_params_from_2pts((x1, y1), (x2, y2), theta_ref=theta_ref)
            grasps.append({"cx": cx, "cy": cy, "theta": theta, "w": w})
            continue

        # one point per line
        if len(vals) == 2:
            buf_pts.append([vals[0], vals[1]])
            if len(buf_pts) == 4:
                pts4 = np.array(buf_pts, dtype=np.float32)
                cx, cy, theta, w, _h = rect_params_from_4pts(pts4, theta_ref=theta_ref)
                grasps.append({"cx": cx, "cy": cy, "theta": theta, "w": w})
                buf_pts = []
            continue

        # inline points: 2N numbers
        if len(vals) >= 8 and len(vals) % 2 == 0:
            pts = np.array(vals, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] == 4:
                cx, cy, theta, w, _h = rect_params_from_4pts(pts, theta_ref=theta_ref)
                grasps.append({"cx": cx, "cy": cy, "theta": theta, "w": w})
            else:
                # 多于4点：按每4点切（如果你的数据就是这样组织的）
                for k in range(0, pts.shape[0] // 4 * 4, 4):
                    pts4 = pts[k:k+4]
                    cx, cy, theta, w, _h = rect_params_from_4pts(pts4, theta_ref=theta_ref)
                    grasps.append({"cx": cx, "cy": cy, "theta": theta, "w": w})
            continue

    return grasps


def save_qmap_visuals(img_path, txt_path, out_dir="vis_out", stride=4, sigma=2.0, theta_ref="approach"):
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    H, W = img.shape[:2]

    grasps = parse_txt_to_grasps(txt_path, theta_ref=theta_ref)
    if len(grasps) == 0:
        raise RuntimeError(f"No grasps parsed from: {txt_path}")

    targets = build_targets(
        grasps=grasps,
        img_h=H,
        img_w=W,
        stride=stride,
        sigma=sigma,
    )
    q = targets["qmap"]  # (H/stride, W/stride)
    mask = targets["mask"]

    # 1) Q-map 灰度图（放大到原图大小）
    q_norm = (q / (q.max() + 1e-8) * 255.0).astype(np.uint8)
    q_up = cv2.resize(q_norm, (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(out_dir, "qmap_gray.png"), q_up)

    # 2) Q-map 热力图叠加到原图
    q_color = cv2.applyColorMap(q_up, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.65, q_color, 0.35, 0)
    cv2.imwrite(os.path.join(out_dir, "qmap_overlay.png"), overlay)

    # 3) mask 可视化（回归监督点位）
    mask_up = cv2.resize((mask * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    mask_color = cv2.cvtColor(mask_up, cv2.COLOR_GRAY2BGR)
    mask_vis = cv2.addWeighted(img, 0.75, mask_color, 0.25, 0)
    cv2.imwrite(os.path.join(out_dir, "reg_mask_overlay.png"), mask_vis)

    print(f"Saved to: {out_dir}")
    print(" - qmap_gray.png")
    print(" - qmap_overlay.png")
    print(" - reg_mask_overlay.png")


if __name__ == "__main__":
    img_path = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Images_dvs_bg/000001.png"
    txt_path = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Annotations_angle_regularize/000001.txt"
    out_dir  = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/heatmap/000001.png"

    save_qmap_visuals(
        img_path=img_path,
        txt_path=txt_path,
        out_dir=out_dir,
        stride=4,
        sigma=2.0,
        theta_ref="approach",
    )
