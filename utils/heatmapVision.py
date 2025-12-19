import os
import cv2
import numpy as np

from geom import rect_params_from_4pts, grasp_params_from_2pts
from heatmap import build_targets


def parse_txt_to_grasps(txt_path, theta_ref="approach"):
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
    buf_pts = []

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
                for k in range(0, (pts.shape[0] // 4) * 4, 4):
                    pts4 = pts[k:k+4]
                    cx, cy, theta, w, _h = rect_params_from_4pts(pts4, theta_ref=theta_ref)
                    grasps.append({"cx": cx, "cy": cy, "theta": theta, "w": w})
            continue

    return grasps


def save_targets_npz(save_path_npz, targets, stride, img_hw,
                     logw_mode="pixel"):
    """
    保存给 decode 用的“数据”(npz)，不是图像。

    logw_mode:
      - "pixel"   : logw = log(w_pixel)          (推荐：直观，decode无需再乘stride)
      - "cell"    : logw = log(w_pixel / stride) (有些人喜欢这样做，更稳定，但decode要乘stride)

    npz 会额外写入：
      - logw_mode (0:pixel, 1:cell)
      - wmap_px   (原始像素宽度图，便于你排查)
    """
    qmap = targets["qmap"].astype(np.float32)
    dx   = targets["dx"].astype(np.float32)
    dy   = targets["dy"].astype(np.float32)
    wmap = targets["w"].astype(np.float32)      # w in pixel (来自你的标注)
    sinm = targets["sin"].astype(np.float32)
    cosm = targets["cos"].astype(np.float32)
    mask = targets["mask"].astype(np.float32)

    # ---- logw 的定义（关键）----
    eps = 1e-6
    if logw_mode == "pixel":
        logw = np.log(np.maximum(wmap, eps)).astype(np.float32)
        logw_mode_id = np.int32(0)
    elif logw_mode == "cell":
        logw = np.log(np.maximum(wmap / float(stride), eps)).astype(np.float32)
        logw_mode_id = np.int32(1)
    else:
        raise ValueError("logw_mode must be 'pixel' or 'cell'")

    reg = np.stack([dx, dy, logw, sinm, cosm], axis=0).astype(np.float32)

    H, W = img_hw
    np.savez_compressed(
        save_path_npz,
        qmap=qmap,
        reg=reg,
        mask=mask,
        stride=np.int32(stride),
        img_h=np.int32(H),
        img_w=np.int32(W),

        # 额外保存用于排查/自动decode
        wmap_px=wmap.astype(np.float32),
        logw_mode=logw_mode_id,
    )


def save_qmap_visuals_and_data(
    img_path,
    txt_path,
    out_dir="vis_out",
    stride=4,
    sigma=2.0,
    theta_ref="approach",
    logw_mode="pixel",   # ✅ 新增：控制保存哪种 logw
):
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    H, W = img.shape[:2]

    grasps = parse_txt_to_grasps(txt_path, theta_ref=theta_ref)
    if len(grasps) == 0:
        raise RuntimeError("No grasps parsed from: {}".format(txt_path))

    targets = build_targets(
        grasps=grasps,
        img_h=H,
        img_w=W,
        stride=stride,
        sigma=sigma,
    )

    # ---------- 1) 保存“数据”给 decode ----------
    stem = os.path.splitext(os.path.basename(img_path))[0]
    npz_path = os.path.join(out_dir, "{}_targets.npz".format(stem))
    save_targets_npz(npz_path, targets, stride=stride, img_hw=(H, W), logw_mode=logw_mode)

    # ---------- 2) 保存可视化 PNG（调试） ----------
    q = targets["qmap"]
    mask = targets["mask"]

    q_norm = (q / (q.max() + 1e-8) * 255.0).astype(np.uint8)
    q_up = cv2.resize(q_norm, (W, H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(out_dir, "{}_qmap_gray.png".format(stem)), q_up)

    q_color = cv2.applyColorMap(q_up, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.65, q_color, 0.35, 0)
    cv2.imwrite(os.path.join(out_dir, "{}_qmap_overlay.png".format(stem)), overlay)

    mask_up = cv2.resize((mask * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    mask_color = cv2.cvtColor(mask_up, cv2.COLOR_GRAY2BGR)
    mask_vis = cv2.addWeighted(img, 0.75, mask_color, 0.25, 0)
    cv2.imwrite(os.path.join(out_dir, "{}_reg_mask_overlay.png".format(stem)), mask_vis)

    print("[OK] Saved to: {}".format(out_dir))
    print(" - {}   (for decode)".format(os.path.basename(npz_path)))
    print(" - {}_qmap_gray.png".format(stem))
    print(" - {}_qmap_overlay.png".format(stem))
    print(" - {}_reg_mask_overlay.png".format(stem))
    print("[INFO] logw_mode = {} (0=pixel, 1=cell)".format("pixel" if logw_mode == "pixel" else "cell"))


if __name__ == "__main__":
    img_path = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Images_dvs_bg/000001.png"
    txt_path = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Annotations_angle_regularize/000001.txt"
    out_dir  = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/heatmap_vis"

    save_qmap_visuals_and_data(
        img_path=img_path,
        txt_path=txt_path,
        out_dir=out_dir,
        stride=4,
        sigma=2.0,
        theta_ref="approach",

        # ✅ 推荐先用 pixel：logw = log(w_pixel)
        logw_mode="pixel",
        # 如果你想存 log(w/stride)，改成：logw_mode="cell"
    )
