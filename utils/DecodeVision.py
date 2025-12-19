import os
import cv2
import numpy as np

from decode import decode_grasps, local_max_nms, topk_from_heatmap
from draw_grasps import draw_grasps


def qmap_to_u8(qmap):
    q = qmap.astype(np.float32)
    q = q / (q.max() + 1e-8)
    return (q * 255.0).astype(np.uint8)


def overlay_qmap(img_bgr, qmap, alpha=0.35):
    """Return overlay image and resized heatmap uint8."""
    H, W = img_bgr.shape[:2]
    q_u8 = qmap_to_u8(qmap)
    q_up = cv2.resize(q_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    q_color = cv2.applyColorMap(q_up, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, q_color, alpha, 0)
    return overlay, q_up


def draw_peaks(img_bgr, peaks, stride, color=(255, 255, 255)):
    """
    peaks: list of (ix, iy, score) in heatmap coords
    """
    vis = img_bgr.copy()
    for (ix, iy, s) in peaks:
        x = int(round(ix * stride))
        y = int(round(iy * stride))
        cv2.circle(vis, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            vis,
            "{:.2f}".format(s),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def decode_npz_with_viz(
    npz_path,
    img_path=None,              # ✅ Python<3.10 兼容：不用 str | None
    out_dir="decode_npz_out",
    topk=20,
    conf_thresh=0.2,
    nms_ksize=3,
    h_mode="ratio",
    h_ratio=1.5,
    h_fixed=40.0,
    save_cands_txt=True,
):
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(npz_path)
    if ("qmap" not in data) or ("reg" not in data):
        raise ValueError("NPZ must contain keys: qmap, reg (and ideally stride,img_h,img_w)")

    qmap = data["qmap"].astype(np.float32)   # (Hm, Wm)
    reg = data["reg"].astype(np.float32)     # (5, Hm, Wm)

    stride = int(data["stride"]) if "stride" in data else 4
    img_h = int(data["img_h"]) if "img_h" in data else qmap.shape[0] * stride
    img_w = int(data["img_w"]) if "img_w" in data else qmap.shape[1] * stride
    img_hw = (img_h, img_w)

    # background image for visualization
    if (img_path is not None) and os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError("Failed to read: {}".format(img_path))
        if img.shape[0] != img_h or img.shape[1] != img_w:
            img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    else:
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    stem = os.path.splitext(os.path.basename(npz_path))[0]

    # -------------------------
    # 1) Q-map gray + overlay
    # -------------------------
    overlay, q_up = overlay_qmap(img, qmap, alpha=0.35)
    cv2.imwrite(os.path.join(out_dir, "{}_1_qmap_gray.png".format(stem)), q_up)
    cv2.imwrite(os.path.join(out_dir, "{}_2_qmap_overlay.png".format(stem)), overlay)

    # -------------------------
    # 2) NMS peaks visualization (detector stage)
    # -------------------------
    q_thr = np.where(qmap >= conf_thresh, qmap, 0.0).astype(np.float32)
    q_nms = local_max_nms(q_thr, ksize=nms_ksize)
    scores, ys, xs = topk_from_heatmap(q_nms, k=topk)

    peaks = []
    for s, y, x in zip(scores, ys, xs):
        if float(s) > 0.0:
            peaks.append((int(x), int(y), float(s)))

    vis_peaks = draw_peaks(overlay, peaks, stride=stride, color=(255, 255, 255))
    cv2.imwrite(os.path.join(out_dir, "{}_3_qnms_peaks.png".format(stem)), vis_peaks)

    # -------------------------
    # 3) Decode -> grasps visualization
    # -------------------------
    cands = decode_grasps(
        Q=qmap,
        reg=reg,
        stride=stride,
        topk=topk,
        conf_thresh=conf_thresh,
        nms_ksize=nms_ksize,
        img_hw=img_hw,
    )

    vis_grasps = draw_grasps(
        vis_peaks,
        cands,
        topk=topk,
        conf_thresh=conf_thresh,
        h_mode=h_mode,
        h_ratio=h_ratio,
        h_fixed=h_fixed,
        draw_center=True,
        draw_theta_arrow=True,
        draw_text=True,
    )
    out_img = os.path.join(out_dir, "{}_4_decoded_grasps.png".format(stem))
    cv2.imwrite(out_img, vis_grasps)

    # -------------------------
    # 4) Save decoded candidates to txt
    # -------------------------
    if save_cands_txt:
        out_txt = os.path.join(out_dir, "{}_cands.txt".format(stem))
        with open(out_txt, "w", encoding="utf-8") as f:
            for g in cands:
                f.write("{:.3f} {:.3f} {:.3f} {:.6f} {:.4f}\n".format(
                    g["x"], g["y"], g["w"], g["theta"], g["conf"]
                ))

    print("[OK] npz: {}".format(npz_path))
    print("[OK] saved dir: {}".format(out_dir))
    print("[OK] decoded grasps: {} (conf_thresh={})".format(len(cands), conf_thresh))
    print(" - {}_1_qmap_gray.png".format(stem))
    print(" - {}_2_qmap_overlay.png".format(stem))
    print(" - {}_3_qnms_peaks.png".format(stem))
    print(" - {}_4_decoded_grasps.png".format(stem))
    if save_cands_txt:
        print(" - {}_cands.txt".format(stem))

    return cands


if __name__ == "__main__":
    npz_path = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/heatmap_vis/000001_targets.npz"
    img_path = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Images_dvs_bg/000001.png"
    out_dir = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/decode_from_npz_out"

    decode_npz_with_viz(
        npz_path=npz_path,
        img_path=img_path,
        out_dir=out_dir,
        topk=20,
        conf_thresh=0.2,
        nms_ksize=3,
        h_mode="ratio",
        h_ratio=1.5,
        save_cands_txt=True,
    )
