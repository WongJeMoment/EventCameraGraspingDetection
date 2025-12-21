import os
import glob
import math
import cv2
import numpy as np

from geom import order_points_convex_hull, rect_params_from_4pts, grasp_params_from_2pts


# -------------------------
# Drawing helpers
# -------------------------
def rotated_rect_corners(cx, cy, theta, w, h):
    ct, st = math.cos(theta), math.sin(theta)
    ux, uy = ct, st
    vx, vy = -st, ct
    hw, hh = w * 0.5, h * 0.5
    p0 = (cx - ux * hh - vx * hw, cy - uy * hh - vy * hw)
    p1 = (cx + ux * hh - vx * hw, cy + uy * hh - vy * hw)
    p2 = (cx + ux * hh + vx * hw, cy + uy * hh + vy * hw)
    p3 = (cx - ux * hh + vx * hw, cy - uy * hh + vy * hw)
    return np.array([p0, p1, p2, p3], dtype=np.float32)

def draw_poly(img, pts, color=(0, 0, 255), thick=2, closed=True):
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_i], closed, color, thick, lineType=cv2.LINE_AA)

def draw_points(img, pts, color=(0, 255, 255), r=3):
    for (x, y) in pts:
        cv2.circle(img, (int(round(x)), int(round(y))), r, color, -1, lineType=cv2.LINE_AA)

def color_palette(i: int):
    palette = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
        (255, 255, 255)
    ]
    return palette[i % len(palette)]


# -------------------------
# TXT parsing (same logic as visualize_demo.py)
# -------------------------
def _parse_line_to_floats(line: str):
    line = line.replace(",", " ").strip()
    if not line:
        return None
    try:
        return [float(x) for x in line.split()]
    except ValueError:
        return None

def load_keypoint_instances(txt_path: str):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(txt_path)

    raw_lines = open(txt_path, "r", encoding="utf-8").read().splitlines()

    # blocks split by blank lines
    blocks, cur = [], []
    for ln in raw_lines:
        if ln.strip() == "":
            if cur:
                blocks.append(cur); cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)

    instances = []

    def add_block_points(points: np.ndarray):
        M = points.shape[0]
        if M == 4:
            instances.append({"type": "rect4", "pts": points, "meta": {}})
        elif M % 4 == 0 and M > 4:
            for k in range(0, M, 4):
                instances.append({"type": "rect4", "pts": points[k:k+4], "meta": {"grouped": True}})
        else:
            instances.append({"type": "poly", "pts": points, "meta": {}})

    for blk in blocks:
        block_points = []
        for ln in blk:
            vals = _parse_line_to_floats(ln)
            if not vals:
                continue

            # label x1 y1 x2 y2
            if len(vals) == 5:
                label = int(vals[0])
                x1, y1, x2, y2 = vals[1:]
                pts2 = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                instances.append({"type": "contacts2", "pts": pts2, "meta": {"label": label}})
                continue

            # x y
            if len(vals) == 2:
                block_points.append([vals[0], vals[1]])
                continue

            # 2N numbers (>=4, even)
            if len(vals) >= 4 and len(vals) % 2 == 0:
                pts = np.array(vals, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] == 4:
                    instances.append({"type": "rect4", "pts": pts, "meta": {"inline": True}})
                else:
                    for p in pts:
                        block_points.append([float(p[0]), float(p[1])])
                continue

        if block_points:
            add_block_points(np.array(block_points, dtype=np.float32))

    return instances


# -------------------------
# Render one image+txt -> output image
# -------------------------
def render_one(image_path: str, txt_path: str, theta_ref: str = "approach") -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    instances = load_keypoint_instances(txt_path)
    vis = img.copy()

    header = f"{os.path.basename(image_path)} | {os.path.basename(txt_path)}"
    cv2.putText(vis, header, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    y_text = 50
    for idx, inst in enumerate(instances):
        pts = inst["pts"]
        color = color_palette(idx)

        if inst["type"] == "rect4":
            draw_points(vis, pts)
            hull = order_points_convex_hull(pts, clockwise=True)
            draw_poly(vis, hull, color=color, thick=2)

            cx, cy, theta, w, h = rect_params_from_4pts(pts, theta_ref=theta_ref)
            rect_pts = rotated_rect_corners(cx, cy, theta, w, h)
            draw_poly(vis, rect_pts, color=(0, 0, 255), thick=2)
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            cv2.putText(vis, f"[{idx}] rect4 w={w:.1f} h={h:.1f} th={theta:.2f}",
                        (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            y_text += 20

        elif inst["type"] == "contacts2":
            draw_points(vis, pts, r=4)
            draw_poly(vis, pts, color=color, thick=2, closed=False)

            cx, cy, theta, w = grasp_params_from_2pts(pts[0], pts[1], theta_ref=theta_ref)
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            cv2.putText(vis, f"[{idx}] contacts2 w={w:.1f} th={theta:.2f}",
                        (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            y_text += 20

        else:
            # poly
            draw_points(vis, pts)
            if len(pts) >= 3:
                hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
                draw_poly(vis, hull, color=color, thick=2)

                rr = cv2.minAreaRect(hull.astype(np.float32))
                box = cv2.boxPoints(rr).astype(np.float32)
                draw_poly(vis, box, color=(0, 0, 255), thick=2)

            cv2.putText(vis, f"[{idx}] poly pts={len(pts)}",
                        (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            y_text += 20

    return vis


# -------------------------
# Batch runner
# -------------------------
def batch_visualize(
    image_dir: str,
    label_dir: str,
    out_dir: str,
    img_exts=(".png", ".jpg", ".jpeg", ".bmp"),
    label_ext=".txt",
    theta_ref: str = "approach",
    skip_missing_label: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    # collect images
    paths = []
    for ext in img_exts:
        paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
    paths = sorted(paths)

    if not paths:
        raise RuntimeError(f"No images found in {image_dir} with {img_exts}")

    ok, miss, fail = 0, 0, 0
    for img_path in paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(label_dir, stem + label_ext)

        if not os.path.exists(txt_path):
            miss += 1
            if not skip_missing_label:
                print(f"[MISS] {stem}: label not found: {txt_path}")
            continue

        try:
            vis = render_one(img_path, txt_path, theta_ref=theta_ref)
            out_path = os.path.join(out_dir, stem + "_vis.png")
            cv2.imwrite(out_path, vis)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {stem}: {e}")

    print(f"Done. ok={ok}, missing_label={miss}, failed={fail}, out_dir={out_dir}")


if __name__ == "__main__":
    # ✅ 你在这里填路径
    image_dir = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Images_dvs_bg"
    label_dir = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Annotations_angle_regularize"
    out_dir   = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Output"

    batch_visualize(
        image_dir=image_dir,
        label_dir=label_dir,
        out_dir=out_dir,
        img_exts=(".png",),      # 需要支持更多就加进去
        label_ext=".txt",
        theta_ref="approach",    # 或 "jaw"
        skip_missing_label=True,
    )
