import os
import math
import cv2
import numpy as np

# 需要你已有的 geom.py（我之前给你的那个）
from geom import order_points_convex_hull, rect_params_from_4pts, grasp_params_from_2pts


# -------------------------
# Drawing helpers
# -------------------------
def rotated_rect_corners(cx, cy, theta, w, h):
    """
    Build 4 corners of rotated rectangle from center, angle, w, h.
    theta: radians
    w: short side (jaw opening) length in pixels
    h: long side (contact) length in pixels
    """
    ct, st = math.cos(theta), math.sin(theta)
    ux, uy = ct, st          # along theta
    vx, vy = -st, ct         # perpendicular

    hw = w * 0.5
    hh = h * 0.5

    # Use u as long axis (h), v as short axis (w)
    p0 = (cx - ux * hh - vx * hw, cy - uy * hh - vy * hw)
    p1 = (cx + ux * hh - vx * hw, cy + uy * hh - vy * hw)
    p2 = (cx + ux * hh + vx * hw, cy + uy * hh + vy * hw)
    p3 = (cx - ux * hh + vx * hw, cy - uy * hh + vy * hw)
    return np.array([p0, p1, p2, p3], dtype=np.float32)


def draw_poly(img, pts, color=(0, 0, 255), thick=2, closed=True):
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_i], closed, color, thick, lineType=cv2.LINE_AA)


def draw_points(img, pts, color=(0, 255, 0), r=3, show_index=False, prefix=""):
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (int(round(x)), int(round(y))), r, color, -1, lineType=cv2.LINE_AA)
        if show_index:
            cv2.putText(
                img,
                f"{prefix}{i}",
                (int(round(x)) + 4, int(round(y)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )


def color_palette(i: int):
    # BGR colors
    palette = [
        (0, 0, 255),    # red
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan
        (255, 255, 255) # white
    ]
    return palette[i % len(palette)]


# -------------------------
# TXT parsing
# -------------------------
def _parse_line_to_floats(line: str):
    """
    Returns list[float] or None if line can't be parsed.
    Accepts spaces or commas as separators.
    """
    line = line.replace(",", " ").strip()
    if not line:
        return None
    parts = line.split()
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def load_keypoint_instances(txt_path: str):
    """
    Parse a txt annotation file into a list of instances.

    Supported line formats:
      1) x y                         (one point per line)
      2) x1 y1 x2 y2 x3 y3 x4 y4     (one rect per line, 8 numbers)
      3) label x1 y1 x2 y2           (two contact points per line, 5 numbers)

    Also supports:
      - blocks separated by blank lines
      - a line with 2N numbers (>= 4) -> treated as N points; later grouped if possible

    Returns:
      instances: list of dict:
        {
          "type": "rect4" | "contacts2" | "poly",
          "pts":  np.ndarray (N,2),
          "meta": dict
        }
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(txt_path)

    raw_lines = open(txt_path, "r", encoding="utf-8").read().splitlines()

    # Split into blocks by blank lines (optional grouping)
    blocks = []
    cur = []
    for ln in raw_lines:
        if ln.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)

    instances = []

    def add_block_points(points: np.ndarray):
        """
        points: (M,2)
        Heuristic:
          - if M == 4 -> rect4
          - elif M % 4 == 0 and M > 4 -> multiple rect4 (every 4 points)
          - else -> poly
        """
        M = points.shape[0]
        if M == 4:
            instances.append({"type": "rect4", "pts": points, "meta": {"from": "block_xy"}})
        elif M % 4 == 0 and M > 4:
            for k in range(0, M, 4):
                instances.append({"type": "rect4", "pts": points[k:k+4], "meta": {"from": "block_xy_grouped"}})
        else:
            instances.append({"type": "poly", "pts": points, "meta": {"from": "block_xy_poly"}})

    # Parse each block
    for blk in blocks:
        block_points = []

        for ln in blk:
            vals = _parse_line_to_floats(ln)
            if vals is None or len(vals) == 0:
                continue

            # Case: label x1 y1 x2 y2 (two contacts)
            if len(vals) == 5:
                label = int(vals[0])
                x1, y1, x2, y2 = vals[1:]
                pts2 = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                instances.append({"type": "contacts2", "pts": pts2, "meta": {"label": label}})
                continue

            # Case: one point per line
            if len(vals) == 2:
                block_points.append([vals[0], vals[1]])
                continue

            # Case: 2N numbers on one line (inline points)
            if len(vals) >= 4 and len(vals) % 2 == 0:
                pts = np.array(vals, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] == 4:
                    instances.append({"type": "rect4", "pts": pts, "meta": {"from": "inline_8nums"}})
                else:
                    # put into block_points; will be grouped later if possible
                    for p in pts:
                        block_points.append([float(p[0]), float(p[1])])
                continue

            # Unknown format -> ignore

        if block_points:
            add_block_points(np.array(block_points, dtype=np.float32))

    return instances


# -------------------------
# Visualization
# -------------------------
def visualize(image_path: str, txt_path: str, theta_ref: str = "approach"):
    """
    theta_ref:
      - "approach": theta is grasp approach direction (jaw normal)  (推荐)
      - "jaw":      theta is jaw-line direction
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    if not os.path.exists(txt_path):
        raise FileNotFoundError(txt_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    instances = load_keypoint_instances(txt_path)
    vis = img.copy()

    header = f"IMG: {os.path.basename(image_path)} | TXT: {os.path.basename(txt_path)} | theta_ref={theta_ref}"
    cv2.putText(vis, header, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    y_text = 50

    for idx, inst in enumerate(instances):
        pts = inst["pts"]
        color = color_palette(idx)

        if inst["type"] == "rect4":
            # raw points
            draw_points(vis, pts, color=(0, 255, 255), r=3, show_index=True, prefix=f"{idx}-")

            # ordered hull (blue-ish)
            hull = order_points_convex_hull(pts, clockwise=True)
            draw_poly(vis, hull, color=color, thick=2)

            # compute params and draw reconstructed rect (red)
            cx, cy, theta, w, h = rect_params_from_4pts(pts, theta_ref=theta_ref)
            rect_pts = rotated_rect_corners(cx, cy, theta, w, h)
            draw_poly(vis, rect_pts, color=(0, 0, 255), thick=2)
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            cv2.putText(vis, f"[{idx}] rect4  w={w:.1f} h={h:.1f} th={theta:.2f}",
                        (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            y_text += 20

        elif inst["type"] == "contacts2":
            # two contacts: draw points and segment
            draw_points(vis, pts, color=(0, 255, 255), r=4, show_index=True, prefix=f"{idx}-")
            draw_poly(vis, pts, color=color, thick=2, closed=False)

            cx, cy, theta, w = grasp_params_from_2pts(pts[0], pts[1], theta_ref=theta_ref)
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

            cv2.putText(vis, f"[{idx}] contacts2  w={w:.1f} th={theta:.2f} label={inst['meta'].get('label','-')}",
                        (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            y_text += 20

        else:
            # poly / many keypoints: draw points and convex hull + minAreaRect reference
            draw_points(vis, pts, color=(0, 255, 255), r=3, show_index=False)

            if len(pts) >= 3:
                hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
                draw_poly(vis, hull, color=color, thick=2)

                rr = cv2.minAreaRect(hull.astype(np.float32))
                box = cv2.boxPoints(rr).astype(np.float32)
                draw_poly(vis, box, color=(0, 0, 255), thick=2)

                (cx, cy), (rw, rh), ang = rr  # OpenCV angle in degrees (convention varies)
                cv2.circle(vis, (int(round(cx)), int(round(cy))), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)

                cv2.putText(vis, f"[{idx}] poly pts={len(pts)} (minAreaRect ref)",
                            (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                y_text += 20
            else:
                cv2.putText(vis, f"[{idx}] poly pts={len(pts)}",
                            (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                y_text += 20

    cv2.imshow("grasp annotation visualization", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------
# Main: explicitly set paths here
# -------------------------
if __name__ == "__main__":
    # ✅ 在这里显式填你的路径（可以是相对路径或绝对路径）
    img_path = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Images_dvs_bg/000001.png"
    txt_path = r"/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Annotations_angle_regularize/000001.txt"

    # theta_ref:
    #  - "approach": 夹取方向（jaw 的法向，常用）
    #  - "jaw":      两指连线方向
    visualize(
        image_path=img_path,
        txt_path=txt_path,
        theta_ref="approach"
    )
