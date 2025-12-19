# draw_grasps.py
from __future__ import annotations
import math
from typing import List, Dict

import cv2
import numpy as np


def normalize_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def rotated_rect_corners(cx: float, cy: float, theta: float, w: float, h: float) -> np.ndarray:
    """
    Build 4 corners of rotated rectangle from center.
    - theta: radians
    - w: short side (jaw opening)
    - h: long side (contact length)
    Return: (4,2) float32
    """
    ct, st = math.cos(theta), math.sin(theta)
    ux, uy = ct, st
    vx, vy = -st, ct

    hw = 0.5 * w
    hh = 0.5 * h

    # u axis -> long side (h), v axis -> short side (w)
    p0 = (cx - ux * hh - vx * hw, cy - uy * hh - vy * hw)
    p1 = (cx + ux * hh - vx * hw, cy + uy * hh - vy * hw)
    p2 = (cx + ux * hh + vx * hw, cy + uy * hh + vy * hw)
    p3 = (cx - ux * hh + vx * hw, cy - uy * hh + vy * hw)

    return np.array([p0, p1, p2, p3], dtype=np.float32)


def draw_poly(img, pts: np.ndarray, color=(0, 0, 255), thick: int = 2, closed: bool = True):
    pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_i], closed, color, thick, lineType=cv2.LINE_AA)


def draw_arrow(img, x: float, y: float, theta: float, length: float = 40.0,
               color=(0, 255, 0), thick: int = 2):
    """
    OpenCV 4.5.5 compatibility:
    cv2.arrowedLine() 不支持 lineType= 关键字，这里用位置参数。
    """
    x2 = x + length * math.cos(theta)
    y2 = y + length * math.sin(theta)

    pt1 = (int(round(x)), int(round(y)))
    pt2 = (int(round(x2)), int(round(y2)))

    # arrowedLine(img, pt1, pt2, color, thickness, line_type, shift, tipLength)
    cv2.arrowedLine(img, pt1, pt2, color, thick, cv2.LINE_AA, 0, 0.25)


def compute_h(w: float, h_mode: str = "ratio", h_fixed: float = 40.0, h_ratio: float = 1.5) -> float:
    if h_mode == "fixed":
        return float(h_fixed)
    if h_mode == "ratio":
        return float(h_ratio * w)
    raise ValueError("h_mode must be 'fixed' or 'ratio'")


def draw_grasps(
    img_bgr: np.ndarray,
    grasps: List[Dict],
    topk: int = 20,
    conf_thresh: float = 0.0,
    h_mode: str = "ratio",
    h_ratio: float = 1.5,
    h_fixed: float = 40.0,
    draw_center: bool = True,
    draw_theta_arrow: bool = True,
    draw_text: bool = True,
) -> np.ndarray:
    """
    Draw decoded grasps on image.

    grasps: list of dict with at least:
      {"x":..., "y":..., "w":..., "theta":..., "conf":...}
    """
    vis = img_bgr.copy()

    # filter & sort
    gs = [g for g in grasps if float(g.get("conf", 0.0)) >= conf_thresh]
    gs = sorted(gs, key=lambda d: float(d.get("conf", 0.0)), reverse=True)[:topk]

    for i, g in enumerate(gs):
        x = float(g["x"])
        y = float(g["y"])
        w = float(g["w"])
        theta = normalize_angle(float(g["theta"]))
        conf = float(g.get("conf", 0.0))

        h = compute_h(w, h_mode=h_mode, h_fixed=h_fixed, h_ratio=h_ratio)
        rect_pts = rotated_rect_corners(x, y, theta, w, h)

        # rectangle
        draw_poly(vis, rect_pts, color=(0, 0, 255), thick=2)

        # center point
        if draw_center:
            cv2.circle(vis, (int(round(x)), int(round(y))), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        # theta arrow
        if draw_theta_arrow:
            arrow_len = max(20.0, min(80.0, 0.8 * w))
            draw_arrow(vis, x, y, theta, length=arrow_len, color=(0, 255, 0), thick=2)

        # text
        if draw_text:
            txt = f"#{i} conf={conf:.2f} w={w:.1f} th={theta:.2f}"
            cv2.putText(
                vis,
                txt,
                (int(round(x)) + 6, int(round(y)) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return vis


if __name__ == "__main__":
    # quick demo (optional)
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    grasps = [
        {"x": 160, "y": 120, "w": 60, "theta": 0.8, "conf": 0.95},
        {"x": 200, "y": 150, "w": 45, "theta": -0.3, "conf": 0.75},
    ]
    vis = draw_grasps(img, grasps, topk=10, conf_thresh=0.0, h_mode="ratio", h_ratio=1.5)
    cv2.imshow("draw_grasps demo", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
