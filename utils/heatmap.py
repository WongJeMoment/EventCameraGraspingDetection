# heatmap.py
"""
Utilities to generate grasp quality heatmaps (Q-map).

This module is backbone-agnostic and ONLY handles geometry-to-heatmap logic.
"""

from __future__ import annotations
import math
import numpy as np


# -----------------------------
# Basic gaussian
# -----------------------------
def gaussian_2d(shape, sigma):
    """
    Generate a 2D Gaussian kernel.

    Args:
        shape: (h, w)
        sigma: standard deviation

    Returns:
        kernel: (h, w)
    """
    h, w = shape
    y = np.arange(0, h, dtype=np.float32) - (h - 1) / 2
    x = np.arange(0, w, dtype=np.float32) - (w - 1) / 2
    yy, xx = np.meshgrid(y, x, indexing="ij")
    g = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    return g


def draw_gaussian(heatmap, center, sigma):
    """
    Draw a 2D Gaussian on heatmap (in-place, max blending).

    Args:
        heatmap: (H, W) float32
        center:  (cx, cy) in heatmap coordinates
        sigma:   gaussian sigma (pixels in heatmap scale)
    """
    H, W = heatmap.shape
    cx, cy = int(round(center[0])), int(round(center[1]))

    radius = int(3 * sigma)
    if radius <= 0:
        return

    x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)

    if x0 >= x1 or y0 >= y1:
        return

    g = gaussian_2d((y1 - y0, x1 - x0), sigma)
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], g)


# -----------------------------
# Q-map generation
# -----------------------------
def generate_qmap(
    grasps,
    out_h,
    out_w,
    stride,
    sigma=2.0,
):
    """
    Generate Q-map (grasp quality heatmap).

    Args:
        grasps: list of dicts, each with:
            {
              "cx": float (pixel in input image),
              "cy": float,
            }
        out_h, out_w: output heatmap size (H/stride, W/stride)
        stride: feature stride (e.g. 4 or 8)
        sigma: gaussian sigma in heatmap pixels

    Returns:
        qmap: (out_h, out_w) float32 in [0,1]
    """
    qmap = np.zeros((out_h, out_w), dtype=np.float32)

    for g in grasps:
        cx_img, cy_img = g["cx"], g["cy"]
        cx = cx_img / stride
        cy = cy_img / stride
        draw_gaussian(qmap, (cx, cy), sigma)

    return qmap


# -----------------------------
# Regression target helpers
# -----------------------------
def generate_regression_maps(
    grasps,
    out_h,
    out_w,
    stride,
):
    """
    Generate regression target maps for anchor-free grasp detection.

    Outputs:
        dx_map, dy_map : offset within cell
        w_map          : grasp width (pixels)
        sin_map, cos_map : angle representation
        mask           : valid mask (1 where grasp exists)

    Args:
        grasps: list of dicts, each with:
            {
              "cx": float (image pixel),
              "cy": float,
              "theta": float (radians),
              "w": float (pixels)
            }
    """
    dx_map = np.zeros((out_h, out_w), dtype=np.float32)
    dy_map = np.zeros((out_h, out_w), dtype=np.float32)
    w_map  = np.zeros((out_h, out_w), dtype=np.float32)
    sin_map = np.zeros((out_h, out_w), dtype=np.float32)
    cos_map = np.zeros((out_h, out_w), dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=np.float32)

    for g in grasps:
        cx_img, cy_img = g["cx"], g["cy"]
        theta = g["theta"]
        w = g["w"]

        fx = cx_img / stride
        fy = cy_img / stride

        ix = int(fx)
        iy = int(fy)

        if ix < 0 or ix >= out_w or iy < 0 or iy >= out_h:
            continue

        dx_map[iy, ix] = fx - ix
        dy_map[iy, ix] = fy - iy
        w_map[iy, ix] = w
        sin_map[iy, ix] = math.sin(theta)
        cos_map[iy, ix] = math.cos(theta)
        mask[iy, ix] = 1.0

    return {
        "dx": dx_map,
        "dy": dy_map,
        "w": w_map,
        "sin": sin_map,
        "cos": cos_map,
        "mask": mask,
    }


# -----------------------------
# High-level helper
# -----------------------------
def build_targets(
    grasps,
    img_h,
    img_w,
    stride=1,
    sigma=2.0,
):
    """
    One-stop target builder for Dataset.

    Args:
        grasps: list of dicts with keys:
            cx, cy, theta, w
        img_h, img_w: input image size
        stride: feature stride
        sigma: gaussian sigma (heatmap scale)

    Returns:
        targets: dict of numpy arrays
    """
    out_h = int(math.ceil(img_h / stride))
    out_w = int(math.ceil(img_w / stride))

    qmap = generate_qmap(
        grasps,
        out_h=out_h,
        out_w=out_w,
        stride=stride,
        sigma=sigma,
    )

    regs = generate_regression_maps(
        grasps,
        out_h=out_h,
        out_w=out_w,
        stride=stride,
    )

    targets = {
        "qmap": qmap,
        "dx": regs["dx"],
        "dy": regs["dy"],
        "w": regs["w"],
        "sin": regs["sin"],
        "cos": regs["cos"],
        "mask": regs["mask"],
    }
    return targets


# -----------------------------
# Debug / self-test
# -----------------------------
if __name__ == "__main__":
    # Fake example
    grasps = [
        {"cx": 160.0, "cy": 120.0, "theta": 0.7, "w": 60.0},
        {"cx": 200.0, "cy": 150.0, "theta": -0.5, "w": 40.0},
    ]
    targets = build_targets(
        grasps,
        img_h=240,
        img_w=320,
        stride=4,
        sigma=2.0,
    )

    print("Q-map shape:", targets["qmap"].shape)
    print("Regression keys:", list(targets.keys()))
