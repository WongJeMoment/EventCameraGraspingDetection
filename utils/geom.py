# geom.py
from __future__ import annotations
import math
from typing import Iterable, Tuple, Optional

import numpy as np

try:
    import cv2  # optional but recommended
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False


EPS = 1e-8


# -----------------------------
# Basic helpers
# -----------------------------
def to_np_pts(pts: Iterable[Iterable[float]]) -> np.ndarray:
    """Convert to (N,2) float32 numpy array."""
    arr = np.asarray(list(pts), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"pts must be (N,2), got {arr.shape}")
    return arr


def normalize_angle(theta: float) -> float:
    """Normalize angle to [-pi, pi)."""
    t = (theta + math.pi) % (2.0 * math.pi) - math.pi
    return t


def unit(v: np.ndarray) -> np.ndarray:
    """Return unit vector of shape (2,)."""
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros_like(v)
    return v / n


def align_same_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Flip v if opposite to u."""
    if float(np.dot(u, v)) < 0:
        return -v
    return v


# -----------------------------
# Point ordering for 4 corners
# -----------------------------
def order_points_convex_hull(pts4: np.ndarray, clockwise: bool = True) -> np.ndarray:
    """
    Robustly order 4 (possibly unordered) corner points in a consistent loop.

    Returns:
        ordered_pts: (4,2) in clockwise (default) or counterclockwise order.
    """
    pts4 = np.asarray(pts4, dtype=np.float32).reshape(-1, 2)
    if pts4.shape[0] != 4:
        raise ValueError(f"Expected 4 points, got {pts4.shape[0]}")

    if _HAS_CV2:
        hull = cv2.convexHull(pts4, clockwise=clockwise)  # (4,1,2)
        hull = hull.reshape(-1, 2)
        if hull.shape[0] == 4:
            return hull

    # Fallback: angle sort around centroid (works for convex 4-point rectangles)
    c = pts4.mean(axis=0)
    ang = np.arctan2(pts4[:, 1] - c[1], pts4[:, 0] - c[0])
    idx = np.argsort(ang)  # CCW
    ordered = pts4[idx]
    if clockwise:
        ordered = ordered[::-1]
    return ordered


def canonical_start(ordered_pts: np.ndarray) -> np.ndarray:
    """
    Rotate the ordered loop so that it starts from a deterministic vertex.
    Here: start from the point with smallest (x + y) (top-left-ish in image coords).
    """
    p = np.asarray(ordered_pts, dtype=np.float32).reshape(4, 2)
    scores = p[:, 0] + p[:, 1]
    k = int(np.argmin(scores))
    return np.roll(p, -k, axis=0)


# -----------------------------
# Geometry extraction
# -----------------------------
def rect_params_from_4pts(
    pts4: Iterable[Iterable[float]],
    theta_ref: str = "approach",
    use_opposite_edge_average: bool = True,
) -> Tuple[float, float, float, float, float]:
    """
    Convert 4 corner points of a grasp rectangle to (cx, cy, theta, w, h).

    Args:
        pts4: 4 corner points (unordered ok) [[x,y],...]
        theta_ref:
            - "jaw": theta is direction along jaw line (short edge direction).
            - "approach": theta is grasp approach direction (jaw normal) = jaw + pi/2.
        use_opposite_edge_average:
            If True, compute direction by averaging opposite edges -> more stable.

    Returns:
        cx, cy, theta(rad), w, h
        where w = jaw opening width in pixels (short side length),
              h = contact length in pixels (long side length).
    """
    pts = to_np_pts(pts4)
    loop = order_points_convex_hull(pts, clockwise=True)
    loop = canonical_start(loop)

    # edges in order: p0->p1, p1->p2, p2->p3, p3->p0
    e0 = loop[1] - loop[0]
    e1 = loop[2] - loop[1]
    e2 = loop[3] - loop[2]
    e3 = loop[0] - loop[3]

    l0 = float(np.linalg.norm(e0))
    l1 = float(np.linalg.norm(e1))
    l2 = float(np.linalg.norm(e2))
    l3 = float(np.linalg.norm(e3))

    # Opposite pairs: (e0,e2) and (e1,e3)
    pairA = 0.5 * (l0 + l2)
    pairB = 0.5 * (l1 + l3)

    # Define short side = w, long side = h
    if pairA <= pairB:
        w = pairA
        h = pairB
        short_edge_u = unit(e0)
        opp_short_u = unit(-e2)  # opposite direction aligned with e0
    else:
        w = pairB
        h = pairA
        short_edge_u = unit(e1)
        opp_short_u = unit(-e3)

    if use_opposite_edge_average:
        opp_short_u = align_same_direction(short_edge_u, opp_short_u)
        u = unit(short_edge_u + opp_short_u)
        if float(np.linalg.norm(u)) < EPS:
            u = short_edge_u
    else:
        u = short_edge_u

    theta_jaw = math.atan2(float(u[1]), float(u[0]))
    theta = theta_jaw if theta_ref == "jaw" else (theta_jaw + math.pi / 2.0)
    theta = normalize_angle(theta)

    cx = float(loop[:, 0].mean())
    cy = float(loop[:, 1].mean())

    return cx, cy, theta, float(w), float(h)


def grasp_params_from_2pts(
    p1: Iterable[float],
    p2: Iterable[float],
    theta_ref: str = "approach",
) -> Tuple[float, float, float, float]:
    """
    Convert two finger contact points to (cx, cy, theta, w).

    Args:
        p1, p2: (x,y) contact points
        theta_ref:
            - "jaw": theta along the line connecting contacts
            - "approach": theta normal to that line (+pi/2)

    Returns:
        cx, cy, theta(rad), w(pixels)
    """
    a = np.asarray(p1, dtype=np.float32).reshape(2)
    b = np.asarray(p2, dtype=np.float32).reshape(2)
    v = b - a
    w = float(np.linalg.norm(v))
    u = unit(v)
    theta_jaw = math.atan2(float(u[1]), float(u[0]))
    theta = theta_jaw if theta_ref == "jaw" else (theta_jaw + math.pi / 2.0)
    theta = normalize_angle(theta)
    cx = float((a[0] + b[0]) * 0.5)
    cy = float((a[1] + b[1]) * 0.5)
    return cx, cy, theta, w


# -----------------------------
# Convenience: 8 numbers -> 4 points
# -----------------------------
def pts4_from_8nums(nums8: Iterable[float]) -> np.ndarray:
    """
    nums8: [x1,y1,x2,y2,x3,y3,x4,y4] or an iterable length 8.
    Returns: (4,2) array.
    """
    arr = np.asarray(list(nums8), dtype=np.float32).reshape(-1)
    if arr.size != 8:
        raise ValueError(f"Expected 8 numbers, got {arr.size}")
    return arr.reshape(4, 2)


# -----------------------------
# Quick self-test (optional)
# -----------------------------
if __name__ == "__main__":
    # Example: 4-point rectangle (unordered)
    pts = np.array([[122.8, 100.8],
                    [205.2, 141.1],
                    [209.5, 115.5],
                    [118.4, 126.4]], dtype=np.float32)

    cx, cy, theta, w, h = rect_params_from_4pts(pts, theta_ref="approach")
    print("4pts ->", cx, cy, theta, w, h)

    # Example: 2-point contacts
    cx2, cy2, th2, w2 = grasp_params_from_2pts((87.5, 44.0), (124.9, 83.2), theta_ref="approach")
    print("2pts ->", cx2, cy2, th2, w2)
