import numpy as np
def shrink_grasp_width(box_pts: np.ndarray, ratio: float = 0.5) -> np.ndarray:
    """
    把抓取四边形沿“宽度(短边)”方向缩放到 ratio 倍。
    要求 box_pts shape=(4,2)，按 decode 输出的顺序通常是一个矩形的四个角点。
    不改变中心、角度、长度，仅改变宽度。
    """
    pts = box_pts.astype(np.float32).copy()

    # 两组对边的长度（0-1, 2-3）和（1-2, 3-0）
    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    l01 = float(np.linalg.norm(e01))
    l12 = float(np.linalg.norm(e12))

    # 选择“宽度方向”为短边的法向量（unit）
    if l01 < l12:
        w_dir = e01 / (l01 + 1e-6)  # 宽度方向（沿短边）
        # 宽度成对点：(0,1) 和 (3,2) 这两条边
        pairA = (0, 1)
        pairB = (3, 2)
    else:
        w_dir = e12 / (l12 + 1e-6)
        # 宽度成对点：(1,2) 和 (0,3)
        pairA = (1, 2)
        pairB = (0, 3)

    # 每条“宽边”上的两个点，向这条边的中点收缩
    def shrink_pair(i, j):
        mid = 0.5 * (pts[i] + pts[j])
        pts[i] = mid + (pts[i] - mid) * ratio
        pts[j] = mid + (pts[j] - mid) * ratio

    shrink_pair(*pairA)
    shrink_pair(*pairB)

    return pts

def shrink_grasp_length(box_pts: np.ndarray, ratio: float = 0.5) -> np.ndarray:
    """
    把抓取四边形沿“长度(长边)”方向缩放到 ratio 倍。
    - 中心不变
    - 角度不变
    - 宽度不变
    - 只缩短长度

    box_pts: (4,2) numpy array，矩形四点
    """
    pts = box_pts.astype(np.float32).copy()

    # 两组相邻边
    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    l01 = float(np.linalg.norm(e01))
    l12 = float(np.linalg.norm(e12))

    # 判断哪一组是“长度方向”（长边）
    if l01 > l12:
        # 0-1 是长边，对边是 2-3
        pairA = (0, 1)
        pairB = (3, 2)
    else:
        # 1-2 是长边，对边是 0-3
        pairA = (1, 2)
        pairB = (0, 3)

    def shrink_pair(i, j):
        mid = 0.5 * (pts[i] + pts[j])
        pts[i] = mid + (pts[i] - mid) * ratio
        pts[j] = mid + (pts[j] - mid) * ratio

    # 对“长度方向”的两条对边做收缩
    shrink_pair(*pairA)
    shrink_pair(*pairB)

    return pts
