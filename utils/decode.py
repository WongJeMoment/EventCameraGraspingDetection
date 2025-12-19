# decode.py
import math
import numpy as np


def _to_numpy(x):
    """Accept numpy or torch tensor; returns numpy float32 array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def normalize_angle(theta):
    """Normalize to [-pi, pi)."""
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def local_max_nms(qmap, ksize=3):
    """
    Local maxima NMS via max filter. Keeps pixels that equal local max.
    qmap: (H,W)
    returns: suppressed qmap (H,W) where non-max set to 0
    """
    assert ksize in (3, 5, 7), "use small odd ksize"
    pad = ksize // 2
    H, W = qmap.shape
    padded = np.pad(qmap, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)

    maxf = np.zeros_like(qmap, dtype=np.float32)
    for dy in range(ksize):
        for dx in range(ksize):
            maxf = np.maximum(maxf, padded[dy:dy+H, dx:dx+W])

    keep = (qmap >= maxf - 1e-12)
    return np.where(keep, qmap, 0.0).astype(np.float32)


def topk_from_heatmap(qmap, k=20):
    """Get top-k (score, iy, ix) from qmap."""
    H, W = qmap.shape
    flat = qmap.reshape(-1)
    if k >= flat.size:
        idx = np.argsort(-flat)
    else:
        idx = np.argpartition(-flat, k)[:k]
        idx = idx[np.argsort(-flat[idx])]
    scores = flat[idx]
    iy = idx // W
    ix = idx % W
    return scores, iy, ix


def decode_grasps(
    Q,
    reg,
    stride=4,
    topk=20,
    conf_thresh=0.1,
    nms_ksize=3,
    clip_xy=True,
    img_hw=None,                     # (H_img, W_img) 或 None
    logw_is_w_over_stride=False,     # ✅ 关键：logw 的定义
    w_min=1.0,                       # ✅ 防止异常小宽度
    w_max=None,                      # ✅ 可选：防止异常大宽度
):
    """
    Decode to grasp list.

    Inputs:
      - Q:   (H,W) or (1,1,H,W)  grasp center confidence (0..1)
      - reg: (5,H,W) or (1,5,H,W) channels: dx, dy, logW, sin, cos

    Output:
      - list of dict:
        {"x","y","w","theta","conf","ij"}

    Width decode:
      - if logw_is_w_over_stride == False:  w = exp(logw)               # w in pixels
      - if logw_is_w_over_stride == True:   w = exp(logw) * stride      # logw = log(w/stride)
    """
    q = _to_numpy(Q)
    r = _to_numpy(reg)

    # ---- squeeze Q to (H,W) ----
    while q.ndim > 2:
        q = q[0]
    if q.ndim == 3 and q.shape[0] == 1:
        q = q[0]
    if q.ndim != 2:
        raise ValueError("Unexpected Q shape: {}".format(q.shape))

    # ---- squeeze reg to (5,H,W) ----
    while r.ndim > 3:
        r = r[0]
    if not (r.ndim == 3 and r.shape[0] == 5):
        raise ValueError("Unexpected reg shape: {} (expected (5,H,W))".format(r.shape))

    Hm, Wm = q.shape
    if r.shape[1:] != (Hm, Wm):
        raise ValueError("reg spatial {} != Q {}".format(r.shape[1:], q.shape))

    # ---- detector: threshold -> NMS -> topK ----
    q_thr = np.where(q >= conf_thresh, q, 0.0).astype(np.float32)
    q_nms = local_max_nms(q_thr, ksize=nms_ksize)
    scores, ys, xs = topk_from_heatmap(q_nms, k=topk)

    dx, dy, logw, sin_t, cos_t = r[0], r[1], r[2], r[3], r[4]

    cands = []
    for s, iy, ix in zip(scores, ys, xs):
        conf = float(s)
        if conf <= 0.0:
            continue

        # center decode
        fx = float(ix + dx[iy, ix])
        fy = float(iy + dy[iy, ix])
        x = fx * stride
        y = fy * stride

        # width decode (关键)
        w = float(math.exp(float(logw[iy, ix])))
        if logw_is_w_over_stride:
            w *= float(stride)

        # clamp width
        if w_min is not None:
            w = max(float(w_min), w)
        if w_max is not None:
            w = min(float(w_max), w)

        # angle decode
        th = float(math.atan2(float(sin_t[iy, ix]), float(cos_t[iy, ix])))
        th = normalize_angle(th)

        # clamp xy
        if clip_xy:
            if img_hw is None:
                Himg, Wimg = Hm * stride, Wm * stride
            else:
                Himg, Wimg = int(img_hw[0]), int(img_hw[1])
            x = float(np.clip(x, 0.0, Wimg - 1.0))
            y = float(np.clip(y, 0.0, Himg - 1.0))

        cands.append({
            "x": x, "y": y, "w": w, "theta": th, "conf": conf,
            "ij": (int(iy), int(ix)),
        })

    cands.sort(key=lambda d: d["conf"], reverse=True)
    return cands


if __name__ == "__main__":
    # sanity test
    Hm, Wm = 60, 80
    Q = np.zeros((Hm, Wm), dtype=np.float32)
    Q[20, 30] = 0.9

    reg = np.zeros((5, Hm, Wm), dtype=np.float32)
    reg[0, :, :] = 0.3                  # dx
    reg[1, :, :] = 0.6                  # dy
    reg[2, :, :] = math.log(50.0)       # logw
    reg[3, :, :] = math.sin(0.7)        # sin
    reg[4, :, :] = math.cos(0.7)        # cos

    cands = decode_grasps(Q, reg, stride=4, topk=5, conf_thresh=0.1, img_hw=(240, 320))
    print(cands[:1])
