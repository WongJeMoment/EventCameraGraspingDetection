# config.py
"""
Central configuration file for NeuroGrasp training.

Usage:
    from config import cfg
"""

from types import SimpleNamespace


# -------------------------
# Dataset config
# -------------------------
DATASET = SimpleNamespace(
    img_dir="/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Images_dvs_bg",
    ann_dir="/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Annotations_angle_regularize",

    resize=False,          # whether resize input images
    height=240,
    width=320,

    theta_ref="approach",
)


# -------------------------
# Model config
# -------------------------
MODEL = SimpleNamespace(
    backbone="mobilenetv3_small",   # mobilenetv3_small | shufflenetv2
    width_mult=1.0,

    stride=4,       # output stride for grasp heatmap
    sigma=2.0,      # gaussian sigma for Q-map

    # head / decoder behavior
    normalize_sincos=True,
    dxdy_sigmoid=True,
    logw_clamp=(-5.0, 6.0),
)


# -------------------------
# Training config
# -------------------------
TRAIN = SimpleNamespace(
    epochs=30,
    batch_size=8,
    workers=4,

    lr=1e-3,
    weight_decay=1e-4,

    w_reg=5.0,          # regression loss weight
    grad_clip=5.0,
)


# -------------------------
# Logging / checkpoint
# -------------------------
LOG = SimpleNamespace(
    print_every=20,
    vis_every=200,
    vis_conf=0.2,

    ckpt_every=1000,
    ckpt_dir="checkpoints",
    vis_dir="train_vis",
)


# -------------------------
# Global config object
# -------------------------
cfg = SimpleNamespace(
    dataset=DATASET,
    model=MODEL,
    train=TRAIN,
    log=LOG,
)
