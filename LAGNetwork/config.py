# config.py
from dataclasses import dataclass
import torch


@dataclass
class TrainConfig:
    # -------------------------
    # Data paths
    # -------------------------
    train_images: str = "/home/wangzhe/ICME2026/ALLDATASET/IMG/apple"
    train_labels: str = "/home/wangzhe/ICME2026/ALLDATASET/Lable/apple"
    val_images: str = "/home/wangzhe/ICME2026/ALLDATASET/IMG/apple"
    val_labels: str = "/home/wangzhe/ICME2026/ALLDATASET/Lable/apple"

    # -------------------------
    # Training hyperparams
    # -------------------------
    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 1
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # -------------------------
    # Model
    # -------------------------
    base_channels: int = 32

    # -------------------------
    # Supervision from boxes
    # -------------------------
    max_boxes_used: int = 1  # 每张图最多用多少个框来生成 GT maps（超出忽略）

    # -------------------------
    # Loss weights
    # -------------------------
    w_q: float = 1.0
    w_a: float = 1.0
    w_w: float = 1.0

    # -------------------------
    # Misc
    # -------------------------
    ckpt_dir: str = "ckpt_lag"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
