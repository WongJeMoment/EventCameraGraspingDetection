import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def load_points(txt_path: Path) -> np.ndarray:
    pts = np.loadtxt(txt_path)
    pts = np.atleast_2d(pts)
    assert pts.shape[1] == 2
    return pts

def process_one(img_path: Path, txt_path: Path, save_path: Path, group_size=4):
    img = np.array(Image.open(img_path).convert("RGB"))
    pts = load_points(txt_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(img)

    # 画点
    plt.scatter(pts[:, 0], pts[:, 1], s=30, c="cyan")

    # 标序号
    for idx, (x, y) in enumerate(pts):
        plt.text(x + 3, y + 3, str(idx), color="yellow", fontsize=8)

    # 每4个点一组画框
    n = len(pts)
    valid_n = (n // group_size) * group_size
    for i in range(0, valid_n, group_size):
        quad = pts[i:i + group_size]
        closed = np.vstack([quad, quad[0]])
        plt.plot(closed[:, 0], closed[:, 1], linewidth=2, color="red")

    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close()  # ⚠️ 非常重要：防止内存爆炸

def batch_process():
    img_dir = Path("/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Images_dvs_bg")
    txt_dir = Path("/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Annotations_angle_regularize")
    out_dir = Path("/home/wangzhe/ICME2026/NeuroGrasp_Dataset/Value")

    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.png"))

    print(f"Found {len(img_paths)} images")

    for img_path in img_paths:
        name = img_path.stem
        txt_path = txt_dir / f"{name}.txt"

        if not txt_path.exists():
            print(f"[Skip] No txt for {name}")
            continue

        save_path = out_dir / f"{name}.png"
        process_one(img_path, txt_path, save_path)

        print(f"[OK] {name}")

if __name__ == "__main__":
    batch_process()
