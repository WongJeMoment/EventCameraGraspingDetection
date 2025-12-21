# grasp_dataset_txt.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class GraspTxtDataset(Dataset):
    """
    支持两种 txt 格式（自动识别）：

    格式 A（你现在用的）：
        每行: x y
        4 行 = 1 个矩形框

    格式 B：
        每行: x1 y1 x2 y2 x3 y3 x4 y4
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        image_size: int = 256,
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size

        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def _read_txt(self, txt_path):
        """
        return:
            Tensor (K, 8)  像素坐标
        """
        if not os.path.exists(txt_path):
            return torch.zeros((0, 8), dtype=torch.float32)

        points = []
        boxes = []

        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                nums = list(map(float, line.split()))

                # 情况 B：一行 8 个数 = 一个框
                if len(nums) == 8:
                    boxes.append(nums)

                # 情况 A：一行 2 个数 = 一个点
                elif len(nums) == 2:
                    points.append(nums)

                else:
                    raise ValueError(
                        f"[ERROR] 无法识别的 txt 格式: {txt_path}, 行内容: {line}"
                    )

        # 如果是 “每行一个点” 的情况
        if len(points) > 0:
            assert len(points) % 4 == 0, \
                f"[ERROR] 点的数量不是 4 的倍数: {txt_path}"

            for i in range(0, len(points), 4):
                p = points[i:i + 4]
                boxes.append([
                    p[0][0], p[0][1],
                    p[1][0], p[1][1],
                    p[2][0], p[2][1],
                    p[3][0], p[3][1],
                ])

        if len(boxes) == 0:
            return torch.zeros((0, 8), dtype=torch.float32)

        return torch.tensor(boxes, dtype=torch.float32)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        txt_path = os.path.join(
            self.labels_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        img_tensor = self.transform(img)

        boxes_px = self._read_txt(txt_path)

        # 像素坐标 -> 归一化
        if boxes_px.numel() > 0:
            boxes = boxes_px.clone()
            boxes[:, 0::2] /= W
            boxes[:, 1::2] /= H
        else:
            boxes = boxes_px

        return img_tensor, boxes
