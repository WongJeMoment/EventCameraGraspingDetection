import os
import cv2
import glob

# ========= 配置区 =========
IMG_DIR = "/home/wangzhe/ICME2026/MyDataset/IMG/l"        # 图片目录
OUT_DIR = "/home/wangzhe/ICME2026/MyDataset/Lable/l_ranc"        # 输出txt目录（YOLO格式）
CLASSES = ["l"]      # 类别列表（可改成你的类别，比如 ["person","car"]）
# =========================

os.makedirs(OUT_DIR, exist_ok=True)

drawing = False
ix, iy = -1, -1
boxes = []  # 存储 (class_id, x1, y1, x2, y2)
current_class = 0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def to_yolo(h, w, x1, y1, x2, y2):
    # 确保 x1<x2, y1<y2
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    bw = x2 - x1
    bh = y2 - y1
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0

    # 归一化
    return xc / w, yc / h, bw / w, bh / h

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, boxes, current_class, img_show

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_show = img.copy()
        cv2.rectangle(img_show, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.putText(img_show, f"class: {current_class} ({CLASSES[current_class]})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 画已完成的框
        for cid, x1, y1, x2, y2 in boxes:
            cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_show, f"{cid}:{CLASSES[cid]}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1 = clamp(ix, 0, img.shape[1]-1)
        y1 = clamp(iy, 0, img.shape[0]-1)
        x2 = clamp(x, 0, img.shape[1]-1)
        y2 = clamp(y, 0, img.shape[0]-1)

        # 太小的框不要
        if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
            boxes.append((current_class, x1, y1, x2, y2))

def save_label_txt(img_path, boxes, out_dir):
    h, w = img.shape[:2]
    base = os.path.splitext(os.path.basename(img_path))[0]
    txt_path = os.path.join(out_dir, base + ".txt")

    lines = []
    for cid, x1, y1, x2, y2 in boxes:
        xc, yc, bw, bh = to_yolo(h, w, x1, y1, x2, y2)
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Saved] {txt_path}  ({len(lines)} boxes)")

# 读取图片
exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
img_paths = []
for e in exts:
    img_paths += glob.glob(os.path.join(IMG_DIR, e))
img_paths.sort()

if not img_paths:
    raise FileNotFoundError(f"在目录 '{IMG_DIR}' 没找到图片。请把图片放进去。")

cv2.namedWindow("Label Tool", cv2.WINDOW_NORMAL)

for idx, img_path in enumerate(img_paths):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Skip] 读图失败: {img_path}")
        continue

    boxes = []
    img_show = img.copy()
    cv2.setMouseCallback("Label Tool", mouse_callback)

    while True:
        # 刷新画面（包含已完成的框）
        img_show = img.copy()
        for cid, x1, y1, x2, y2 in boxes:
            cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_show, f"{cid}:{CLASSES[cid]}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        info = f"[{idx+1}/{len(img_paths)}] {os.path.basename(img_path)} | class={current_class}:{CLASSES[current_class]} | boxes={len(boxes)}"
        cv2.putText(img_show, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(img_show, "Keys: 0-9 switch class | u undo | s save | n save+next | q quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Label Tool", img_show)
        k = cv2.waitKey(20) & 0xFF

        # 切换类别（0-9）
        if ord('0') <= k <= ord('9'):
            cid = k - ord('0')
            if cid < len(CLASSES):
                current_class = cid
                print(f"[Class] {current_class} -> {CLASSES[current_class]}")

        elif k == ord('u'):
            if boxes:
                boxes.pop()
                print("[Undo] remove last box")

        elif k == ord('s'):
            save_label_txt(img_path, boxes, OUT_DIR)

        elif k == ord('n'):
            save_label_txt(img_path, boxes, OUT_DIR)
            break

        elif k == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit

cv2.destroyAllWindows()
print("Done.")
