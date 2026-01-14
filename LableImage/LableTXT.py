import os
import glob
import cv2

# ---------------- 配置 ----------------
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
WINDOW = "Point Labeler (Zoom)"

# 显示放大倍数（只影响显示，不影响保存坐标）
DISPLAY_SCALE = 2.0  # 1.5 / 2.0 / 3.0 都可以

# 绘制参数（更粗更清晰）
PT_RADIUS = 6
LINE_THICKNESS = 3
TEXT_SCALE = 0.8
TEXT_THICKNESS = 2

# -------------------------------------
# 当前图像的标注：list of groups, each group is list of (x,y)
groups = [[]]
img = None           # 原图
img_show = None      # 原图上画完标注（未缩放）
cur_path = ""
cur_idx = 0
img_paths = []
out_dir = ""


def txt_path_for_image(img_path: str, out_dir_: str) -> str:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(out_dir_, stem + ".txt")


def load_points_from_txt(txt_path: str):
    """读取已有标注：支持空行分组，每行两个浮点数"""
    if not os.path.exists(txt_path):
        return [[]]

    gs = [[]]
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                # 空行：新组
                if len(gs[-1]) > 0:
                    gs.append([])
                continue

            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    gs[-1].append((x, y))
                except:
                    pass

    if len(gs) == 0:
        gs = [[]]
    return gs


def save_points_to_txt(txt_path: str, gs):
    """写入：每行一个点，组之间空行分隔"""
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for gi, g in enumerate(gs):
            for (x, y) in g:
                f.write(f"{x:.8f} {y:.8f}\n")
            if gi != len(gs) - 1 and len(g) > 0:
                f.write("\n")


def redraw_and_show():
    """在原图上绘制标注，然后缩放显示（不改变保存坐标）"""
    global img_show
    if img is None:
        return

    img_show = img.copy()

    # 画点 + 序号 + 连线
    for gi, g in enumerate(groups):
        # 点 & 序号
        for i, (x, y) in enumerate(g):
            cx, cy = int(round(x)), int(round(y))
            cv2.circle(img_show, (cx, cy), PT_RADIUS, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                img_show, str(i + 1),
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE,
                (0, 255, 0), TEXT_THICKNESS, lineType=cv2.LINE_AA
            )

        # 折线
        if len(g) >= 2:
            pts = [(int(round(x)), int(round(y))) for (x, y) in g]
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(img_show, a, b, (0, 255, 255), LINE_THICKNESS, lineType=cv2.LINE_AA)

    # 显示时缩放
    disp = cv2.resize(
        img_show, None,
        fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_LINEAR
    )

    # 信息提示（写在缩放后的图上）
    info1 = f"[{cur_idx+1}/{len(img_paths)}] {os.path.basename(cur_path)}"
    info2 = "LClick:add  RClick/BS:undo  e:new group  s:save  n:save+next  p:prev  c:clear  q/ESC:quit"
    cv2.putText(disp, info1, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(disp, info2, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(WINDOW, disp)


def undo_last_point():
    global groups
    if len(groups) == 0:
        groups = [[]]
        return
    if len(groups[-1]) > 0:
        groups[-1].pop()
    else:
        if len(groups) > 1:
            groups.pop()
            if len(groups[-1]) > 0:
                groups[-1].pop()


def mouse_cb(event, x, y, flags, param):
    global groups

    # 显示坐标 -> 原图坐标
    ox = x / DISPLAY_SCALE
    oy = y / DISPLAY_SCALE

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(groups) == 0:
            groups = [[]]
        groups[-1].append((float(ox), float(oy)))
        redraw_and_show()

    elif event == cv2.EVENT_RBUTTONDOWN:
        undo_last_point()
        redraw_and_show()


def open_image_at(idx: int):
    global img, cur_path, groups, cur_idx
    cur_idx = max(0, min(idx, len(img_paths) - 1))
    cur_path = img_paths[cur_idx]

    img0 = cv2.imread(cur_path)
    if img0 is None:
        raise FileNotFoundError(cur_path)
    img = img0

    # 自动加载已有标注（如果存在）
    tpath = txt_path_for_image(cur_path, out_dir)
    gs = load_points_from_txt(tpath)
    groups = gs if len(gs) > 0 else [[]]
    if len(groups) == 0:
        groups = [[]]

    redraw_and_show()


def main(img_dir: str, out_txt_dir: str):
    global img_paths, out_dir
    out_dir = out_txt_dir
    os.makedirs(out_dir, exist_ok=True)

    # 收集图像
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(img_dir, ext)))
    img_paths = sorted(paths)

    if len(img_paths) == 0:
        raise RuntimeError(f"No images found in {img_dir}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, mouse_cb)

    open_image_at(0)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
            break

        elif key == ord('c'):
            # 清空当前图所有点
            global groups
            groups = [[]]
            redraw_and_show()

        elif key == ord('e'):
            # 结束当前组，开始新组（txt 里用空行分隔组）
            if len(groups) == 0:
                groups = [[]]
            if len(groups[-1]) > 0:
                groups.append([])
            redraw_and_show()

        elif key == 8:  # Backspace
            undo_last_point()
            redraw_and_show()

        elif key == ord('s'):
            # 保存
            tpath = txt_path_for_image(cur_path, out_dir)
            gs_to_save = groups[:]
            while len(gs_to_save) > 1 and len(gs_to_save[-1]) == 0:
                gs_to_save.pop()
            save_points_to_txt(tpath, gs_to_save)
            print(f"[SAVE] {tpath}")

        elif key == ord('n'):
            # 保存并下一张
            tpath = txt_path_for_image(cur_path, out_dir)
            gs_to_save = groups[:]
            while len(gs_to_save) > 1 and len(gs_to_save[-1]) == 0:
                gs_to_save.pop()
            save_points_to_txt(tpath, gs_to_save)
            print(f"[SAVE] {tpath}")

            if cur_idx + 1 < len(img_paths):
                open_image_at(cur_idx + 1)

        elif key == ord('p'):
            # 上一张
            if cur_idx - 1 >= 0:
                open_image_at(cur_idx - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 改成你的路径
    IMAGE_DIR = "/home/wangzhe/ICME2026/MyDataset/IMG/l"
    OUT_TXT_DIR = "/home/wangzhe/ICME2026/MyDataset/Lable/l"
    main(IMAGE_DIR, OUT_TXT_DIR)
