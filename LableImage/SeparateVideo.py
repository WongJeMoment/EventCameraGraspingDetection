import os
import cv2

def split_avi_to_frames(video_path, out_dir, prefix="frame", start_index=0, ext="png"):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    idx = start_index
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_path = os.path.join(out_dir, f"{prefix}_{idx:06d}.{ext}")
        ok = cv2.imwrite(out_path, frame)
        if not ok:
            raise RuntimeError(f"Failed to write frame to: {out_path}")

        idx += 1
        saved += 1

    cap.release()
    print(f"Done. Saved {saved} frames to: {out_dir}")

if __name__ == "__main__":
    split_avi_to_frames(
        video_path="/home/wangzhe/ICME2026/MyDataset/Video/st.avi",
        out_dir="/home/wangzhe/ICME2026/MyDataset/IMG/st",
        prefix="frame",
        ext="png"   # 也可以用 "jpg"
    )
