import os

def rename_to_numbers(folder, ext=".png", start=1):
    files = sorted(os.listdir(folder))
    idx = start

    for fname in files:
        old_path = os.path.join(folder, fname)
        if not os.path.isfile(old_path):
            continue

        new_name = f"{idx}{ext}"
        new_path = os.path.join(folder, new_name)

        if os.path.exists(new_path):
            raise FileExistsError(f"File already exists: {new_path}")

        os.rename(old_path, new_path)
        idx += 1

    print(f"Done. Renamed {idx - start} files in {folder}")


if __name__ == "__main__":
    rename_to_numbers(
        folder="/home/wangzhe/ICME2026/MyDataset/IMG/st",  # 🔁 改成你的文件夹路径
        ext=".png",           # 如果是 jpg 就改成 ".jpg"
        start=1               # 从 1 开始
    )
