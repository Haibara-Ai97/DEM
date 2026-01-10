import argparse
from pathlib import Path
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def read_classes(classes_path: Path):
    if not classes_path.exists():
        return None
    names = [l.strip() for l in classes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return names

def parse_yolo_label_file(label_path: Path):
    """
    返回：class_ids(list[int]), num_boxes(int)
    若文件不存在或为空，返回空列表/0
    """
    if (not label_path.exists()) or label_path.stat().st_size == 0:
        return [], 0
    class_ids = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # YOLO: class xc yc w h
            try:
                cid = int(float(parts[0]))
                class_ids.append(cid)
            except Exception:
                continue
    return class_ids, len(class_ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True,
                    help="包含 train/valid/test 目录与 classes.txt 的根目录")
    ap.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--abs_paths", action="store_true", help="写入绝对路径（推荐）")
    ap.add_argument("--with_label_stats", action="store_true",
                    help="额外写入 label_path/has_bbox/num_boxes/labels 列（便于后续 bbox 引导）")
    ap.add_argument("--require_label", action="store_true",
                    help="仅保留有对应 label 文件且至少 1 个框的图像")
    ap.add_argument("--sample_n", type=int, default=0,
                    help="从 split 中随机抽样 N 张（0 表示不抽样）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    img_dir = root / args.split / "images"
    lbl_dir = root / args.split / "labels"
    assert img_dir.exists(), f"images dir not found: {img_dir}"
    assert lbl_dir.exists(), f"labels dir not found: {lbl_dir}"

    classes = read_classes(root / "classes.txt")

    # 收集图片
    image_paths = []
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            image_paths.append(p)

    image_paths = sorted(image_paths)
    rows = []
    for img_path in image_paths:
        stem = img_path.stem
        label_path = lbl_dir / f"{stem}.txt"

        if args.with_label_stats or args.require_label:
            cids, nbox = parse_yolo_label_file(label_path)
            has_bbox = int(nbox > 0)
            if args.require_label and not has_bbox:
                continue

            # 映射到类别名（若有 classes.txt）
            if classes is not None and len(classes) > 0:
                names = sorted({classes[c] for c in cids if 0 <= c < len(classes)})
            else:
                names = sorted({str(c) for c in cids})
            label_names = ";".join(names)
        else:
            has_bbox = ""
            nbox = ""
            label_names = ""

        rows.append({
            "image_path": str(img_path.resolve()) if args.abs_paths else str(img_path),
            **({
                "label_path": str(label_path.resolve()) if args.abs_paths else str(label_path),
                "has_bbox": has_bbox,
                "num_boxes": nbox,
                "labels": label_names,
                "split": args.split
            } if args.with_label_stats else {})
        })

    df = pd.DataFrame(rows)

    # 抽样（用于你说的 1k+ 场景）
    if args.sample_n and args.sample_n > 0 and len(df) > args.sample_n:
        df = df.sample(n=args.sample_n, random_state=args.seed).reset_index(drop=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path}  rows={len(df)}  cols={list(df.columns)}")

if __name__ == "__main__":
    main()
