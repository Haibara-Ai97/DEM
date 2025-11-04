import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 你 classes.txt 里的类别名（Crack/Breakage/Comb/Hole/Reinforcement/Seepage）:contentReference[oaicite:2]{index=2}
# 为了与论文/领域表述一致，建议映射到更标准的工程英文术语：
DEFAULT_CANONICAL_MAP = {
    "Crack": "crack",
    "Breakage": "spalling",          # 破损/剥落更接近 spalling（建议你论文里也这样统一术语）
    "Comb": "honeycomb",
    "Hole": "hole",
    "Reinforcement": "exposed rebar",
    "Seepage": "seepage",
}

SYSTEM_PROMPT = (
    "You are a professional bridge inspection assistant. "
    "Answer questions about concrete surface defects precisely and concisely."
)

# --------- Helpers ----------
def read_lines(p: Path) -> List[str]:
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def parse_yolo_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    YOLO label format: class_id x_center y_center width height (normalized 0..1)
    Returns list of (cid, xc, yc, w, h)
    """
    if (not label_path.exists()) or label_path.stat().st_size == 0:
        return []
    items = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        items.append((cid, xc, yc, w, h))
    return items

def bbox_area(w: float, h: float) -> float:
    return max(w, 0.0) * max(h, 0.0)

def grid_cell(xc: float, yc: float, grid: int) -> str:
    """
    grid cell id in "r{row}c{col}" (1-indexed), top-left is r1c1.
    """
    col = min(grid - 1, max(0, int(xc * grid)))
    row = min(grid - 1, max(0, int(yc * grid)))
    return f"r{row+1}c{col+1}"

def severity_from_area_ratio(area_ratio: float, t1: float, t2: float) -> str:
    # small / medium / large
    if area_ratio < t1:
        return "small"
    if area_ratio < t2:
        return "medium"
    return "large"

def choose_primary_box(boxes: List[Tuple[int, float, float, float, float]]) -> Tuple[int, float, float, float, float]:
    # pick max area
    return max(boxes, key=lambda x: bbox_area(x[3], x[4]))

# --------- Question templates ----------
YN_TEMPLATES = [
    "Is there any {defect} in the image? Answer Yes or No.",
    "Does the image show {defect}? Answer Yes or No.",
]
MULTI_LABEL_TEMPLATES = [
    "What defect types are present in the image? Reply with a comma-separated list. If none, reply 'none'.",
]
COUNT_TEMPLATES = [
    "How many {defect} regions are visible? Reply with an integer.",
    "Count the number of {defect} instances. Reply with an integer.",
]
GRID_TEMPLATES = [
    "In which grid cell ({g}x{g}) is the primary {defect} located? Reply in the format r#c#.",
    "Locate the primary {defect} on a {g}x{g} grid. Reply in the format r#c#.",
]
JSON_TEMPLATES = [
    "Generate a JSON inspection summary with the schema: "
    "{\"defects\":[{\"type\":...,\"count\":...,\"severity\":...,\"primary_cell\":...}],\"overall_condition\":...}. "
    "Use only the listed defect types. Reply with JSON only.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True,
                    help="Root containing train/valid/test folders and classes.txt")
    ap.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--grid", type=int, default=4, help="Grid size for localization, e.g., 4 => 4x4")
    ap.add_argument("--seed", type=int, default=42)

    # 控制每张图生成多少样本（避免数据爆炸）
    ap.add_argument("--yn_per_image", type=int, default=2, help="Yes/No questions per image (mixed pos/neg)")
    ap.add_argument("--make_multilabel", action="store_true")
    ap.add_argument("--make_count", action="store_true")
    ap.add_argument("--make_grid", action="store_true")
    ap.add_argument("--make_json", action="store_true")

    # 严重度阈值（用 bbox area ratio 代理）
    ap.add_argument("--sev_t1", type=float, default=0.01, help="small/medium threshold")
    ap.add_argument("--sev_t2", type=float, default=0.05, help="medium/large threshold")

    # 可选：只采样 N 张图（便于快速实验）
    ap.add_argument("--max_images", type=int, default=0)

    args = ap.parse_args()
    random.seed(args.seed)

    root = Path(args.dataset_root)
    img_dir = root / args.split / "images"
    lbl_dir = root / args.split / "labels"
    classes_path = root / "classes.txt"
    assert img_dir.exists(), f"Missing: {img_dir}"
    assert lbl_dir.exists(), f"Missing: {lbl_dir}"
    assert classes_path.exists(), f"Missing: {classes_path}"

    class_names = read_lines(classes_path)  # original
    # canonical mapping
    canonical = {}
    for name in class_names:
        canonical[name] = DEFAULT_CANONICAL_MAP.get(name, name.lower())

    # Collect image files
    imgs = [p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if args.max_images and args.max_images > 0:
        imgs = imgs[: args.max_images]

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def all_defect_vocab() -> List[str]:
        return [canonical[n] for n in class_names]

    vocab = all_defect_vocab()

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for img_path in imgs:
            stem = img_path.stem
            label_path = lbl_dir / f"{stem}.txt"
            boxes = parse_yolo_label_file(label_path)

            # group by canonical defect
            by_def: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
            for (cid, xc, yc, w, h) in boxes:
                if 0 <= cid < len(class_names):
                    d = canonical[class_names[cid]]
                else:
                    d = f"class_{cid}"
                by_def.setdefault(d, []).append((cid, xc, yc, w, h))

            present = sorted(by_def.keys())

            # 1) Yes/No (mixed positives + negatives)
            # choose some defects to ask about
            ask_defs = []
            # positives
            if present:
                ask_defs.extend(random.sample(present, k=min(len(present), max(1, args.yn_per_image // 2))))
            # negatives
            neg_pool = [d for d in vocab if d not in present]
            if neg_pool:
                ask_defs.extend(random.sample(neg_pool, k=min(len(neg_pool), args.yn_per_image - len(ask_defs))))
            random.shuffle(ask_defs)

            for d in ask_defs:
                q = random.choice(YN_TEMPLATES).format(defect=d)
                a = "Yes" if d in present else "No"
                ex = {
                    "image_path": str(img_path.resolve()),
                    "system": SYSTEM_PROMPT,
                    "user": q,
                    "assistant": a,
                    "task": "yesno",
                    "split": args.split,
                    "meta": {"defect": d, "grid": args.grid},
                }
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_written += 1

            # 2) multi-label
            if args.make_multilabel:
                q = random.choice(MULTI_LABEL_TEMPLATES)
                a = "none" if not present else ", ".join(present)
                ex = {
                    "image_path": str(img_path.resolve()),
                    "system": SYSTEM_PROMPT,
                    "user": q,
                    "assistant": a,
                    "task": "multilabel",
                    "split": args.split,
                    "meta": {"present": present, "grid": args.grid},
                }
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_written += 1

            # 3) counting (choose one present defect)
            if args.make_count and present:
                d = random.choice(present)
                q = random.choice(COUNT_TEMPLATES).format(defect=d)
                a = str(len(by_def[d]))
                ex = {
                    "image_path": str(img_path.resolve()),
                    "system": SYSTEM_PROMPT,
                    "user": q,
                    "assistant": a,
                    "task": "count",
                    "split": args.split,
                    "meta": {"defect": d, "count": len(by_def[d]), "grid": args.grid},
                }
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_written += 1

            # 4) grid localization (primary bbox)
            if args.make_grid and present:
                d = random.choice(present)
                primary = choose_primary_box(by_def[d])
                _, xc, yc, w, h = primary
                cell = grid_cell(xc, yc, args.grid)
                q = random.choice(GRID_TEMPLATES).format(defect=d, g=args.grid)
                a = cell
                ex = {
                    "image_path": str(img_path.resolve()),
                    "system": SYSTEM_PROMPT,
                    "user": q,
                    "assistant": a,
                    "task": "grid",
                    "split": args.split,
                    "meta": {"defect": d, "primary_cell": cell, "grid": args.grid},
                }
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_written += 1

            # 5) structured JSON
            if args.make_json:
                defects = []
                for d in present:
                    # total area ratio proxy
                    area = sum(bbox_area(b[3], b[4]) for b in by_def[d])
                    sev = severity_from_area_ratio(area, args.sev_t1, args.sev_t2)
                    primary = choose_primary_box(by_def[d])
                    _, xc, yc, w, h = primary
                    cell = grid_cell(xc, yc, args.grid)
                    defects.append({
                        "type": d,
                        "count": len(by_def[d]),
                        "severity": sev,
                        "primary_cell": cell,
                    })

                # overall condition proxy: worst severity
                sev_rank = {"small": 0, "medium": 1, "large": 2}
                worst = "small"
                for d in defects:
                    if sev_rank[d["severity"]] > sev_rank[worst]:
                        worst = d["severity"]
                overall = {"small": "good", "medium": "fair", "large": "poor"}[worst]

                q = random.choice(JSON_TEMPLATES)
                a = json.dumps({"defects": defects, "overall_condition": overall}, ensure_ascii=False)
                ex = {
                    "image_path": str(img_path.resolve()),
                    "system": SYSTEM_PROMPT,
                    "user": q,
                    "assistant": a,
                    "task": "json",
                    "split": args.split,
                    "meta": {"grid": args.grid, "present": present},
                }
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"Saved: {out_path}  examples={n_written}  images={len(imgs)}")
    print("Tip: generate train from split=train, and eval from split=valid/test to avoid leakage.")


if __name__ == "__main__":
    main()
