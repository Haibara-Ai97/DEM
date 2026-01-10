from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

from ..datasets.yolo_dataset import _list_images, _yolo_line_to_xyxy, _IMG_EXTS

def build_coco_gt_from_yolo_split(split_dir: str, class_names: List[str]) -> Dict[str, Any]:
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    image_paths = _list_images(images_dir)

    categories = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]  # 1..K

    images = []
    annotations = []
    ann_id = 1

    for idx, img_path in enumerate(image_paths):
        image_id = 1 + idx
        with Image.open(img_path) as im:
            W, H = im.size
        file_name = os.path.relpath(img_path, os.path.dirname(split_dir)).replace("\\", "/")

        images.append({"id": image_id, "file_name": file_name, "width": W, "height": H})

        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(labels_dir, base + ".txt")
        if not os.path.exists(lbl_path):
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            for ln in f:
                parsed = _yolo_line_to_xyxy(ln, W=W, H=H)
                if parsed is None:
                    continue
                cls, xyxy = parsed
                x1, y1, x2, y2 = xyxy
                w = x2 - x1
                h = y2 - y1
                cat_id = int(cls) + 1
                area = float(w * h)
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "area": area,
                    "iscrowd": 0,
                })
                ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    return coco
