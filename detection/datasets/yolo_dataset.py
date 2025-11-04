from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import Compose, ToTensor, RandomHorizontalFlip

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def read_classes_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f.readlines()]
    return [n for n in names if n]

def _list_images(images_dir: str) -> List[str]:
    files = []
    for fn in os.listdir(images_dir):
        if fn.lower().endswith(_IMG_EXTS):
            files.append(os.path.join(images_dir, fn))
    files.sort()
    return files

def _image_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return w, h

def _yolo_line_to_xyxy(line: str, W: int, H: int):
    # YOLO: cls xc yc w h (all normalized [0,1])
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    xc = float(parts[1]); yc = float(parts[2])
    bw = float(parts[3]); bh = float(parts[4])

    x1 = (xc - bw / 2.0) * W
    y1 = (yc - bh / 2.0) * H
    x2 = (xc + bw / 2.0) * W
    y2 = (yc + bh / 2.0) * H

    # clamp
    x1 = max(0.0, min(x1, W - 1.0))
    y1 = max(0.0, min(y1, H - 1.0))
    x2 = max(0.0, min(x2, W - 1.0))
    y2 = max(0.0, min(y2, H - 1.0))
    if x2 <= x1 or y2 <= y1:
        return None
    return cls, [x1, y1, x2, y2]

class YoloFolderDetection(Dataset):
    """YOLO folder-format dataset.

    Expected split dir:
      split_dir/images/*.jpg|png
      split_dir/labels/*.txt  (same basename)

    Label file format per line:
      class_id xc yc w h   (normalized)

    Returned target matches Torchvision detection API:
      boxes: FloatTensor [N,4] xyxy (pixel)
      labels: Int64Tensor [N] where labels are **1..K** (0 is background)
      image_id: Int64Tensor([id])
      area, iscrowd
    """

    def __init__(
        self,
        split_dir: str,
        transforms=None,
        num_classes_fg: Optional[int] = None,
        filter_empty: bool = False,
        image_id_base: int = 1,
    ) -> None:
        super().__init__()
        self.split_dir = split_dir
        self.images_dir = os.path.join(split_dir, "images")
        self.labels_dir = os.path.join(split_dir, "labels")
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Missing images dir: {self.images_dir}")
        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(f"Missing labels dir: {self.labels_dir}")

        self.image_paths = _list_images(self.images_dir)
        self.transforms = transforms
        self.num_classes_fg = num_classes_fg
        self.filter_empty = bool(filter_empty)
        self.image_id_base = int(image_id_base)

        if self.filter_empty:
            kept = []
            for p in self.image_paths:
                lbl = self._label_path(p)
                if os.path.exists(lbl) and os.path.getsize(lbl) > 0:
                    kept.append(p)
            self.image_paths = kept

    def _label_path(self, img_path: str) -> str:
        base = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.labels_dir, base + ".txt")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            W, H = im.size

        lbl_path = self._label_path(img_path)
        boxes = []
        labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for ln in f:
                    parsed = _yolo_line_to_xyxy(ln, W=W, H=H)
                    if parsed is None:
                        continue
                    cls, xyxy = parsed
                    # Torchvision uses 1..K for foreground (0 is background)
                    lab = cls + 1
                    if self.num_classes_fg is not None:
                        if lab < 1 or lab > int(self.num_classes_fg):
                            continue
                    boxes.append(xyxy)
                    labels.append(lab)

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        areas = ((boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])) if boxes_t.numel() else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((boxes_t.shape[0],), dtype=torch.int64)

        target: Dict[str, Any] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([self.image_id_base + idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }

        # load image tensor after reading size
        with Image.open(img_path) as im2:
            im2 = im2.convert("RGB")
            image = im2

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

def build_transforms(train: bool):
    t = [ToTensor()]
    if train:
        t.append(RandomHorizontalFlip(0.5))
    return Compose(t)
