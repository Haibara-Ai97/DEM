from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from datasets.yolo_dataset import YoloFolderDetection, build_transforms, read_classes_txt
from models.detector_factory import build_baseline_fasterrcnn, build_dem_fasterrcnn, build_convnext_fasterrcnn, build_swin_fasterrcnn, build_r50_custom_fasterrcnn
from utils.engine import evaluate
from utils.misc import collate_fn, load_checkpoint
from utils.yolo_gt_to_coco import build_coco_gt_from_yolo_split

def parse_args():
    p = argparse.ArgumentParser("Evaluate Faster R-CNN on YOLO folder dataset (COCO metrics via generated GT dict)")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split", type=str, default="valid")
    p.add_argument("--classes_txt", type=str, default="classes.txt")
    p.add_argument("--ckpt", type=str, required=True)

    p.add_argument(
        "--model",
        type=str,
        default="r50_custom_fpn",
        choices=[
            "r50_custom_fpn",
            "convnext_tiny_fpn",
            "swin_tiny_fpn",
            "dem_resnet50",
            "baseline_resnet50_fpn",
        ],
    )

    p.add_argument("--num_classes", type=int, default=None)

    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes_path = os.path.join(args.data_root, args.classes_txt)
    class_names = read_classes_txt(classes_path)
    num_classes_fg = args.num_classes if args.num_classes is not None else len(class_names)

    split_dir = os.path.join(args.data_root, args.split)
    ds = YoloFolderDetection(split_dir=split_dir, transforms=build_transforms(train=False),
                             num_classes_fg=num_classes_fg, filter_empty=False, image_id_base=1)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)

    anchor_sizes = [16, 32, 64, 128]
    aspect_ratios = [0.5, 1.0, 2.0]

    if args.model == "baseline_resnet50_fpn":
        model = build_baseline_fasterrcnn(num_classes_fg)

    elif args.model == "r50_custom_fpn":
        model = build_r50_custom_fasterrcnn(num_classes_fg, anchor_sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    elif args.model == "convnext_tiny_fpn":
        model = build_convnext_fasterrcnn(num_classes_fg, anchor_sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    elif args.model == "swin_tiny_fpn":
        model = build_swin_fasterrcnn(num_classes_fg, anchor_sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    elif args.model == "dem_resnet50":
        model = build_dem_fasterrcnn(num_classes_fg, anchor_sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.to(device)
    load_checkpoint(args.ckpt, model, map_location="cpu")

    coco_gt = build_coco_gt_from_yolo_split(split_dir, class_names=class_names[:num_classes_fg])
    stats = evaluate(model, loader, device, ann_dict=coco_gt)
    print("COCO Eval:", stats)

if __name__ == "__main__":
    main()
