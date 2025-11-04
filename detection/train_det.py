from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import yaml

from datasets.yolo_dataset import YoloFolderDetection, build_transforms, read_classes_txt
from models.detector_factory import build_baseline_fasterrcnn, build_dem_fasterrcnn, build_swin_fasterrcnn, build_convnext_fasterrcnn, build_r50_custom_fasterrcnn
from utils.engine import train_one_epoch, evaluate
from utils.misc import set_seed, collate_fn, save_checkpoint
from utils.yolo_gt_to_coco import build_coco_gt_from_yolo_split
from utils.misc import load_checkpoint

def parse_args():
    p = argparse.ArgumentParser("Detection Experiment (YOLO folder dataset): Faster R-CNN baseline vs DEM-Encoder backbone")

    p.add_argument("--data_root", type=str, required=True, help="Dataset root containing train/valid/test and classes.txt")
    p.add_argument("--split_train", type=str, default="train")
    p.add_argument("--split_val", type=str, default="valid")
    p.add_argument("--classes_txt", type=str, default="classes.txt")

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

    p.add_argument("--num_classes", type=int, default=None, help="Foreground classes K. If omitted, inferred from classes.txt")

    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--output_dir", type=str, default="outputs")

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)

    # DEM args
    p.add_argument("--dem_C", type=int, default=None)
    p.add_argument("--dem_init_gamma", type=float, default=None)
    p.add_argument("--dem_lf_kernel", type=int, default=None)

    p.add_argument("--disable_dem2", action="store_true")
    p.add_argument("--disable_dem3", action="store_true")
    p.add_argument("--disable_dem4", action="store_true")
    p.add_argument("--disable_dem5", action="store_true")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pth) to resume from, e.g. outputs/last.pth")

    return p.parse_args()

def merge_cfg(args):
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    def pick(name, default=None):
        v = getattr(args, name)
        if v is None:
            return cfg.get(name, default)
        return v

    merged = {
        "seed": pick("seed", 42),
        "epochs": pick("epochs", 24),
        "batch_size": pick("batch_size", 4),
        "num_workers": pick("num_workers", 4),
        "lr": pick("lr", 0.005),
        "weight_decay": pick("weight_decay", 0.0005),
        "warmup_iters": cfg.get("warmup_iters", 500),
        "lr_step_milestones": cfg.get("lr_step_milestones", [16, 22]),
        "lr_gamma": cfg.get("lr_gamma", 0.1),
        "dem_C": pick("dem_C", 256),
        "dem_init_gamma": pick("dem_init_gamma", 0.5),
        "dem_lf_kernel": pick("dem_lf_kernel", 7),
        "anchor_sizes": cfg.get("anchor_sizes", [16, 32, 64, 128]),
        "aspect_ratios": cfg.get("aspect_ratios", [0.5, 1.0, 2.0]),
    }
    return merged

def main():
    args = parse_args()
    cfg = merge_cfg(args)

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(int(args.seed if args.seed is not None else cfg["seed"]))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    classes_path = os.path.join(args.data_root, args.classes_txt)
    class_names = read_classes_txt(classes_path)
    num_classes_fg = args.num_classes if args.num_classes is not None else len(class_names)
    if num_classes_fg <= 0:
        raise ValueError("num_classes must be > 0 (or classes.txt must contain class names).")

    train_dir = os.path.join(args.data_root, args.split_train)
    val_dir = os.path.join(args.data_root, args.split_val)

    train_ds = YoloFolderDetection(
        split_dir=train_dir,
        transforms=build_transforms(train=True),
        num_classes_fg=num_classes_fg,
        filter_empty=True,
        image_id_base=1,
    )
    val_ds = YoloFolderDetection(
        split_dir=val_dir,
        transforms=build_transforms(train=False),
        num_classes_fg=num_classes_fg,
        filter_empty=False,
        image_id_base=1,
    )

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=int(cfg["num_workers"]), pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, int(cfg["num_workers"]) // 2), pin_memory=True,
        collate_fn=collate_fn
    )

    anchor_sizes = list(cfg["anchor_sizes"])
    aspect_ratios = list(cfg["aspect_ratios"])

    if args.model == "baseline_resnet50_fpn":
        model = build_baseline_fasterrcnn(num_classes_fg)

    elif args.model == "r50_custom_fpn":
        model = build_r50_custom_fasterrcnn(
            num_classes_fg,
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
        )

    elif args.model == "convnext_tiny_fpn":
        model = build_convnext_fasterrcnn(
            num_classes_fg,
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            convnext_name="convnext_tiny",
            pretrained_convnext=True,
        )

    elif args.model == "swin_tiny_fpn":
        model = build_swin_fasterrcnn(
            num_classes_fg,
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            swin_name="swin_t",
            pretrained_swin=True,
        )

    elif args.model == "dem_resnet50":
        model = build_dem_fasterrcnn(
            num_classes_fg,
            dem_C=int(cfg["dem_C"]),
            init_gamma=float(cfg["dem_init_gamma"]),
            lf_kernel=int(cfg["dem_lf_kernel"]),
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            disable_dem2=args.disable_dem2,
            disable_dem3=args.disable_dem3,
            disable_dem4=args.disable_dem4,
            disable_dem5=args.disable_dem5,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=float(cfg["lr"]), momentum=0.9, weight_decay=float(cfg["weight_decay"]))

    lr_scheduler_main = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg["lr_step_milestones"]), gamma=float(cfg["lr_gamma"])
    )

    warmup_iters = int(cfg["warmup_iters"])
    def warmup_lambda(it):
        if it >= warmup_iters:
            return 1.0
        return float(it + 1) / float(warmup_iters)
    lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Build COCO GT dict once for evaluation (from YOLO val split)
    coco_gt_val = build_coco_gt_from_yolo_split(val_dir, class_names=class_names[:num_classes_fg])

    best_map = -1.0
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    start_epoch = 0
    if args.resume:
        model_to_load = model
        ckpt = load_checkpoint(
            args.resume,
            model_to_load,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler_main,
            map_location="cpu",
        )
        start_epoch = int(ckpt.get("epoch", -1))+1
        best_map = float(ckpt.get("best_map", -1.0))

        print(f"[Resume] Loaded {args.resume}")
        print(f"[Resume] start epoch: {start_epoch}, best_map: {best_map:.4}")

    for epoch in range(start_epoch, int(cfg["epochs"])):
        lr_sched = lr_scheduler_warmup if (epoch == 0 and warmup_iters > 0) else None

        train_stats = train_one_epoch(model, optimizer, train_loader, device, epoch, lr_scheduler=lr_sched)
        lr_scheduler_main.step()

        val_stats = evaluate(model, val_loader, device, ann_dict=coco_gt_val)

        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "lr": optimizer.param_groups[0]["lr"],
            "model": args.model,
            "num_classes_fg": int(num_classes_fg),
            "disable_dem": {"dem2": bool(args.disable_dem2), "dem3": bool(args.disable_dem3), "dem4": bool(args.disable_dem4), "dem5": bool(args.disable_dem5)},
            "cfg": cfg,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # checkpoints
        save_checkpoint(os.path.join(args.output_dir, "last.pth"), model, optimizer, lr_scheduler_main, epoch, best_map)
        if val_stats.get("mAP", 0.0) > best_map:
            best_map = float(val_stats["mAP"])
            save_checkpoint(os.path.join(args.output_dir, "model_best.pth"), model, optimizer, lr_scheduler_main, epoch, best_map)

        print(f"[Epoch {epoch}] train_loss={train_stats['loss']:.4f}  val_mAP={val_stats.get('mAP', 0.0):.4f}  best={best_map:.4f}")

if __name__ == "__main__":
    main()
