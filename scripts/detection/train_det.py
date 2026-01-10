from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from dem.config_utils import apply_overrides, get_by_path, load_yaml, set_by_path, write_yaml

from apps.detection.datasets.yolo_dataset import YoloFolderDetection, build_transforms, read_classes_txt
from apps.detection.models.detector_factory import (
    build_baseline_fasterrcnn,
    build_dem_fasterrcnn,
    build_swin_fasterrcnn,
    build_convnext_fasterrcnn,
    build_r50_custom_fasterrcnn,
)
from apps.detection.utils.engine import train_one_epoch, evaluate
from apps.detection.utils.misc import set_seed, collate_fn, save_checkpoint
from apps.detection.utils.yolo_gt_to_coco import build_coco_gt_from_yolo_split
from apps.detection.utils.misc import load_checkpoint

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "apps" / "detection" / "configs" / "default.yaml"

def parse_args(argv=None):
    p = argparse.ArgumentParser("Detection Experiment (YOLO folder dataset): Faster R-CNN baseline vs DEM-Encoder backbone")

    p.add_argument("--data_root", type=str, default=None, help="Dataset root containing train/valid/test and classes.txt")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pth) to resume from, e.g. outputs/last.pth")

    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--set", action="append", default=[], help="Override config values with key=value")

    return p.parse_args(argv)

def resolve_config_path(path: str | None) -> str | None:
    if not path:
        return None
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = Path(__file__).resolve().parents[2] / "apps" / "detection" / path
    if candidate.exists():
        return str(candidate)
    return path

def main(argv=None):
    args = parse_args(argv)
    cfg_path = resolve_config_path(args.config)
    cfg = load_yaml(cfg_path) if cfg_path and os.path.exists(cfg_path) else {}
    apply_overrides(cfg, args.set)

    if args.data_root:
        set_by_path(cfg, "data.root", args.data_root)
    if args.output_dir:
        set_by_path(cfg, "output.dir", args.output_dir)
    if args.device:
        set_by_path(cfg, "runtime.device", args.device)
    if args.resume:
        set_by_path(cfg, "resume.path", args.resume)

    output_dir = get_by_path(cfg, "output.dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    write_yaml(Path(output_dir) / "config.yaml", cfg)
    set_seed(int(get_by_path(cfg, "training.seed", 42)))

    device_name = get_by_path(cfg, "runtime.device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    data_root = get_by_path(cfg, "data.root")
    if not data_root:
        raise ValueError("Config must set data.root or pass --data_root.")

    classes_path = os.path.join(data_root, get_by_path(cfg, "data.classes_txt", "classes.txt"))
    class_names = read_classes_txt(classes_path)
    num_classes_cfg = get_by_path(cfg, "data.num_classes")
    num_classes_fg = int(num_classes_cfg) if num_classes_cfg is not None else len(class_names)
    if num_classes_fg <= 0:
        raise ValueError("num_classes must be > 0 (or classes.txt must contain class names).")

    train_dir = os.path.join(data_root, get_by_path(cfg, "data.split_train", "train"))
    val_dir = os.path.join(data_root, get_by_path(cfg, "data.split_val", "valid"))

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
        train_ds, batch_size=int(get_by_path(cfg, "training.batch_size", 4)), shuffle=True,
        num_workers=int(get_by_path(cfg, "training.num_workers", 4)), pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, int(get_by_path(cfg, "training.num_workers", 4)) // 2), pin_memory=True,
        collate_fn=collate_fn
    )

    anchor_sizes = list(get_by_path(cfg, "model.anchor_sizes", [16, 32, 64, 128]))
    aspect_ratios = list(get_by_path(cfg, "model.aspect_ratios", [0.5, 1.0, 2.0]))

    model_name = get_by_path(cfg, "model.name", "r50_custom_fpn")
    if model_name == "baseline_resnet50_fpn":
        model = build_baseline_fasterrcnn(num_classes_fg)

    elif model_name == "r50_custom_fpn":
        model = build_r50_custom_fasterrcnn(
            num_classes_fg,
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
        )

    elif model_name == "convnext_tiny_fpn":
        model = build_convnext_fasterrcnn(
            num_classes_fg,
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            convnext_name="convnext_tiny",
            pretrained_convnext=True,
        )

    elif model_name == "swin_tiny_fpn":
        model = build_swin_fasterrcnn(
            num_classes_fg,
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            swin_name="swin_t",
            pretrained_swin=True,
        )

    elif model_name == "dem_resnet50":
        model = build_dem_fasterrcnn(
            num_classes_fg,
            dem_C=int(get_by_path(cfg, "model.dem.C", 256)),
            init_gamma=float(get_by_path(cfg, "model.dem.init_gamma", 0.5)),
            lf_kernel=int(get_by_path(cfg, "model.dem.lf_kernel", 7)),
            anchor_sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
            disable_dem2=bool(get_by_path(cfg, "model.dem.disable_dem2", False)),
            disable_dem3=bool(get_by_path(cfg, "model.dem.disable_dem3", False)),
            disable_dem4=bool(get_by_path(cfg, "model.dem.disable_dem4", False)),
            disable_dem5=bool(get_by_path(cfg, "model.dem.disable_dem5", False)),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=float(get_by_path(cfg, "training.lr", 0.005)),
        momentum=0.9,
        weight_decay=float(get_by_path(cfg, "training.weight_decay", 0.0005)),
    )

    lr_scheduler_main = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(get_by_path(cfg, "training.lr_step_milestones", [16, 22])),
        gamma=float(get_by_path(cfg, "training.lr_gamma", 0.1)),
    )

    warmup_iters = int(get_by_path(cfg, "training.warmup_iters", 500))
    def warmup_lambda(it):
        if it >= warmup_iters:
            return 1.0
        return float(it + 1) / float(warmup_iters)
    lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Build COCO GT dict once for evaluation (from YOLO val split)
    coco_gt_val = build_coco_gt_from_yolo_split(val_dir, class_names=class_names[:num_classes_fg])

    best_map = -1.0
    metrics_path = os.path.join(output_dir, "metrics.jsonl")

    start_epoch = 0
    resume_path = get_by_path(cfg, "resume.path", "")
    if resume_path:
        model_to_load = model
        ckpt = load_checkpoint(
            resume_path,
            model_to_load,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler_main,
            map_location="cpu",
        )
        start_epoch = int(ckpt.get("epoch", -1))+1
        best_map = float(ckpt.get("best_map", -1.0))

        print(f"[Resume] Loaded {resume_path}")
        print(f"[Resume] start epoch: {start_epoch}, best_map: {best_map:.4}")

    for epoch in range(start_epoch, int(get_by_path(cfg, "training.epochs", 24))):
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
            "model": model_name,
            "num_classes_fg": int(num_classes_fg),
            "disable_dem": {
                "dem2": bool(get_by_path(cfg, "model.dem.disable_dem2", False)),
                "dem3": bool(get_by_path(cfg, "model.dem.disable_dem3", False)),
                "dem4": bool(get_by_path(cfg, "model.dem.disable_dem4", False)),
                "dem5": bool(get_by_path(cfg, "model.dem.disable_dem5", False)),
            },
            "cfg": cfg,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # checkpoints
        save_checkpoint(os.path.join(output_dir, "last.pth"), model, optimizer, lr_scheduler_main, epoch, best_map)
        if val_stats.get("mAP", 0.0) > best_map:
            best_map = float(val_stats["mAP"])
            save_checkpoint(os.path.join(output_dir, "model_best.pth"), model, optimizer, lr_scheduler_main, epoch, best_map)

        print(f"[Epoch {epoch}] train_loss={train_stats['loss']:.4f}  val_mAP={val_stats.get('mAP', 0.0):.4f}  best={best_map:.4f}")

if __name__ == "__main__":
    main()
