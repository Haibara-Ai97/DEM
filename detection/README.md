# DEM Vision Encoder — Object Detection Experiment (YOLO-format dataset)

This scaffold runs **standard object detection** experiments using **Torchvision Faster R-CNN**, but your dataset is the
common **YOLO folder format** (as in Roboflow exports):

```
dataset_root/
  classes.txt
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

This code supports:
- **Baseline**: `fasterrcnn_resnet50_fpn`
- **Ours**: ResNet pyramid + **DEM-Encoder** + top-down FPN as detector backbone
- **Ablations**: disable DEM2/DEM3/DEM4/DEM5 per scale

## 0) Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pycocotools tqdm pyyaml
```

## 1) Train

If you have K foreground classes (from `classes.txt`), run:

Baseline:
```bash
python train_det.py --data_root /path/to/dataset_root --model baseline_resnet50_fpn --num_classes K --epochs 24 --batch_size 4
```

DEM-Encoder backbone:
```bash
python train_det.py --data_root /path/to/dataset_root --model dem_resnet50 --num_classes K --epochs 24 --batch_size 4
```

Ablation examples:
```bash
# remove deep color-enhancement branches
python train_det.py --data_root ... --model dem_resnet50 --num_classes K --disable_dem4 --disable_dem5
```

## 2) Evaluate only

```bash
python eval_det.py --data_root /path/to/dataset_root --split valid --ckpt outputs/model_best.pth --model dem_resnet50 --num_classes K
```

## 3) About normalization and Lab

Torchvision detectors normalize images internally (ImageNet mean/std) before passing into the backbone.
For DEM_D (Lab branch), this scaffold **automatically unnormalizes** the tensor inside the DEM backbone to recover
approximate RGB in [0,1] and then computes Lab. This keeps the detector API unchanged.

If you later want to use *true* Lab from raw images (no unnormalize approximation), we can extend the pipeline to pass
Lab explicitly via a custom transform.

## 4) Outputs

- `outputs/last.pth`
- `outputs/model_best.pth`
- `outputs/metrics.jsonl`
