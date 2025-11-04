# DEM Vision Encoder — Encoder-only 目标检测对比实验（Torchvision Faster R-CNN）

本项目用于在**固定检测器框架（Faster R-CNN）**的前提下，进行 **Encoder-only** 对比实验：
仅替换 Encoder（ResNet / ConvNeXt / Swin / DEM-Encoder），其余检测器组件（RPN/ROI/anchors/roi_align/训练策略）保持一致，从而将性能差异尽量归因到编码器表征能力。

## 1. 目录结构

推荐目录结构如下（与你当前 YOLO/Roboflow 数据导出一致）：

```
project/
  train_det.py
  eval_det.py
  configs/default.yaml
  datasets/
  models/
  utils/
  outputs/

  dataset/
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

### 关键输出

训练会在 `outputs/` 下生成：

* `outputs/last.pth`：最后一次保存的 checkpoint（建议用于续训）
* `outputs/model_best.pth`：验证集 mAP 最优的 checkpoint（论文报告建议用这个）
* `outputs/metrics.jsonl`：逐 epoch 记录的训练/验证指标（你已在用）

---

## 2. 环境安装

### 2.1 requirements.txt（示例）

如果你按 CUDA 12.1（cu121）安装：

```txt
--extra-index-url https://download.pytorch.org/whl/cu121
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
pillow>=10.0.0
pyyaml>=6.0.1
tqdm>=4.66.0
pycocotools>=2.0.7
```

安装：

```bash
pip install -r requirements.txt
```

> 评估依赖 `pycocotools` 必须安装，否则只能训练不能跑 COCO mAP。

---

## 3. 数据集要求与检查

### 3.1 labels 格式（YOLO）

每张图像对应 `labels/*.txt`，每行：

```
class_id  x_center  y_center  width  height
```

其中坐标均为归一化到 [0,1]。

### 3.2 classes.txt

`classes.txt` 每行一个类别名（共 K 行），脚本将自动推断 `K`。

建议快速检查类别数是否一致：

```bash
wc -l dataset/classes.txt
```

---

## 4. Encoder-only 对比：可用的 `--model` 选项

本项目建议使用这些 model 名称（与你前面改造的代码一致）：

* `r50_custom_fpn`
  **推荐作为 Encoder-only baseline**：ResNet50 金字塔 +（统一实现）projection + FPN + Faster R-CNN
* `convnext_tiny_fpn`
  ConvNeXt-Tiny 金字塔 +（同一套）projection + FPN + Faster R-CNN
* `swin_tiny_fpn`
  Swin-Tiny 金字塔 +（同一套）projection + FPN + Faster R-CNN
* `dem_resnet50`
  ResNet 金字塔 + DEM-Encoder（按尺度增强）+ Faster R-CNN
* `baseline_resnet50_fpn`
  Torchvision 官方 `fasterrcnn_resnet50_fpn`（不推荐用于严格 encoder-only，因为其内部 FPN/anchors 与我们自写实现不同）

> 严格控制变量对比时，请优先使用 `r50_custom_fpn` 作为 baseline，而不是 `baseline_resnet50_fpn`。

---

## 5. 训练命令（建议直接复制）

假设你的数据根目录是 `./dataset`。

### 5.1 ResNet（Encoder-only Baseline）

```bash
python train_det.py \
  --data_root /path/to/dataset \
  --model r50_custom_fpn \
  --epochs 24 \
  --batch_size 4 \
  --output_dir outputs/r50_custom_fpn
```

### 5.2 ConvNeXt-Tiny（Encoder-only）

```bash
python train_det.py \
  --data_root /path/to/dataset \
  --model convnext_tiny_fpn \
  --epochs 24 \
  --batch_size 4 \
  --output_dir outputs/convnext_tiny_fpn
```

### 5.3 Swin-Tiny（Encoder-only）

```bash
python train_det.py \
  --data_root /path/to/dataset \
  --model swin_tiny_fpn \
  --epochs 24 \
  --batch_size 4 \
  --output_dir outputs/swin_tiny_fpn
```

### 5.4 DEM-Encoder

```bash
python train_det.py \
  --data_root /path/to/dataset \
  --model dem_resnet50 \
  --epochs 24 \
  --batch_size 4 \
  --output_dir outputs/dem_resnet50
```

#### DEM 消融（可选）

* 去掉深层颜色增强（DEM4/DEM5）：

```bash
python train_det.py --data_root /path/to/dataset --model dem_resnet50 --disable_dem4 --disable_dem5
```

---

## 6. 评估命令（只评估）

对 `valid` 集评估：

```bash
python eval_det.py \
  --data_root /path/to/dataset \
  --split valid \
  --model dem_resnet50 \
  --ckpt outputs/model_best.pth
```

评估输出：COCO bbox

* `mAP`（AP@[.50:.95]）
* `mAP50`（AP@0.50）
* `mAP75`（AP@0.75）

---

## 7. 对比实验的“控制变量规范”

为保证 encoder-only 对比公平性，请务必统一以下设置：

1. 数据划分：同一 train/valid/test
2. 输入尺寸与增强：同一套 transform（不要对某个 encoder 单独增强）
3. 优化器与学习率策略：同一 lr / warmup / milestones
4. anchors 与 ROIAlign：同一 `anchor_sizes`、`aspect_ratios`
5. epoch 数与 seed：建议固定最大 epoch（如 24），并使用相同 seed；更严格可跑 3 个 seed 取均值±方差
6. 最终报告：用 `model_best.pth` 对应的 best mAP（而不是最后一轮）

---

## 8. 常用参数调整

### 8.1 修改 anchors（强烈建议针对裂缝/小目标尝试）

在 `configs/default.yaml` 修改：

```yaml
anchor_sizes: [8, 16, 32, 64]
aspect_ratios: [0.5, 1.0, 2.0]
```

如果裂缝极细长，可加更极端比例（注意 FP 也可能增加）：

```yaml
aspect_ratios: [0.2, 0.5, 1.0, 2.0, 5.0]
```
...

---

## 9. 续训（从 .pth 继续训练）

### 9.1 续训命令示例

例如你跑完 24 epoch，想继续到 36：

```bash
python train_det.py \
  --data_root ./dataset \
  --model dem_resnet50 \
  --epochs 36 \
  --resume outputs/last.pth
```

> 建议用 `last.pth` 续训（包含 optimizer 与 lr_scheduler 状态）。
> 若你用 `model_best.pth`，通常更像“从最好点继续微调”，也可以，但可能不如 last 稳定。
