# DEM

## 迁移说明

- 模型结构代码统一整理到主目录 `models/`，按 `dem_encoder/`、`da_adapter/`、`vlm_baselines/` 划分。
- 训练 / 微调 / 验证 / 数据脚本统一整理到主目录 `scripts/`。

## 检测子项目入口与关系说明

- 检测子项目位于 `apps/detection/`，统一通过入口 `python -m apps.detection.main {train|eval} ...` 运行。
- 检测与 VLM/Adapter 共享的核心视觉组件集中在 `models/`，避免重复实现。

## 基线实验入口

- VLM Baselines 统一入口：`python -m scripts.vlm_baselines.train` 与 `python -m scripts.vlm_baselines.eval`。
- 支持的模型列表（与 `doc/vlm_baselines.md` 保持一致）：
  1. `Qwen/Qwen2.5-VL-7B-Instruct` (family: `qwen2_5_vl`)
  2. `Qwen/Qwen2-VL-7B-Instruct` (family: `qwen2_vl`)
  3. `llava-hf/llava-1.5-7b-hf` (family: `llava_1_5`)
  4. `HuggingFaceM4/idefics2-8b` (family: `idefics2`)
  5. `microsoft/Phi-3.5-vision-instruct` (family: `phi3v`)
