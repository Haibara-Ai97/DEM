# DEM

## 迁移说明

- 以 `dem/` 作为唯一权威实现目录，模型实现统一迁移至 `dem/models/`。
- 根目录的 `da_adapter.py` 与 `dem_encoder.py` 现在仅做兼容导出，建议后续直接从 `dem.models` 引用。

## 检测子项目入口与关系说明

- 检测子项目位于 `apps/detection/`，统一通过入口 `python -m apps.detection.main {train|eval} ...` 运行。
- 检测与 VLM/Adapter 共享的核心视觉组件（如 `DEMEncoder`）已集中在 `dem/models/`，避免重复实现；检测侧通过 `dem.models` 引用这些模块。

## 基线实验入口

- VLM Baselines 统一入口：`python -m dem.vlm_baselines.train` 与 `python -m dem.vlm_baselines.eval`。
- 支持的模型列表（与 `Adapter/README_vlm_baselines.md` 保持一致）：
  1. `Qwen/Qwen2.5-VL-7B-Instruct` (family: `qwen2_5_vl`)
  2. `Qwen/Qwen2-VL-7B-Instruct` (family: `qwen2_vl`)
  3. `llava-hf/llava-1.5-7b-hf` (family: `llava_1_5`)
  4. `HuggingFaceM4/idefics2-8b` (family: `idefics2`)
  5. `microsoft/Phi-3.5-vision-instruct` (family: `phi3v`)
