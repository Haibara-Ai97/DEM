# DEM

## 迁移说明

- 以 `dem/` 作为唯一权威实现目录，模型实现统一迁移至 `dem/models/`。
- 根目录的 `da_adapter.py` 与 `dem_encoder.py` 现在仅做兼容导出，建议后续直接从 `dem.models` 引用。

## 检测子项目入口与关系说明

- 检测子项目位于 `apps/detection/`，统一通过入口 `python -m apps.detection.main {train|eval} ...` 运行。
- 检测与 VLM/Adapter 共享的核心视觉组件（如 `DEMEncoder`）已集中在 `dem/models/`，避免重复实现；检测侧通过 `dem.models` 引用这些模块。
