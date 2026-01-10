# DEM

## 迁移说明

- 以 `dem/` 作为唯一权威实现目录，模型实现统一迁移至 `dem/models/`。
- 根目录的 `da_adapter.py` 与 `dem_encoder.py` 现在仅做兼容导出，建议后续直接从 `dem.models` 引用。
