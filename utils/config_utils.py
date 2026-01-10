from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: str | Path, cfg: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def set_by_path(cfg: dict, key_path: str, value: Any) -> None:
    parts = key_path.split(".")
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def get_by_path(cfg: dict, key_path: str, default: Any = None) -> Any:
    cur = cfg
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        set_by_path(cfg, key, value)
