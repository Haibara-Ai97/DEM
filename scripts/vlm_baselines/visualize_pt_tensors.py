#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize .pt tensors as PNG heatmaps.

This script is intentionally standalone and does not modify inference scripts.
It can visualize common artifacts such as:
- vision_feature_map_tmean.pt
- vision_last_layer_attention.pt
- vision_attentions.pt
- any other tensor / tuple[list[tensor]] saved by torch.save

Examples:
  python -m scripts.vlm_baselines.visualize_pt_tensors \
    --input /path/to/vision_feature_map_tmean.pt

  python -m scripts.vlm_baselines.visualize_pt_tensors \
    --input /path/to/visual_artifacts_dir \
    --grid_thw /path/to/visual_artifacts_dir/image_grid_thw.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize saved PyTorch .pt tensors as PNG images.")
    ap.add_argument("--input", type=str, required=True, help="A .pt file path or a directory containing .pt files.")
    ap.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for PNG files. Defaults to '<input_parent>/visualized_png'.",
    )
    ap.add_argument(
        "--grid_thw",
        type=str,
        default="",
        help="Optional path to image_grid_thw.pt for attention spatial projection.",
    )
    return ap.parse_args()


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)


def _save_gray(arr_2d: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(arr_2d), mode="L").save(out_path)


def _collapse_to_2d(t: torch.Tensor) -> np.ndarray:
    x = t.detach().cpu().float()
    while x.ndim > 2:
        x = x.mean(dim=0)
    if x.ndim != 2:
        raise ValueError(f"Cannot collapse tensor with shape {tuple(t.shape)} to 2D.")
    return x.numpy()


def _load_optional_grid(grid_path: str) -> tuple[int, int, int] | None:
    if not grid_path:
        return None
    p = Path(grid_path)
    if not p.exists():
        raise FileNotFoundError(f"grid_thw file not found: {grid_path}")
    grid = torch.load(p, map_location="cpu")
    if not torch.is_tensor(grid):
        return None
    g = grid[0] if grid.ndim == 2 else grid
    if g.numel() != 3:
        return None
    t, h, w = [int(v) for v in g.tolist()]
    return t, h, w


def _iter_pt_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        if input_path.suffix != ".pt":
            raise ValueError(f"Input file must be .pt: {input_path}")
        yield input_path
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    for fp in sorted(input_path.glob("*.pt")):
        yield fp


def _visualize_attention_tensor(t: torch.Tensor, base: Path, grid_thw: tuple[int, int, int] | None) -> None:
    # Common shape: (B, heads, Q, K)
    if t.ndim != 4:
        _save_gray(_collapse_to_2d(t), base.with_suffix(".png"))
        return

    attn_map = t[0].mean(dim=0).detach().cpu().float().numpy()  # (Q, K)
    _save_gray(attn_map, base.parent / f"{base.stem}_matrix.png")

    if grid_thw is None:
        return

    tt, hh, ww = grid_thw
    token_count = int(tt * hh * ww)
    if attn_map.shape[-1] != token_count:
        return

    key_importance = attn_map.mean(axis=0).reshape(tt, hh, ww).mean(axis=0)
    _save_gray(key_importance, base.parent / f"{base.stem}_spatial.png")


def _visualize_loaded_obj(obj: object, base: Path, grid_thw: tuple[int, int, int] | None) -> int:
    saved = 0

    if torch.is_tensor(obj):
        if "attention" in base.stem and obj.ndim >= 4:
            _visualize_attention_tensor(obj, base, grid_thw)
        else:
            _save_gray(_collapse_to_2d(obj), base.with_suffix(".png"))
        return 1

    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            if not torch.is_tensor(item):
                continue
            sub_base = base.parent / f"{base.stem}_{i:03d}"
            if "attention" in base.stem and item.ndim >= 4:
                _visualize_attention_tensor(item, sub_base, grid_thw)
            else:
                _save_gray(_collapse_to_2d(item), sub_base.with_suffix(".png"))
            saved += 1
        return saved

    return 0


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "visualized_png"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_thw = _load_optional_grid(args.grid_thw)

    total = 0
    for pt_file in _iter_pt_files(input_path):
        obj = torch.load(pt_file, map_location="cpu")
        base = out_dir / pt_file.stem
        saved = _visualize_loaded_obj(obj, base, grid_thw)
        if saved == 0:
            print(f"[skip] unsupported object in {pt_file}")
            continue
        total += saved
        print(f"[ok] {pt_file.name} -> {saved} image(s)")

    print(f"[done] saved {total} image(s) to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
