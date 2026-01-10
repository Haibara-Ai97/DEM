from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

import multiprocessing as mp

from .datasets import CACHE_INDEX_COLUMNS, ImgCsv

CACHE_INDEX_FILENAME = "index.csv"
CACHE_NPZ_KEYS = ("topv", "topi", "h", "w")


class ClipCollator:
    def __init__(self, processor: CLIPProcessor):
        self.processor = processor

    def __call__(self, batch: list[tuple[str, object]]):
        paths = [x[0] for x in batch]
        images = [x[1] for x in batch]
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        return paths, pixel_values


@torch.no_grad()
def clip_patch_embeds_compat(clip: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    out = clip.vision_model(pixel_values=pixel_values)
    hidden = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
    if hasattr(clip.vision_model, "post_layernorm"):
        hidden = clip.vision_model.post_layernorm(hidden)
    patch = hidden[:, 1:, :]
    patch = clip.visual_projection(patch)
    return F.normalize(patch, dim=-1)


@torch.no_grad()
def clip_text_embeds(
    clip: CLIPModel,
    processor: CLIPProcessor,
    phrases: list[str],
    device: torch.device,
) -> torch.Tensor:
    enc = processor(text=phrases, return_tensors="pt", padding=True, truncation=True).to(device)
    t = clip.get_text_features(**enc)
    return F.normalize(t, dim=-1)


def cache_index_path(out_dir: Path) -> Path:
    return out_dir / CACHE_INDEX_FILENAME


def cache_item_path(out_dir: Path, image_path: str) -> Path:
    return out_dir / f"{Path(image_path).stem}.npz"


def build_clip_cache(
    train_csv: str,
    domain_vocab: str,
    clip_name: str = "openai/clip-vit-base-patch16",
    out_dir: str | Path = "data/stage1_clip_cache",
    batch_size: int = 32,
    num_workers: int = 4,
    topk: int = 10,
    dtype: str = "float16",
    device: torch.device | None = None,
) -> Path:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    mp.set_start_method("spawn", force=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phrases = [l.strip() for l in open(domain_vocab, "r", encoding="utf-8") if l.strip()]
    if not phrases:
        raise ValueError("domain_vocab is empty; provide at least one phrase.")

    clip = CLIPModel.from_pretrained(clip_name).eval().to(device)
    processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
    for p in clip.parameters():
        p.requires_grad = False

    text = clip_text_embeds(clip, processor, phrases, device)

    ds = ImgCsv(train_csv)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ClipCollator(processor),
    )

    index_rows = []
    for paths, pixel_values in tqdm(dl, total=len(dl)):
        pixel_values = pixel_values.to(device)
        patch = clip_patch_embeds_compat(clip, pixel_values)
        _, n_tokens, _ = patch.shape

        sims = torch.einsum("bnd,vd->bnv", patch, text)
        k = min(topk, sims.size(-1))
        topv, topi = torch.topk(sims, k=k, dim=-1)

        side = int(math.sqrt(n_tokens))
        if side * side != n_tokens:
            raise ValueError(f"Expected square CLIP grid, got N={n_tokens}.")

        for b, image_path in enumerate(paths):
            fn = cache_item_path(out_dir, image_path)
            tv = topv[b].detach().cpu().numpy()
            ti = topi[b].detach().cpu().numpy().astype(np.int32)

            if dtype == "float16":
                tv = tv.astype(np.float16)

            np.savez_compressed(fn, topv=tv, topi=ti, h=side, w=side)
            index_rows.append({"image_path": image_path, "cache_path": str(fn)})

    index_path = cache_index_path(out_dir)
    pd.DataFrame(index_rows, columns=CACHE_INDEX_COLUMNS).to_csv(index_path, index=False)
    print(f"Saved cache to: {out_dir}")
    print(f"Index: {index_path}   num_images={len(index_rows)}")
    return index_path
