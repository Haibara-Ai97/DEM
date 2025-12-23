import argparse, os, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

import os
import multiprocessing as mp
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ImgCsv(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.paths = df["image_path"].tolist()

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return p, img


class ClipCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        paths = [x[0] for x in batch]
        images = [x[1] for x in batch]
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]
        return paths, pixel_values


@torch.no_grad()
def clip_patch_embeds_compat(clip, pixel_values):
    # 兼容你环境的旧版 transformers（不传 return_dict）
    out = clip.vision_model(pixel_values=pixel_values)
    hidden = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
    if hasattr(clip.vision_model, "post_layernorm"):
        hidden = clip.vision_model.post_layernorm(hidden)
    patch = hidden[:, 1:, :]              # (B,N,Dv)
    patch = clip.visual_projection(patch) # (B,N,De)
    return F.normalize(patch, dim=-1)


@torch.no_grad()
def clip_text_embeds(clip, processor, phrases, device):
    enc = processor(text=phrases, return_tensors="pt", padding=True, truncation=True).to(device)
    t = clip.get_text_features(**enc)
    return F.normalize(t, dim=-1)


def make_collate_fn(processor):
    def collate(batch):
        paths = [x[0] for x in batch]
        images = [x[1] for x in batch]
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
        return paths, pixel_values

    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--domain_vocab", type=str, required=True)
    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch16")
    ap.add_argument("--out_dir", type=str, default="data/stage1_clip_cache")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16","float32"])
    args = ap.parse_args()

    mp.set_start_method("spawn", force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phrases = [l.strip() for l in open(args.domain_vocab, "r", encoding="utf-8") if l.strip()]
    assert len(phrases) > 0

    # load CLIP & CLIPProcessor
    clip = CLIPModel.from_pretrained(args.clip_name).eval().to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_name, use_fast=True)
    for p in clip.parameters():
        p.requires_grad = False

    text = clip_text_embeds(clip, processor, phrases, device)   # (V,De)

    ds = ImgCsv(args.train_csv)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=ClipCollator(processor))

    # 保存一个 index，方便训练阶段读取
    index_rows = []

    for paths, pixel_values in tqdm(dl, total=len(dl)):
        pixel_values = pixel_values.to(device)
        patch = clip_patch_embeds_compat(clip, pixel_values)     # (B,N,De)
        B, N, De = patch.shape

        sims = torch.einsum("bnd,vd->bnv", patch, text)          # (B,N,V)
        K = min(args.topk, sims.size(-1))
        topv, topi = torch.topk(sims, k=K, dim=-1)              # (B,N,K)

        side = int(math.sqrt(N))
        assert side * side == N

        # 写入每张图独立 npz，避免一个大文件过大
        for b in range(B):
            fn = out_dir / (Path(paths[b]).stem + ".npz")
            tv = topv[b].detach().cpu().numpy()
            ti = topi[b].detach().cpu().numpy().astype(np.int32)

            if args.dtype == "float16":
                tv = tv.astype(np.float16)

            np.savez_compressed(fn, topv=tv, topi=ti, h=side, w=side)
            index_rows.append({"image_path": paths[b], "cache_path": str(fn)})

    pd.DataFrame(index_rows).to_csv(out_dir / "index.csv", index=False)
    print(f"Saved cache to: {out_dir}")
    print(f"Index: {out_dir/'index.csv'}   num_images={len(index_rows)}")


if __name__ == "__main__":
    main()
