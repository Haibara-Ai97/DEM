# scripts/train_stage1_adapter_alignment_cached.py
from __future__ import annotations
import argparse, os, math, random
from pathlib import Path
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.backbone import SimplePyramidBackbone
from models.da_adapter import DAAdapter, DAAdapterConfig
from models.dem_encoder import DEMEncoderConfig, DEMVisionBackbone


# -------------------------
# Dataset: index.csv -> (image_path, cache_path)
# -------------------------
class CacheIndexDataset(Dataset):
    def __init__(self, index_csv: str):
        import pandas as pd
        df = pd.read_csv(index_csv)
        assert "image_path" in df.columns and "cache_path" in df.columns
        self.image_paths = df["image_path"].tolist()
        self.cache_paths = df["cache_path"].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.image_paths[idx], img, self.cache_paths[idx]


def pil_resize_square(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), resample=Image.BICUBIC)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def normalize(t: torch.Tensor, mean: Tuple[float,float,float], std: Tuple[float,float,float]) -> torch.Tensor:
    m = torch.tensor(mean, dtype=t.dtype).view(3,1,1)
    s = torch.tensor(std, dtype=t.dtype).view(3,1,1)
    return (t - m) / s


def collate_cached(batch, enc_cfg: DEMEncoderConfig, image_size: int):
    paths, imgs, cache_paths = zip(*batch)

    enc_imgs = [pil_resize_square(im, image_size) for im in imgs]
    enc_x = torch.stack([normalize(pil_to_tensor(im), enc_cfg.image_mean, enc_cfg.image_std) for im in enc_imgs], dim=0)

    # 读取 cache（npz）——每张图一个文件
    topi_list, topv_list, hw_list = [], [], []
    for cp in cache_paths:
        z = np.load(cp)
        topi_list.append(torch.from_numpy(z["topi"]).long())      # (N,K)
        topv_list.append(torch.from_numpy(z["topv"]).float())     # (N,K)
        h = int(z["h"]); w = int(z["w"])
        hw_list.append((h, w))

    # 假设本 batch 的 CLIP grid 一致（同一个 CLIP 模型通常一致，例如 14x14）
    # 若不一致，可做 pad/分组，这里先 assert 保证训练简化
    assert len(set(hw_list)) == 1, f"Cache grid differs inside a batch: {hw_list}"
    h, w = hw_list[0]

    topi = torch.stack(topi_list, dim=0)  # (B,N,K)
    topv = torch.stack(topv_list, dim=0)  # (B,N,K)

    return list(paths), enc_x, topi, topv, (h, w)


def window_sample_indices(h: int, w: int, win: int, seed: int = None) -> torch.Tensor:
    if seed is not None:
        random.seed(seed)
    idxs = []
    for r in range(0, h, win):
        for c in range(0, w, win):
            rs = list(range(r, min(r + win, h)))
            cs = list(range(c, min(c + win, w)))
            rr = random.choice(rs)
            cc = random.choice(cs)
            idxs.append(rr * w + cc)
    return torch.tensor(idxs, dtype=torch.long)


def symmetric_infonce(v: torch.Tensor, s: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = (v @ s.t()) / max(temperature, 1e-6)
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_index_csv", type=str, required=True, help="data/stage1_clip_cache/index.csv")
    ap.add_argument("--llm_phrase_pt", type=str, required=True)
    ap.add_argument("--domain_vocab", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="checkpoints/stage1_cached")

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--win", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--feat_key", type=str, default="0", choices=["0","1","2","3"])
    ap.add_argument("--align_to_cache_grid", action="store_true", help="把 encoder 特征插值到 cache 的 h,w")
    ap.add_argument("--train_encoder", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # 可选：DEM 分支开关
    ap.add_argument("--disable_dem2", action="store_true")
    ap.add_argument("--disable_dem3", action="store_true")
    ap.add_argument("--disable_dem4", action="store_true")
    ap.add_argument("--disable_dem5", action="store_true")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # domain vocab（用于顺序一致性检查）
    vocab = [l.strip() for l in open(args.domain_vocab, "r", encoding="utf-8") if l.strip()]

    payload = torch.load(args.llm_phrase_pt, map_location="cpu")
    assert payload["phrases"] == vocab, "domain_vocab.txt 与 llm_phrase_embeds.pt 的短语顺序不一致，请重新预计算。"
    llm_phrase = F.normalize(payload["embeds"].to(device), dim=-1)   # (V,d_llm)
    llm_dim = llm_phrase.size(-1)

    # DEM-Encoder
    enc_cfg = DEMEncoderConfig()
    pyramid = SimplePyramidBackbone()
    encoder = DEMVisionBackbone(
        pyramid_backbone=pyramid,
        cfg=enc_cfg,
        disable_dem2=args.disable_dem2,
        disable_dem3=args.disable_dem3,
        disable_dem4=args.disable_dem4,
        disable_dem5=args.disable_dem5,
    ).to(device)

    if not args.train_encoder:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

    adapter = DAAdapter(DAAdapterConfig(in_channels=encoder.out_channels, llm_dim=llm_dim)).to(device)
    adapter.train()

    params = list(adapter.parameters()) + (list(encoder.parameters()) if args.train_encoder else [])
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    ds = CacheIndexDataset(args.cache_index_csv)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 建议先 0，确认无误后再加（spawn 情况需避免 lambda）
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_cached(b, enc_cfg, args.image_size),
    )

    global_step = 0
    for epoch in range(args.epochs):
        for paths, enc_x, topi, topv, (h, w) in dl:
            enc_x = enc_x.to(device, non_blocking=True)
            topi = topi.to(device, non_blocking=True)   # (B,N,K)
            topv = topv.to(device, non_blocking=True)   # (B,N,K)

            # Encoder -> features
            if args.train_encoder:
                feats: OrderedDict = encoder(enc_x)
            else:
                with torch.no_grad():
                    feats: OrderedDict = encoder(enc_x)

            feat = feats[args.feat_key]                  # (B,C,Hf,Wf)

            # align to cache grid (h,w)
            if args.align_to_cache_grid:
                feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)

            B, C, Hf, Wf = feat.shape
            N_cache = h * w

            # Adapter -> V (B,Nv,d)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                V = adapter(feat)                        # (B, Hf*Wf, d)
                V = F.normalize(V, dim=-1)

                # build S from cached topi/topv
                # topi/topv are on cache grid (N_cache)
                w_soft = F.softmax(topv, dim=-1)         # (B,N_cache,K)
                s_k = llm_phrase[topi]                   # (B,N_cache,K,d)
                S = (w_soft.unsqueeze(-1) * s_k).sum(dim=-2)  # (B,N_cache,d)
                S = F.normalize(S, dim=-1)

                # if V grid differs from cache grid, up/down sample S to match V token count
                if V.size(1) != S.size(1):
                    Sd = S.transpose(1,2).reshape(B, llm_dim, h, w)
                    Sd = F.interpolate(Sd, size=(Hf, Wf), mode="bilinear", align_corners=False)
                    S = Sd.flatten(2).transpose(1,2).contiguous()
                    S = F.normalize(S, dim=-1)

                idx = window_sample_indices(Hf, Wf, win=args.win, seed=args.seed + global_step).to(device)
                V_s = V[:, idx, :].reshape(-1, llm_dim)
                S_s = S[:, idx, :].reshape(-1, llm_dim)

                loss = symmetric_infonce(V_s, S_s, temperature=args.temperature)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            global_step += 1
            if global_step % 20 == 0:
                print(f"step={global_step} loss={loss.item():.4f} feat={Hf}x{Wf}")

        ckpt = {"adapter": adapter.state_dict(), "args": vars(args)}
        if args.train_encoder:
            ckpt["encoder"] = encoder.state_dict()
        torch.save(ckpt, os.path.join(args.output_dir, f"stage1_cached_epoch{epoch+1}.pt"))

    print(f"Done. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
