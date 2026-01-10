# Archived: legacy stage1 training script (v3). Replaced by Adapter/train_stage1.py + configs.
# scripts/train_stage1_adapter_alignment_cached.py
from __future__ import annotations
import argparse, os, math, random, re, atexit
from pathlib import Path
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
from PIL import Image
import time
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from dem.models.backbone import SimplePyramidBackbone, ResNetPyramidBackbone
from dem.models.da_adapter import DAAdapter, DAAdapterConfig
from dem.models.dem_encoder import DEMEncoderConfig, DEMVisionBackbone


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


def normalize(t: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    m = torch.tensor(mean, dtype=t.dtype).view(3, 1, 1)
    s = torch.tensor(std, dtype=t.dtype).view(3, 1, 1)
    return (t - m) / s


def collate_cached(batch, enc_cfg: DEMEncoderConfig, image_size: int):
    paths, imgs, cache_paths = zip(*batch)

    enc_imgs = [pil_resize_square(im, image_size) for im in imgs]
    enc_x = torch.stack([normalize(pil_to_tensor(im), enc_cfg.image_mean, enc_cfg.image_std) for im in enc_imgs], dim=0)

    # 读取 cache（npz）——每张图一个文件
    topi_list, topv_list, hw_list = [], [], []
    for cp in cache_paths:
        z = np.load(cp)
        topi_list.append(torch.from_numpy(z["topi"]).long())  # (N,K)
        topv_list.append(torch.from_numpy(z["topv"]).float())  # (N,K)
        h = int(z["h"]);
        w = int(z["w"])
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


def supervised_xmodal_contrastive(
        v: torch.Tensor,
        s: torch.Tensor,
        y: torch.Tensor,
        temperature: float,
        chunk_size: int = 0,
) -> torch.Tensor:
    """Multi-positive cross-modal contrastive loss (SupCon-style).

    For each visual token v_i (anchor), all semantic tokens s_j with the SAME label y are treated as positives.
    Symmetric direction (s as anchor, v as positives) is also applied.

    Args:
        v: (M, d) normalized visual token embeddings
        s: (M, d) normalized semantic token embeddings (teacher targets)
        y: (M,) integer labels (e.g., top-1 phrase id) aligned with v/s tokens
        temperature: contrastive temperature
        chunk_size: if >0, compute logits in chunks to reduce memory

    Returns:
        scalar loss
    """
    assert v.dim() == 2 and s.dim() == 2, f"v/s must be 2D, got {v.shape}, {s.shape}"
    assert v.size(0) == s.size(0), f"M mismatch: {v.size(0)} vs {s.size(0)}"
    y = y.view(-1)
    assert y.numel() == v.size(0), f"label size mismatch: {y.numel()} vs {v.size(0)}"

    temp = max(float(temperature), 1e-6)

    def _dir_loss(a: torch.Tensor, b: torch.Tensor, ya: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        M = a.size(0)
        Bt = b.t()  # (d, M)
        losses = []

        if chunk_size and M > chunk_size:
            for i0 in range(0, M, chunk_size):
                i1 = min(M, i0 + chunk_size)
                logits = (a[i0:i1] @ Bt) / temp  # (m, M)
                # log denominator
                log_denom = torch.logsumexp(logits, dim=1)  # (m,)
                # positives mask
                pos_mask = (ya[i0:i1].unsqueeze(1) == yb.unsqueeze(0))  # (m, M)
                neg_inf = torch.finfo(logits.dtype).min
                logits_pos = logits.masked_fill(~pos_mask, neg_inf)
                log_num = torch.logsumexp(logits_pos, dim=1)  # (m,)
                losses.append(-(log_num - log_denom))
            return torch.cat(losses, dim=0).mean()
        else:
            logits = (a @ Bt) / temp  # (M, M)
            log_denom = torch.logsumexp(logits, dim=1)
            pos_mask = (ya.unsqueeze(1) == yb.unsqueeze(0))
            neg_inf = torch.finfo(logits.dtype).min
            logits_pos = logits.masked_fill(~pos_mask, neg_inf)
            log_num = torch.logsumexp(logits_pos, dim=1)
            return (-(log_num - log_denom)).mean()

    loss_v2s = _dir_loss(v, s, y, y)
    loss_s2v = _dir_loss(s, v, y, y)
    return 0.5 * (loss_v2s + loss_s2v)


def window_sample_indices_by_conf(conf_hw: torch.Tensor, win: int, mode: str, seed: int = None) -> torch.Tensor:
    """
    conf_hw: (H,W) confidence map on the SAME grid as V tokens
    mode:
      - random: uniform random in each window
      - max: pick max confidence in each window
      - weighted: sample proportional to confidence in each window
    """
    if seed is not None:
        random.seed(seed)

    H, W = conf_hw.shape
    idxs = []
    for r in range(0, H, win):
        for c in range(0, W, win):
            r2 = min(r + win, H)
            c2 = min(c + win, W)

            window = conf_hw[r:r2, c:c2].reshape(-1)
            n = window.numel()

            if mode == "max":
                j = int(window.argmax().item())
            elif mode == "weighted":
                p = window.clamp_min(0)
                s = float(p.sum().item())
                if s <= 0:
                    j = random.randrange(n)
                else:
                    j = int(torch.multinomial(p / p.sum(), 1).item())
            else:  # random
                j = random.randrange(n)

            ww = (c2 - c)
            rr = r + (j // ww)
            cc = c + (j % ww)
            idxs.append(rr * W + cc)

    return torch.tensor(idxs, dtype=torch.long, device=conf_hw.device)


def load_encoder_ckpt_by_suffix(encoder: torch.nn.Module, ckpt_path: str):
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 1) 抽取 state_dict
    sd = None
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "net", "ema"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                break
    if sd is None:
        sd = ckpt if isinstance(ckpt, dict) else None
    if sd is None:
        raise ValueError("Cannot find state_dict in ckpt.")

    # 2) 去掉常见前缀
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    msd = encoder.state_dict()
    loadable = {}

    # 3) 先尝试完全同名
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            loadable[k] = v

    # 4) 再做 suffix 匹配（解决 backbone./model.backbone./encoder. 等前缀）
    if len(loadable) < len(msd) * 0.8:
        # 建一个 “suffix -> key” 索引（避免 O(N^2) 太慢）
        # 注意：suffix 长度可按需调，这里用全长 endswith 做即可（key 数量通常不大）
        for mk in msd.keys():
            if mk in loadable:
                continue
            # 找到一个 ckpt key 使得 ckpt_key.endswith(mk)
            candidates = [ck for ck in sd.keys() if ck.endswith(mk)]
            if not candidates:
                continue
            # 若有多个候选，选最短的（前缀最少，通常最正确）
            ck_best = min(candidates, key=len)
            v = sd[ck_best]
            if msd[mk].shape == v.shape:
                loadable[mk] = v

    encoder.load_state_dict(loadable, strict=False)

    print(f"[encoder load] loadable tensors: {len(loadable)}/{len(msd)}")
    # 你也可以打印缺失 key（可选）
    # missing = [k for k in msd.keys() if k not in loadable]
    # print(f"[encoder load] missing={len(missing)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_index_csv", type=str, required=True, help="data/stage1_clip_cache/index.csv")
    ap.add_argument("--llm_phrase_pt", type=str, required=True)
    ap.add_argument("--domain_vocab", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="checkpoints/stage1_cached")

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=20, help="Log/update tqdm postfix every N steps")

    # ---- resume / checkpoint ----
    ap.add_argument("--resume", type=str, default="", help="Path to a checkpoint to resume training")
    ap.add_argument("--resume_strict", action="store_true", help="Use strict=True when loading state_dict")
    ap.add_argument("--reset_optimizer", action="store_true",
                    help="When resuming, do NOT load optimizer/scaler state (fresh optimizer)")
    ap.add_argument("--reset_rng", action="store_true",
                    help="When resuming, do NOT restore RNG states (default restores if present)")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--win", type=int, default=2)
    ap.add_argument("--loss_type", type=str, default="supcon",
                    choices=["infonce", "supcon", "mix"],
                    help="Loss: infonce (diagonal), supcon (multi-positive), or mix")
    ap.add_argument("--supcon_weight", type=float, default=0.7,
                    help="When loss_type=mix: weight for supcon term (0~1)")
    ap.add_argument("--supcon_chunk", type=int, default=0,
                    help="Chunk size for supcon logits to save memory; 0=full matmul")
    # ---- teacher soft targets (Top-N) ----
    ap.add_argument("--teacher_topk", type=int, default=3, help="Top-N soft target from CLIP cache")
    ap.add_argument("--teacher_temperature", type=float, default=0.03, help="Softmax temp for topv -> weights")

    # ---- confidence filtering / sampling ----
    ap.add_argument("--conf_thresh", type=float, default=0.0, help="Drop tokens whose max(topv) < thresh")
    ap.add_argument("--min_tokens", type=int, default=64, help="Skip step if kept tokens < this")
    ap.add_argument("--sample_mode", type=str, default="weighted",
                    choices=["random", "max", "weighted"],
                    help="How to pick one token per window")

    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--feat_key", type=str, default="0", choices=["0", "1", "2", "3"])
    ap.add_argument("--align_to_cache_grid", action="store_true", help="把 encoder 特征插值到 cache 的 h,w")
    ap.add_argument("--train_encoder", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # 可选：DEM 分支开关
    ap.add_argument("--disable_dem2", action="store_true")
    ap.add_argument("--disable_dem3", action="store_true")
    ap.add_argument("--disable_dem4", action="store_true")
    ap.add_argument("--disable_dem5", action="store_true")

    ap.add_argument("--encoder_ckpt", type=str, default="", help="Path to pretrained DEM-Encoder checkpoint (.pt/.pth)")
    ap.add_argument("--encoder_ckpt_key", type=str, default="encoder",
                    help="State dict key name if checkpoint is a dict")

    ap.add_argument("--backbone", type=str, default="resnet50",
                    choices=["resnet50", "simple"])
    ap.add_argument("--backbone_pretrained", action="store_true",
                    help="Only used when backbone=resnet50 and no ckpt is loaded")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------
    # Logging setup (text log + jsonl metrics)
    # -------------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / f"train_{run_id}.log"
    jsonl_path = out_dir / f"metrics_{run_id}.jsonl"

    logger = logging.getLogger("train_v3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    _fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    _fh = logging.FileHandler(log_path, encoding="utf-8")
    _fh.setLevel(logging.INFO)
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)

    _sh = logging.StreamHandler()
    _sh.setLevel(logging.INFO)
    _sh.setFormatter(_fmt)
    logger.addHandler(_sh)

    jsonl_f = open(jsonl_path, "a", encoding="utf-8", buffering=1)

    def log_json(obj: dict):
        obj["ts"] = datetime.now().isoformat(timespec="seconds")
        jsonl_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ensure the handle is closed on normal exit
    atexit.register(lambda: jsonl_f.close())

    logger.info(f"run_id={run_id}")
    logger.info(f"log_path={log_path}")
    logger.info(f"jsonl_path={jsonl_path}")
    logger.info("args=" + json.dumps(vars(args), ensure_ascii=False))

    # minimal environment info
    try:
        logger.info(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device}")
        if torch.cuda.is_available():
            logger.info(f"gpu={torch.cuda.get_device_name(0)}")
    except Exception as _e:
        logger.info(f"env_log_failed: {_e}")

    log_json({"event": "run_start", "run_id": run_id, "device": str(device)})

    # domain vocab（用于顺序一致性检查）
    vocab = [l.strip() for l in open(args.domain_vocab, "r", encoding="utf-8") if l.strip()]

    payload = torch.load(args.llm_phrase_pt, map_location="cpu")
    assert payload["phrases"] == vocab, "domain_vocab.txt 与 llm_phrase_embeds.pt 的短语顺序不一致，请重新预计算。"
    llm_phrase = F.normalize(payload["embeds"].to(device), dim=-1)  # (V,d_llm)
    llm_dim = llm_phrase.size(-1)

    # DEM-Encoder
    enc_cfg = DEMEncoderConfig()

    if args.backbone == "resnet50":
        pyramid = ResNetPyramidBackbone(name="resnet50", pretrained=args.backbone_pretrained)
    else:
        pyramid = SimplePyramidBackbone()

    encoder = DEMVisionBackbone(
        pyramid_backbone=pyramid,
        cfg=enc_cfg,
        disable_dem2=args.disable_dem2,
        disable_dem3=args.disable_dem3,
        disable_dem4=args.disable_dem4,
        disable_dem5=args.disable_dem5,
    ).to(device)

    if args.encoder_ckpt:
        load_encoder_ckpt_by_suffix(encoder, args.encoder_ckpt)

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

    # -------------------------
    # Resume support
    # -------------------------
    start_epoch = 0
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

        # ---- model states ----
        if "adapter" in ckpt:
            adapter.load_state_dict(ckpt["adapter"], strict=args.resume_strict)
        else:
            # backward compatibility: if checkpoint is a pure state_dict
            if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
                adapter.load_state_dict(ckpt, strict=args.resume_strict)
            else:
                raise KeyError("Checkpoint missing 'adapter' state.")

        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"], strict=False)

        # ---- optimizer / scaler ----
        if (not args.reset_optimizer) and ("optim" in ckpt):
            try:
                optim.load_state_dict(ckpt["optim"])
            except Exception as e:
                print(f"[resume] Warning: failed to load optimizer state: {e}")
        elif args.resume and (not args.reset_optimizer):
            print("[resume] Warning: no optimizer state in checkpoint (will resume weights only).")

        if (not args.reset_optimizer) and ("scaler" in ckpt) and (ckpt["scaler"] is not None) and args.amp:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"[resume] Warning: failed to load GradScaler state: {e}")

        # ---- epoch / step ----
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"])
        else:
            # fallback: parse from filename '...epoch{n}.pt'
            m = re.search(r"epoch(\d+)", os.path.basename(args.resume))
            start_epoch = int(m.group(1)) if m else 0

        global_step = int(ckpt.get("global_step", 0))

        # ---- RNG states (optional) ----
        if (not args.reset_rng) and ("rng" in ckpt) and (ckpt["rng"] is not None):
            rng = ckpt["rng"]
            try:
                random.setstate(rng["py"])
                np.random.set_state(rng["np"])
                torch.set_rng_state(rng["torch"])
                if torch.cuda.is_available() and (rng.get("cuda") is not None):
                    torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as e:
                print(f"[resume] Warning: failed to restore RNG states: {e}")

        print(f"[resume] Loaded {args.resume} (start_epoch={start_epoch}, global_step={global_step})")

        try:
            logger.info(
                f"[resume] path={args.resume} start_epoch={start_epoch} global_step={global_step} "
                f"reset_optimizer={args.reset_optimizer} reset_rng={args.reset_rng}"
            )
            log_json({
                "event": "resume",
                "path": args.resume,
                "start_epoch": int(start_epoch),
                "global_step": int(global_step),
                "reset_optimizer": bool(args.reset_optimizer),
                "reset_rng": bool(args.reset_rng),
            })
        except Exception as _e:
            print(f"[resume] log failed: {_e}")

        # ensure modes are correct after loading
        adapter.train()
        if args.train_encoder:
            encoder.train()
        else:
            encoder.eval()

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        # epoch running stats (averaged over batches)
        epoch_t0 = time.time()
        ep_skips = 0
        ep_loss_sum = 0.0
        ep_diag_sum = 0.0
        ep_phrase_sum = 0.0
        ep_group_sum = 0.0
        ep_n = 0
        ep_phrase_n = 0
        ep_group_n = 0
        for paths, enc_x, topi, topv, (h, w) in pbar:
            enc_x = enc_x.to(device, non_blocking=True)
            topi = topi.to(device, non_blocking=True)  # (B,N,K)
            topv = topv.to(device, non_blocking=True)  # (B,N,K)

            # DataLoader collate may stack h/w into tensors; convert to python ints
            if torch.is_tensor(h):
                if h.numel() > 1:
                    if not torch.all(h == h.flatten()[0]):
                        # if mixed, fall back to per-sample handling (not supported here)
                        raise ValueError(f"Mixed h in one batch: {h}")
                    h = int(h.flatten()[0].item())
                else:
                    h = int(h.item())
            else:
                h = int(h)
            if torch.is_tensor(w):
                if w.numel() > 1:
                    if not torch.all(w == w.flatten()[0]):
                        raise ValueError(f"Mixed w in one batch: {w}")
                    w = int(w.flatten()[0].item())
                else:
                    w = int(w.item())
            else:
                w = int(w)

            # Encoder -> features
            if args.train_encoder:
                feats: OrderedDict = encoder(enc_x)
            else:
                with torch.no_grad():
                    feats: OrderedDict = encoder(enc_x)

            feat = feats[args.feat_key]  # (B,C,Hf,Wf)

            # align to cache grid (h,w)
            if args.align_to_cache_grid:
                feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)

            B, C, Hf, Wf = feat.shape
            N_cache = h * w

            # feat: (B,C,H,W)
            # print("feat std(all) =", feat.float().std().item())
            # print("feat std(spatial avg) =", feat.float().flatten(2).std(dim=2).mean().item())  # 每个样本在空间维度的方差

            # Adapter -> V (B,Nv,d)
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                V = adapter(feat)
                # print("V token std =", V.float().std(dim=1).mean().item())  # token 维度方差（关键）
                # print("V batch std =", V.float().std(dim=0).mean().item())
                V = F.normalize(V, dim=-1)

                # build soft S from cached topi/topv (Top-N weighted)
                K_total = topi.size(-1)
                K = min(args.teacher_topk, K_total)

                # weights: softmax(topv / tau)
                w_logits = topv[..., :K] / max(args.teacher_temperature, 1e-6)  # (B,N_cache,K)
                w_soft = F.softmax(w_logits, dim=-1)  # (B,N_cache,K)

                # phrase embeds: (B,N_cache,K,d)
                P = llm_phrase[topi[..., :K]]
                S = (w_soft.unsqueeze(-1) * P).sum(dim=-2)  # (B,N_cache,d)
                S = F.normalize(S, dim=-1)

                # if V grid differs from cache grid, up/down sample S to match V token count
                if V.size(1) != S.size(1):
                    Sd = S.transpose(1, 2).reshape(B, llm_dim, h, w)
                    Sd = F.interpolate(Sd, size=(Hf, Wf), mode="bilinear", align_corners=False)
                    S = Sd.flatten(2).transpose(1, 2).contiguous()
                    S = F.normalize(S, dim=-1)
                # ---- build confidence map (from cache topv) and align to V grid ----
                conf_cache = topv.max(dim=-1).values  # (B,N_cache)
                conf_map = conf_cache.reshape(B, 1, h, w)  # (B,1,h,w)

                if V.size(1) != N_cache:
                    conf_map = F.interpolate(conf_map, size=(Hf, Wf), mode="bilinear", align_corners=False)

                conf_map = conf_map.squeeze(1)  # (B,Hf,Wf)

                # ---- per-sample window sampling (better for sparse defects) ----
                idx_batch = []
                for b in range(B):
                    idx_b = window_sample_indices_by_conf(
                        conf_map[b], win=args.win, mode=args.sample_mode,
                        seed=args.seed + global_step * 131 + b
                    )
                    idx_batch.append(idx_b)
                idx_batch = torch.stack(idx_batch, dim=0)  # (B, n_win)
                n_win = idx_batch.size(1)

                # gather tokens
                gather_index = idx_batch.unsqueeze(-1).expand(B, n_win, llm_dim)
                V_sel = V.gather(dim=1, index=gather_index)  # (B,n_win,d)
                S_sel = S.gather(dim=1, index=gather_index)  # (B,n_win,d)

                # gather confidence for filtering
                conf_flat = conf_map.flatten(1)  # (B,Hf*Wf)
                conf_sel = conf_flat.gather(1, idx_batch)  # (B,n_win)
                keep = (conf_sel.reshape(-1) >= args.conf_thresh)  # (M,)

                V_s = V_sel.reshape(-1, llm_dim)[keep]  # (M',d)
                S_s = S_sel.reshape(-1, llm_dim)[keep]  # (M',d)

                # labels for supervised contrastive / metrics (Top-1 phrase id)
                y_train = None
                if V.size(1) == topi.size(1):
                    y_train = topi[..., 0].gather(1, idx_batch).reshape(-1)[keep]  # (M',)

                # if too few tokens remain, skip this step (avoid degenerate CE)

                if V_s.size(0) < args.min_tokens:
                    ep_skips += 1
                    if global_step % args.log_every == 0:
                        kept_tokens = int(V_s.size(0))
                        total_tokens = int(V_sel.numel() // llm_dim)
                        lr_now = float(optim.param_groups[0]["lr"]) if len(optim.param_groups) else None
                        try:
                            logger.info(
                                f"step={global_step} SKIP kept_tokens={kept_tokens} total_tokens={total_tokens} "
                                f"min_tokens={args.min_tokens} lr={lr_now}"
                            )
                            log_json({
                                "event": "skip",
                                "epoch": int(epoch),
                                "step": int(global_step),
                                "kept_tokens": kept_tokens,
                                "total_tokens": total_tokens,
                                "min_tokens": int(args.min_tokens),
                                "lr": lr_now,
                            })
                        except Exception:
                            pass
                    global_step += 1
                    continue

                # ---- centerize + normalize (reduce hubness / anisotropy) ----
                V_s = V_s - V_s.mean(dim=0, keepdim=True)
                S_s = S_s - S_s.mean(dim=0, keepdim=True)
                V_s = F.normalize(V_s, dim=-1)
                S_s = F.normalize(S_s, dim=-1)

                # ---- metrics in fp32 (disable autocast) ----
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        Vn = V_s.float()
                        Sn = S_s.float()

                        # (1) strict diagonal top-1 (in-batch)
                        sim = Vn @ Sn.t()
                        pred_inbatch = sim.argmax(dim=1)
                        diag_top1_acc = (
                                pred_inbatch == torch.arange(sim.size(0), device=sim.device)).float().mean().item()

                        # (2) phrase-level 38-way top-1 acc (更稳定、更符合你任务)
                        # 只有在 token 与 cache grid 一致时才有离散标签（你现在通常 align_to_cache_grid=True，因此成立）
                        phrase_top1_acc = -1.0
                        if y_train is not None:
                            sim_phrase = Vn @ llm_phrase.float().t()  # (M, Vocab)
                            pred_phrase = sim_phrase.argmax(dim=1)
                            phrase_top1_acc = (pred_phrase == y_train).float().mean().item()

                        # (3) group-correct acc: 允许命中同类 token（缓解“对角线过苛刻”的低估）
                        group_correct_acc = -1.0
                        if y_train is not None:
                            group_correct_acc = (y_train[pred_inbatch] == y_train).float().mean().item()

                # ---- training loss (fp32 for stability) ----
                with torch.cuda.amp.autocast(enabled=False):
                    V_fp = V_s.float()
                    S_fp = S_s.float()
                    if args.loss_type == "supcon" and (y_train is not None):
                        loss = supervised_xmodal_contrastive(V_fp, S_fp, y_train, temperature=args.temperature,
                                                             chunk_size=args.supcon_chunk)
                    elif args.loss_type == "mix" and (y_train is not None):
                        loss_sup = supervised_xmodal_contrastive(V_fp, S_fp, y_train, temperature=args.temperature,
                                                                 chunk_size=args.supcon_chunk)
                        loss_inf = symmetric_infonce(V_fp, S_fp, temperature=args.temperature)
                        w = float(args.supcon_weight)
                        loss = w * loss_sup + (1.0 - w) * loss_inf
                    else:
                        # fallback: diagonal InfoNCE
                        loss = symmetric_infonce(V_fp, S_fp, temperature=args.temperature)

                # ---- accumulate epoch stats ----
                ep_loss_sum += float(loss.item())
                ep_diag_sum += float(diag_top1_acc)
                ep_n += 1
                if phrase_top1_acc >= 0:
                    ep_phrase_sum += float(phrase_top1_acc)
                    ep_phrase_n += 1
                if group_correct_acc >= 0:
                    ep_group_sum += float(group_correct_acc)
                    ep_group_n += 1

                if global_step % args.log_every == 0:
                    # update tqdm postfix with running epoch averages
                    avg_loss = ep_loss_sum / max(ep_n, 1)
                    avg_diag = ep_diag_sum / max(ep_n, 1)
                    avg_phrase = (ep_phrase_sum / ep_phrase_n) if ep_phrase_n > 0 else -1.0
                    avg_group = (ep_group_sum / ep_group_n) if ep_group_n > 0 else -1.0
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "diag": f"{avg_diag:.4f}",
                        "phrase": f"{avg_phrase:.4f}" if avg_phrase >= 0 else "NA",
                        "group": f"{avg_group:.4f}" if avg_group >= 0 else "NA",
                        "step": global_step,
                    })
                    tqdm.write(
                        f"step={global_step} loss={loss.item():.4f} diag={diag_top1_acc:.4f} "
                        f"phrase={phrase_top1_acc:.4f} group={group_correct_acc:.4f} feat={Hf}x{Wf}"
                    )

                    kept_tokens = int(V_s.size(0))
                    total_tokens = int(V_sel.numel() // llm_dim)
                    lr_now = float(optim.param_groups[0]["lr"]) if len(optim.param_groups) else None
                    try:
                        logger.info(
                            f"step={global_step} loss={loss.item():.4f} diag={diag_top1_acc:.4f} "
                            f"phrase={phrase_top1_acc:.4f} group={group_correct_acc:.4f} "
                            f"kept={kept_tokens}/{total_tokens} lr={lr_now}"
                        )
                        log_json({
                            "event": "step",
                            "epoch": int(epoch),
                            "step": int(global_step),
                            "lr": lr_now,
                            "loss": float(loss.item()),
                            "diag": float(diag_top1_acc),
                            "phrase": float(phrase_top1_acc),
                            "group": float(group_correct_acc),
                            "avg_loss": float(avg_loss),
                            "avg_diag": float(avg_diag),
                            "avg_phrase": float(avg_phrase),
                            "avg_group": float(avg_group),
                            "kept_tokens": kept_tokens,
                            "total_tokens": total_tokens,
                        })
                    except Exception:
                        pass

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            global_step += 1


        # ---- epoch summary ----
        epoch_sec = time.time() - epoch_t0
        avg_loss = ep_loss_sum / max(ep_n, 1)
        avg_diag = ep_diag_sum / max(ep_n, 1)
        avg_phrase = (ep_phrase_sum / ep_phrase_n) if ep_phrase_n > 0 else -1.0
        avg_group = (ep_group_sum / ep_group_n) if ep_group_n > 0 else -1.0
        msg = (
            f"[epoch {epoch + 1}/{args.epochs}] "
            f"loss={avg_loss:.4f} diag={avg_diag:.4f} "
            f"phrase={avg_phrase:.4f} group={avg_group:.4f} "
            f"skips={ep_skips} time_sec={epoch_sec:.1f}"
        )
        tqdm.write(msg)
        try:
            logger.info(msg)
            log_json({
                "event": "epoch_end",
                "epoch": int(epoch),
                "avg_loss": float(avg_loss),
                "avg_diag": float(avg_diag),
                "avg_phrase": float(avg_phrase),
                "avg_group": float(avg_group),
                "skips": int(ep_skips),
                "time_sec": float(epoch_sec),
            })
        except Exception:
            pass

        ckpt = {
            "adapter": adapter.state_dict(),
            "args": vars(args),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict() if (args.amp and device.type == "cuda") else None,
            "epoch": epoch + 1,  # next epoch to run (0-based)
            "global_step": global_step,
            "rng": {
                "py": random.getstate(),
                "np": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        if args.train_encoder:
            ckpt["encoder"] = encoder.state_dict()

        save_path = os.path.join(args.output_dir, f"stage1_cached_epoch{epoch + 1}.pt")
        torch.save(ckpt, save_path)
        print(f"[ckpt] saved: {save_path}")

        try:
            logger.info(f"[ckpt] saved: {save_path}")
            log_json({"event": "ckpt", "epoch": int(epoch), "path": str(save_path)})
        except Exception:
            pass


    try:
        jsonl_f.flush()
        jsonl_f.close()
    except Exception:
        pass
    print(f"Done. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
