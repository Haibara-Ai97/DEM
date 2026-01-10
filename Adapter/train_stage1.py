from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import random
import re
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from dem.config_utils import apply_overrides, get_by_path, load_yaml, write_yaml

from dem.models.backbone import ResNetPyramidBackbone, SimplePyramidBackbone
from dem.models.da_adapter import DAAdapter, DAAdapterConfig
from dem.models.dem_encoder import DEMEncoderConfig, DEMVisionBackbone
from dem.data.datasets import CacheCollator, CacheIndexDataset


def window_sample_indices(h: int, w: int, win: int, seed: int | None = None) -> torch.Tensor:
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


def window_sample_indices_by_conf(conf_hw: torch.Tensor, win: int, mode: str, seed: int | None = None) -> torch.Tensor:
    if seed is not None:
        random.seed(seed)

    h, w = conf_hw.shape
    idxs = []
    for r in range(0, h, win):
        for c in range(0, w, win):
            r2 = min(r + win, h)
            c2 = min(c + win, w)

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
            else:
                j = random.randrange(n)

            ww = c2 - c
            rr = r + (j // ww)
            cc = c + (j % ww)
            idxs.append(rr * w + cc)

    return torch.tensor(idxs, dtype=torch.long, device=conf_hw.device)


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
    assert v.dim() == 2 and s.dim() == 2, f"v/s must be 2D, got {v.shape}, {s.shape}"
    assert v.size(0) == s.size(0), f"M mismatch: {v.size(0)} vs {s.size(0)}"
    y = y.view(-1)
    assert y.numel() == v.size(0), f"label size mismatch: {y.numel()} vs {v.size(0)}"

    temp = max(float(temperature), 1e-6)

    def _dir_loss(a: torch.Tensor, b: torch.Tensor, ya: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        m = a.size(0)
        bt = b.t()
        losses = []

        if chunk_size and m > chunk_size:
            for i0 in range(0, m, chunk_size):
                i1 = min(m, i0 + chunk_size)
                logits = (a[i0:i1] @ bt) / temp
                log_denom = torch.logsumexp(logits, dim=1)
                pos_mask = (ya[i0:i1].unsqueeze(1) == yb.unsqueeze(0))
                neg_inf = torch.finfo(logits.dtype).min
                logits_pos = logits.masked_fill(~pos_mask, neg_inf)
                log_num = torch.logsumexp(logits_pos, dim=1)
                losses.append(-(log_num - log_denom))
            return torch.cat(losses, dim=0).mean()

        logits = (a @ bt) / temp
        log_denom = torch.logsumexp(logits, dim=1)
        pos_mask = (ya.unsqueeze(1) == yb.unsqueeze(0))
        neg_inf = torch.finfo(logits.dtype).min
        logits_pos = logits.masked_fill(~pos_mask, neg_inf)
        log_num = torch.logsumexp(logits_pos, dim=1)
        return (-(log_num - log_denom)).mean()

    loss_v2s = _dir_loss(v, s, y, y)
    loss_s2v = _dir_loss(s, v, y, y)
    return 0.5 * (loss_v2s + loss_s2v)


def load_encoder_ckpt_by_suffix(encoder: torch.nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
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

    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    msd = encoder.state_dict()
    loadable = {}

    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            loadable[k] = v

    if len(loadable) < len(msd) * 0.8:
        for mk in msd.keys():
            if mk in loadable:
                continue
            candidates = [ck for ck in sd.keys() if ck.endswith(mk)]
            if not candidates:
                continue
            ck_best = min(candidates, key=len)
            v = sd[ck_best]
            if msd[mk].shape == v.shape:
                loadable[mk] = v

    encoder.load_state_dict(loadable, strict=False)
    print(f"[encoder load] loadable tensors: {len(loadable)}/{len(msd)}")


def build_scheduler(optim: torch.optim.Optimizer, cfg: dict):
    sched_type = get_by_path(cfg, "scheduler.type", "none")
    if sched_type == "none":
        return None
    if sched_type == "step":
        step_size = int(get_by_path(cfg, "scheduler.step_size", 1))
        gamma = float(get_by_path(cfg, "scheduler.gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
    if sched_type == "cosine":
        t_max = int(get_by_path(cfg, "scheduler.t_max", 10))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max)
    raise ValueError(f"Unknown scheduler.type: {sched_type}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to configs/stage1/*.yaml")
    ap.add_argument("--set", action="append", default=[], help="Override config values with key=value")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.set)

    cache_index_csv = get_by_path(cfg, "data.cache_index") or get_by_path(cfg, "data.cache_index_csv")
    llm_phrase_pt = get_by_path(cfg, "data.llm_phrase_pt")
    domain_vocab = get_by_path(cfg, "data.domain_vocab")
    if not cache_index_csv or not llm_phrase_pt or not domain_vocab:
        raise ValueError("Config must set data.cache_index, data.llm_phrase_pt, data.domain_vocab")

    output_dir = get_by_path(cfg, "output.dir", "checkpoints/stage1_cached")
    os.makedirs(output_dir, exist_ok=True)
    write_yaml(Path(output_dir) / "config.yaml", cfg)

    seed = int(get_by_path(cfg, "training.seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_cfg = get_by_path(cfg, "logging", {})
    log_enable = bool(get_by_path(log_cfg, "enable", True))
    log_every = int(get_by_path(log_cfg, "log_every", 20))
    use_tqdm = bool(get_by_path(log_cfg, "use_tqdm", True))
    write_log = bool(get_by_path(log_cfg, "write_log", True))
    write_jsonl = bool(get_by_path(log_cfg, "write_jsonl", True))
    run_id = get_by_path(log_cfg, "run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = get_by_path(log_cfg, "log_dir") or output_dir

    logger = None
    jsonl_f = None
    if log_enable and (write_log or write_jsonl):
        log_path = Path(log_dir) / f"train_{run_id}.log"
        jsonl_path = Path(output_dir) / "metrics.jsonl"

        if write_log:
            logger = logging.getLogger("train_stage1")
            logger.setLevel(logging.INFO)
            logger.handlers.clear()
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(fmt)
            logger.addHandler(sh)

        if write_jsonl:
            jsonl_f = open(jsonl_path, "a", encoding="utf-8", buffering=1)
            atexit.register(lambda: jsonl_f.close())

        if logger:
            logger.info(f"run_id={run_id}")
            logger.info(f"log_path={log_path}")
            if write_jsonl:
                logger.info(f"jsonl_path={jsonl_path}")
            logger.info("config=" + json.dumps(cfg, ensure_ascii=False))
            logger.info(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device}")
            if torch.cuda.is_available():
                logger.info(f"gpu={torch.cuda.get_device_name(0)}")

    def log_json(obj: dict):
        if not jsonl_f:
            return
        obj["ts"] = datetime.now().isoformat(timespec="seconds")
        jsonl_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    log_json({"event": "run_start", "run_id": run_id, "device": str(device)})

    vocab = [l.strip() for l in open(domain_vocab, "r", encoding="utf-8") if l.strip()]

    payload = torch.load(llm_phrase_pt, map_location="cpu")
    assert payload["phrases"] == vocab, "domain_vocab.txt 与 llm_phrase_embeds.pt 的短语顺序不一致，请重新预计算。"
    llm_phrase = F.normalize(payload["embeds"].to(device), dim=-1)
    llm_dim = llm_phrase.size(-1)

    enc_cfg = DEMEncoderConfig()

    backbone = get_by_path(cfg, "model.backbone", "resnet50")
    backbone_pretrained = bool(get_by_path(cfg, "model.backbone_pretrained", False))
    if backbone == "resnet50":
        pyramid = ResNetPyramidBackbone(name="resnet50", pretrained=backbone_pretrained)
    else:
        pyramid = SimplePyramidBackbone()

    encoder = DEMVisionBackbone(
        pyramid_backbone=pyramid,
        cfg=enc_cfg,
        disable_dem2=bool(get_by_path(cfg, "model.disable_dem2", False)),
        disable_dem3=bool(get_by_path(cfg, "model.disable_dem3", False)),
        disable_dem4=bool(get_by_path(cfg, "model.disable_dem4", False)),
        disable_dem5=bool(get_by_path(cfg, "model.disable_dem5", False)),
    ).to(device)

    encoder_ckpt = get_by_path(cfg, "model.encoder_ckpt", "")
    if encoder_ckpt:
        load_encoder_ckpt_by_suffix(encoder, encoder_ckpt)

    train_encoder = bool(get_by_path(cfg, "training.train_encoder", False))
    if not train_encoder:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

    adapter = DAAdapter(DAAdapterConfig(in_channels=encoder.out_channels, llm_dim=llm_dim)).to(device)
    adapter.train()

    params = list(adapter.parameters()) + (list(encoder.parameters()) if train_encoder else [])
    lr = float(get_by_path(cfg, "training.lr", 1e-4))
    weight_decay = float(get_by_path(cfg, "training.weight_decay", 0.01))
    optim = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    use_amp = bool(get_by_path(cfg, "training.amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    scheduler = build_scheduler(optim, cfg)
    scheduler_step_on = get_by_path(cfg, "scheduler.step_on", "epoch")

    ds = CacheIndexDataset(cache_index_csv)
    collator = CacheCollator(enc_cfg, int(get_by_path(cfg, "training.image_size", 224)))
    dl = DataLoader(
        ds,
        batch_size=int(get_by_path(cfg, "training.batch_size", 8)),
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )

    start_epoch = 0
    global_step = 0
    resume_path = get_by_path(cfg, "resume.path", "")
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)

        if "adapter" in ckpt:
            adapter.load_state_dict(ckpt["adapter"], strict=bool(get_by_path(cfg, "resume.strict", False)))
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            adapter.load_state_dict(ckpt, strict=bool(get_by_path(cfg, "resume.strict", False)))
        else:
            raise KeyError("Checkpoint missing 'adapter' state.")

        if "encoder" in ckpt:
            encoder.load_state_dict(ckpt["encoder"], strict=False)

        reset_optimizer = bool(get_by_path(cfg, "resume.reset_optimizer", False))
        if (not reset_optimizer) and ("optim" in ckpt):
            try:
                optim.load_state_dict(ckpt["optim"])
            except Exception as e:
                print(f"[resume] Warning: failed to load optimizer state: {e}")
        elif resume_path and (not reset_optimizer):
            print("[resume] Warning: no optimizer state in checkpoint (will resume weights only).")

        if (not reset_optimizer) and ("scaler" in ckpt) and (ckpt["scaler"] is not None) and use_amp:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"[resume] Warning: failed to load GradScaler state: {e}")

        if (not reset_optimizer) and scheduler and ("scheduler" in ckpt):
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"[resume] Warning: failed to load scheduler state: {e}")

        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"])
        else:
            m = re.search(r"epoch(\d+)", os.path.basename(resume_path))
            start_epoch = int(m.group(1)) if m else 0
        global_step = int(ckpt.get("global_step", 0))

        reset_rng = bool(get_by_path(cfg, "resume.reset_rng", False))
        if (not reset_rng) and ("rng" in ckpt) and (ckpt["rng"] is not None):
            rng = ckpt["rng"]
            try:
                random.setstate(rng["py"])
                np.random.set_state(rng["np"])
                torch.set_rng_state(rng["torch"])
                if torch.cuda.is_available() and (rng.get("cuda") is not None):
                    torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as e:
                print(f"[resume] Warning: failed to restore RNG states: {e}")

        print(f"[resume] Loaded {resume_path} (start_epoch={start_epoch}, global_step={global_step})")
        if logger:
            logger.info(
                f"[resume] path={resume_path} start_epoch={start_epoch} global_step={global_step} "
                f"reset_optimizer={reset_optimizer} reset_rng={reset_rng}"
            )
        log_json({
            "event": "resume",
            "path": resume_path,
            "start_epoch": int(start_epoch),
            "global_step": int(global_step),
            "reset_optimizer": bool(reset_optimizer),
            "reset_rng": bool(reset_rng),
        })

        adapter.train()
        encoder.train() if train_encoder else encoder.eval()

    epochs = int(get_by_path(cfg, "training.epochs", 1))
    feat_key = str(get_by_path(cfg, "training.feat_key", "0"))
    align_to_cache_grid = bool(get_by_path(cfg, "training.align_to_cache_grid", False))
    win = int(get_by_path(cfg, "training.win", 2))
    teacher_mode = get_by_path(cfg, "teacher.mode", "soft")
    teacher_topk = int(get_by_path(cfg, "teacher.topk", 3))
    teacher_temp = float(get_by_path(cfg, "teacher.temperature", 0.03))
    conf_thresh = float(get_by_path(cfg, "sampling.conf_thresh", 0.0))
    min_tokens = int(get_by_path(cfg, "sampling.min_tokens", 64))
    sample_strategy = get_by_path(cfg, "sampling.strategy", "confidence")
    sample_mode = get_by_path(cfg, "sampling.mode", "weighted")
    loss_type = get_by_path(cfg, "loss.type", "supcon")
    supcon_weight = float(get_by_path(cfg, "loss.supcon_weight", 0.7))
    supcon_chunk = int(get_by_path(cfg, "loss.supcon_chunk", 0))
    temperature = float(get_by_path(cfg, "training.temperature", 0.07))

    for epoch in range(start_epoch, epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True) if use_tqdm else dl
        epoch_t0 = time.time()
        ep_skips = 0
        ep_loss_sum = 0.0
        ep_diag_sum = 0.0
        ep_phrase_sum = 0.0
        ep_group_sum = 0.0
        ep_n = 0
        ep_phrase_n = 0
        ep_group_n = 0
        for batch in pbar:
            paths, enc_x, topi, topv, (h, w) = batch
            enc_x = enc_x.to(device, non_blocking=True)
            topi = topi.to(device, non_blocking=True)
            topv = topv.to(device, non_blocking=True)

            if torch.is_tensor(h):
                h = int(h.flatten()[0].item())
            else:
                h = int(h)
            if torch.is_tensor(w):
                w = int(w.flatten()[0].item())
            else:
                w = int(w)

            if train_encoder:
                feats: OrderedDict = encoder(enc_x)
            else:
                with torch.no_grad():
                    feats: OrderedDict = encoder(enc_x)

            feat = feats[feat_key]
            if align_to_cache_grid:
                feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)

            bsz, _, hf, wf = feat.shape
            n_cache = h * w

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                v_tokens = adapter(feat)
                v_tokens = F.normalize(v_tokens, dim=-1)

                if teacher_mode == "hard":
                    s_tokens = llm_phrase[topi[..., 0]]
                else:
                    k_total = topi.size(-1)
                    k = min(teacher_topk, k_total)
                    w_logits = topv[..., :k] / max(teacher_temp, 1e-6)
                    w_soft = F.softmax(w_logits, dim=-1)
                    phrases = llm_phrase[topi[..., :k]]
                    s_tokens = (w_soft.unsqueeze(-1) * phrases).sum(dim=-2)

                s_tokens = F.normalize(s_tokens, dim=-1)

                if v_tokens.size(1) != s_tokens.size(1):
                    sd = s_tokens.transpose(1, 2).reshape(bsz, llm_dim, h, w)
                    sd = F.interpolate(sd, size=(hf, wf), mode="bilinear", align_corners=False)
                    s_tokens = sd.flatten(2).transpose(1, 2).contiguous()
                    s_tokens = F.normalize(s_tokens, dim=-1)

                conf_cache = topv.max(dim=-1).values
                conf_map = conf_cache.reshape(bsz, 1, h, w)
                if v_tokens.size(1) != n_cache:
                    conf_map = F.interpolate(conf_map, size=(hf, wf), mode="bilinear", align_corners=False)
                conf_map = conf_map.squeeze(1)

                if sample_strategy == "random":
                    idx_batch = []
                    for b in range(bsz):
                        idx_b = window_sample_indices(hf, wf, win=win, seed=seed + global_step * 131 + b)
                        idx_batch.append(idx_b)
                    idx_batch = torch.stack(idx_batch, dim=0).to(device)
                else:
                    idx_batch = []
                    for b in range(bsz):
                        idx_b = window_sample_indices_by_conf(
                            conf_map[b], win=win, mode=sample_mode, seed=seed + global_step * 131 + b
                        )
                        idx_batch.append(idx_b)
                    idx_batch = torch.stack(idx_batch, dim=0)

                n_win = idx_batch.size(1)
                gather_index = idx_batch.unsqueeze(-1).expand(bsz, n_win, llm_dim)
                v_sel = v_tokens.gather(dim=1, index=gather_index)
                s_sel = s_tokens.gather(dim=1, index=gather_index)

                conf_flat = conf_map.flatten(1)
                conf_sel = conf_flat.gather(1, idx_batch)
                keep = (conf_sel.reshape(-1) >= conf_thresh)

                v_s = v_sel.reshape(-1, llm_dim)[keep]
                s_s = s_sel.reshape(-1, llm_dim)[keep]

                y_train = None
                if v_tokens.size(1) == topi.size(1):
                    y_train = topi[..., 0].gather(1, idx_batch).reshape(-1)[keep]

                if v_s.size(0) < min_tokens:
                    ep_skips += 1
                    if log_enable and (global_step % log_every == 0):
                        kept_tokens = int(v_s.size(0))
                        total_tokens = int(v_sel.numel() // llm_dim)
                        lr_now = float(optim.param_groups[0]["lr"]) if optim.param_groups else None
                        if logger:
                            logger.info(
                                f"step={global_step} SKIP kept_tokens={kept_tokens} total_tokens={total_tokens} "
                                f"min_tokens={min_tokens} lr={lr_now}"
                            )
                        log_json({
                            "event": "skip",
                            "epoch": int(epoch),
                            "step": int(global_step),
                            "kept_tokens": kept_tokens,
                            "total_tokens": total_tokens,
                            "min_tokens": int(min_tokens),
                            "lr": lr_now,
                        })
                    global_step += 1
                    continue

                v_s = v_s - v_s.mean(dim=0, keepdim=True)
                s_s = s_s - s_s.mean(dim=0, keepdim=True)
                v_s = F.normalize(v_s, dim=-1)
                s_s = F.normalize(s_s, dim=-1)

                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        vn = v_s.float()
                        sn = s_s.float()
                        sim = vn @ sn.t()
                        pred_inbatch = sim.argmax(dim=1)
                        diag_top1_acc = (
                            pred_inbatch == torch.arange(sim.size(0), device=sim.device)
                        ).float().mean().item()

                        phrase_top1_acc = -1.0
                        if y_train is not None:
                            sim_phrase = vn @ llm_phrase.float().t()
                            pred_phrase = sim_phrase.argmax(dim=1)
                            phrase_top1_acc = (pred_phrase == y_train).float().mean().item()

                        group_correct_acc = -1.0
                        if y_train is not None:
                            group_correct_acc = (y_train[pred_inbatch] == y_train).float().mean().item()

                with torch.cuda.amp.autocast(enabled=False):
                    v_fp = v_s.float()
                    s_fp = s_s.float()
                    if loss_type == "supcon" and (y_train is not None):
                        loss = supervised_xmodal_contrastive(
                            v_fp, s_fp, y_train, temperature=temperature, chunk_size=supcon_chunk
                        )
                    elif loss_type == "mix" and (y_train is not None):
                        loss_sup = supervised_xmodal_contrastive(
                            v_fp, s_fp, y_train, temperature=temperature, chunk_size=supcon_chunk
                        )
                        loss_inf = symmetric_infonce(v_fp, s_fp, temperature=temperature)
                        loss = supcon_weight * loss_sup + (1.0 - supcon_weight) * loss_inf
                    else:
                        loss = symmetric_infonce(v_fp, s_fp, temperature=temperature)

                ep_loss_sum += float(loss.item())
                ep_diag_sum += float(diag_top1_acc)
                ep_n += 1
                if phrase_top1_acc >= 0:
                    ep_phrase_sum += float(phrase_top1_acc)
                    ep_phrase_n += 1
                if group_correct_acc >= 0:
                    ep_group_sum += float(group_correct_acc)
                    ep_group_n += 1

                if log_enable and (global_step % log_every == 0):
                    avg_loss = ep_loss_sum / max(ep_n, 1)
                    avg_diag = ep_diag_sum / max(ep_n, 1)
                    avg_phrase = (ep_phrase_sum / ep_phrase_n) if ep_phrase_n > 0 else -1.0
                    avg_group = (ep_group_sum / ep_group_n) if ep_group_n > 0 else -1.0
                    if use_tqdm:
                        pbar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "diag": f"{avg_diag:.4f}",
                            "phrase": f"{avg_phrase:.4f}" if avg_phrase >= 0 else "NA",
                            "group": f"{avg_group:.4f}" if avg_group >= 0 else "NA",
                            "step": global_step,
                        })
                        tqdm.write(
                            f"step={global_step} loss={loss.item():.4f} diag={diag_top1_acc:.4f} "
                            f"phrase={phrase_top1_acc:.4f} group={group_correct_acc:.4f} feat={hf}x{wf}"
                        )
                    if logger:
                        kept_tokens = int(v_s.size(0))
                        total_tokens = int(v_sel.numel() // llm_dim)
                        lr_now = float(optim.param_groups[0]["lr"]) if optim.param_groups else None
                        logger.info(
                            f"step={global_step} loss={loss.item():.4f} diag={diag_top1_acc:.4f} "
                            f"phrase={phrase_top1_acc:.4f} group={group_correct_acc:.4f} "
                            f"kept={kept_tokens}/{total_tokens} lr={lr_now}"
                        )
                    log_json({
                        "event": "step",
                        "epoch": int(epoch),
                        "step": int(global_step),
                        "lr": float(optim.param_groups[0]["lr"]) if optim.param_groups else None,
                        "loss": float(loss.item()),
                        "diag": float(diag_top1_acc),
                        "phrase": float(phrase_top1_acc),
                        "group": float(group_correct_acc),
                        "avg_loss": float(avg_loss),
                        "avg_diag": float(avg_diag),
                        "avg_phrase": float(avg_phrase),
                        "avg_group": float(avg_group),
                    })

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if scheduler and scheduler_step_on == "step":
                scheduler.step()

            global_step += 1

        epoch_sec = time.time() - epoch_t0
        avg_loss = ep_loss_sum / max(ep_n, 1)
        avg_diag = ep_diag_sum / max(ep_n, 1)
        avg_phrase = (ep_phrase_sum / ep_phrase_n) if ep_phrase_n > 0 else -1.0
        avg_group = (ep_group_sum / ep_group_n) if ep_group_n > 0 else -1.0
        msg = (
            f"[epoch {epoch + 1}/{epochs}] loss={avg_loss:.4f} diag={avg_diag:.4f} "
            f"phrase={avg_phrase:.4f} group={avg_group:.4f} skips={ep_skips} time_sec={epoch_sec:.1f}"
        )
        if use_tqdm:
            tqdm.write(msg)
        else:
            print(msg)

        if logger:
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

        if scheduler and scheduler_step_on == "epoch":
            scheduler.step()

        ckpt = {
            "adapter": adapter.state_dict(),
            "config": cfg,
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict() if (use_amp and device.type == "cuda") else None,
            "epoch": epoch + 1,
            "global_step": global_step,
            "rng": {
                "py": random.getstate(),
                "np": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        if train_encoder:
            ckpt["encoder"] = encoder.state_dict()
        if scheduler:
            ckpt["scheduler"] = scheduler.state_dict()

        save_path = os.path.join(output_dir, f"stage1_cached_epoch{epoch + 1}.pt")
        torch.save(ckpt, save_path)
        print(f"[ckpt] saved: {save_path}")
        if logger:
            logger.info(f"[ckpt] saved: {save_path}")
        log_json({"event": "ckpt", "epoch": int(epoch), "path": str(save_path)})

    if jsonl_f:
        try:
            jsonl_f.flush()
            jsonl_f.close()
        except Exception:
            pass
    print(f"Done. Saved to {output_dir}")


if __name__ == "__main__":
    main()
