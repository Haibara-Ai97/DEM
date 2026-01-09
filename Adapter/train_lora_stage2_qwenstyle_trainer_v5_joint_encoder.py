#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 LoRA fine-tuning (Trainer-based, Qwen2.5-7B LLM) with joint tuning options:
  - Your custom Vision Encoder (DEMVisionBackbone) : frozen by default, can be unfrozen
  - Your custom DAAdapter                         : frozen by default, can be unfrozen
  - Qwen2.5-7B-Instruct LLM + LoRA                : trainable (LoRA params)

This file fixes two critical issues that can make training "not start":
  1) All training logic is inside main() (no accidental dedent/early-return).
  2) VisionPrefixQwen properly defines train() and forward() as class methods.

JSONL format per line:
  {"image_path": "...", "system": "...", "user": "...", "assistant": "...", ...}

Example:
  torchrun --nproc_per_node=8 train_lora_stage2_qwenstyle_trainer_v5_joint_encoder.py \
    --train_jsonl /path/train.jsonl \
    --output_dir /path/out \
    --llm_name Qwen/Qwen2.5-7B-Instruct \
    --stage1_ckpt /path/stage1.pt \
    --encoder_ckpt /path/encoder_best.pth \
    --use_lora \
    --no_freeze_encoder --encoder_lr 2e-5 \
    --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --max_text_len 512 --bf16
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType

# ---- Your project modules (same import paths as your existing script) ----
from models.dem_encoder import DEMEncoderConfig, DEMVisionBackbone
from models.backbone import ResNetPyramidBackbone
from models.da_adapter import DAAdapter, DAAdapterConfig

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -------------------------
# Image preprocessing (match DEMEncoderConfig)
# -------------------------
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


def freeze_batchnorm_stats(module: nn.Module) -> None:
    """Freeze BatchNorm running stats by forcing BN layers to eval()."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# -------------------------
# Checkpoint loading (robust, suffix matching)
# -------------------------
def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if not isinstance(sd, dict):
        return sd
    if any(k.startswith(prefix) for k in sd.keys()):
        return {k.replace(prefix, "", 1): v for k, v in sd.items()}
    return sd


def _is_tensor_state_dict(d: Any) -> bool:
    return isinstance(d, dict) and len(d) > 0 and all(isinstance(v, torch.Tensor) for v in d.values())


def _extract_state_dict(ckpt: Any, key: Optional[str] = None, fallback_keys: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    if fallback_keys is None:
        fallback_keys = ["state_dict", "model", "net", "ema", "encoder", "adapter"]

    if _is_tensor_state_dict(ckpt):
        return ckpt

    if isinstance(ckpt, dict):
        if key and key in ckpt and isinstance(ckpt[key], dict) and _is_tensor_state_dict(ckpt[key]):
            return ckpt[key]
        for k in fallback_keys:
            if k in ckpt and isinstance(ckpt[k], dict) and _is_tensor_state_dict(ckpt[k]):
                return ckpt[k]

        tensor_items = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        if len(tensor_items) > 0:
            return tensor_items

    raise ValueError("Cannot find a valid state_dict in the checkpoint.")


def load_encoder_ckpt_by_suffix(
    encoder: nn.Module,
    ckpt_path: str,
    map_location: str = "cpu",
    ckpt_key: Optional[str] = None,
) -> None:
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    sd = _extract_state_dict(ckpt, key=ckpt_key, fallback_keys=["state_dict", "model", "net", "ema", "encoder"])
    sd = _strip_prefix(sd, "module.")

    msd = encoder.state_dict()
    loadable: Dict[str, torch.Tensor] = {}

    # exact match
    for k, v in sd.items():
        if k in msd and msd[k].shape == v.shape:
            loadable[k] = v

    # suffix match
    if len(loadable) < int(len(msd) * 0.8):
        sd_keys = list(sd.keys())
        for mk in msd.keys():
            if mk in loadable:
                continue
            candidates = [ck for ck in sd_keys if ck.endswith(mk)]
            if not candidates:
                continue
            ck_best = min(candidates, key=len)
            v = sd[ck_best]
            if msd[mk].shape == v.shape:
                loadable[mk] = v

    encoder.load_state_dict(loadable, strict=False)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[encoder load] {ckpt_path} | loadable: {len(loadable)}/{len(msd)}", flush=True)


def load_adapter_from_stage1(
    adapter: nn.Module,
    stage1_ckpt_path: str,
    map_location: str = "cpu",
    key: str = "adapter",
    strict: bool = True,
) -> None:
    ckpt = torch.load(stage1_ckpt_path, map_location=map_location, weights_only=False)
    sd = _extract_state_dict(ckpt, key=key, fallback_keys=[key, "state_dict", "model", "net", "ema", "adapter"])
    sd = _strip_prefix(sd, "module.")
    adapter.load_state_dict(sd, strict=strict)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[adapter load] {stage1_ckpt_path} | key='{key}' | strict={strict}", flush=True)


# -------------------------
# Dataset
# -------------------------
class JsonlSFTDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        ex = self.items[idx]
        img = Image.open(ex["image_path"]).convert("RGB")
        return ex, img


def build_input_and_labels(tokenizer, system: str, user: str, assistant: str, max_len: int):
    """
    Build full ids and prompt ids; labels mask prompt tokens.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        msgs_full = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        msgs_prompt = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        full_ids = tokenizer.apply_chat_template(
            msgs_full, tokenize=True, add_generation_prompt=False, return_tensors="pt"
        )[0]
        prompt_ids = tokenizer.apply_chat_template(
            msgs_prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )[0]

        full_ids = full_ids[:max_len]
        attn = torch.ones_like(full_ids, dtype=torch.long)

        labels = full_ids.clone()
        pl = min(prompt_ids.numel(), labels.numel())
        labels[:pl] = -100
        return full_ids, attn, labels

    # fallback
    text = f"System: {system}\nUser: {user}\nAssistant: {assistant}"
    enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    attn = enc["attention_mask"][0]
    labels = input_ids.clone()
    return input_ids, attn, labels


@dataclass
class DataCollatorSFT:
    tokenizer: Any
    enc_cfg: DEMEncoderConfig
    image_size: int
    max_text_len: int

    def __call__(self, batch):
        exs, imgs = zip(*batch)

        enc_imgs = [pil_resize_square(im, self.image_size) for im in imgs]
        image_tensor = torch.stack(
            [normalize(pil_to_tensor(im), self.enc_cfg.image_mean, self.enc_cfg.image_std) for im in enc_imgs],
            dim=0,
        )

        input_ids_list, attn_list, labels_list = [], [], []
        for ex in exs:
            inp, attn, lab = build_input_and_labels(
                self.tokenizer,
                ex.get("system", ""),
                ex["user"],
                ex["assistant"],
                self.max_text_len,
            )
            input_ids_list.append(inp)
            attn_list.append(attn)
            labels_list.append(lab)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        maxL = max(x.numel() for x in input_ids_list)

        def pad1(x, fill):
            out = torch.full((maxL,), fill_value=fill, dtype=x.dtype)
            out[: x.numel()] = x
            return out

        input_ids = torch.stack([pad1(x, pad_id) for x in input_ids_list], dim=0)
        attention_mask = torch.stack([pad1(x, 0) for x in attn_list], dim=0)
        labels = torch.stack([pad1(x, -100) for x in labels_list], dim=0)

        return {
            "images": image_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# -------------------------
# Model: vision prefix + LLM
# -------------------------
class VisionPrefixQwen(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        adapter: nn.Module,
        llm: nn.Module,
        feat_key: str = "0",
        prefix_grid: int = 8,
        freeze_encoder_bn: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.llm = llm
        self.feat_key = feat_key
        self.prefix_grid = prefix_grid
        self.freeze_encoder_bn = freeze_encoder_bn

    def train(self, mode: bool = True):
        super().train(mode)

        # keep frozen encoder/adapter deterministic
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        else:
            if self.freeze_encoder_bn:
                freeze_batchnorm_stats(self.encoder)

        if not any(p.requires_grad for p in self.adapter.parameters()):
            self.adapter.eval()

        return self

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        feats = self.encoder(images)
        feat = feats[self.feat_key]
        feat = F.adaptive_avg_pool2d(feat, output_size=(self.prefix_grid, self.prefix_grid))
        vtok = self.adapter(feat)  # (B, N, D)

        tok_emb = self.llm.get_input_embeddings()(input_ids)  # (B,L,D)
        vtok = vtok.to(dtype=tok_emb.dtype)

        inputs_embeds = torch.cat([vtok, tok_emb], dim=1)

        B, N, _ = vtok.shape
        prefix_mask = torch.ones((B, N), device=attention_mask.device, dtype=attention_mask.dtype)
        attn = torch.cat([prefix_mask, attention_mask], dim=1)

        lab = None
        if labels is not None:
            prefix_labels = torch.full((B, N), -100, device=labels.device, dtype=labels.dtype)
            lab = torch.cat([prefix_labels, labels], dim=1)

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=lab)


# -------------------------
# Trainer with param groups
# -------------------------
class MultiGroupTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        lr = self.args.learning_rate

        adapter_params, encoder_params, other_params = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("adapter.") or ".adapter." in name:
                adapter_params.append(p)
            elif name.startswith("encoder.") or ".encoder." in name:
                encoder_params.append(p)
            else:
                other_params.append(p)

        groups = []
        if other_params:
            groups.append({"params": other_params, "lr": lr})
        if adapter_params:
            groups.append({"params": adapter_params, "lr": getattr(self, "_adapter_lr", lr)})
        if encoder_params:
            groups.append({"params": encoder_params, "lr": getattr(self, "_encoder_lr", lr)})

        from torch.optim import AdamW
        self.optimizer = AdamW(
            groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.weight_decay
        )
        return self.optimizer


def _torch_dtype_from_arg(arg: str):
    if arg == "auto":
        return "auto"
    if arg == "float16":
        return torch.float16
    if arg == "bfloat16":
        return torch.bfloat16
    if arg == "float32":
        return torch.float32
    raise ValueError(f"Unknown torch_dtype: {arg}")


def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_text_len", type=int, default=512)

    # LLM
    p.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default=None)

    # ckpts
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage1_adapter_key", type=str, default="adapter")
    p.add_argument("--adapter_strict", action="store_true")
    p.add_argument("--no_adapter_strict", action="store_false", dest="adapter_strict")
    p.set_defaults(adapter_strict=True)

    p.add_argument("--encoder_ckpt", type=str, default="")
    p.add_argument("--encoder_ckpt_key", type=str, default="")
    p.add_argument("--ckpt_map_location", type=str, default="cpu")

    # Vision config switches
    p.add_argument("--backbone_pretrained", action="store_true", default=True)
    p.add_argument("--no_backbone_pretrained", action="store_false", dest="backbone_pretrained")
    p.add_argument("--disable_dem2", action="store_true")
    p.add_argument("--disable_dem3", action="store_true")
    p.add_argument("--disable_dem4", action="store_true")
    p.add_argument("--disable_dem5", action="store_true")

    # Prefix
    p.add_argument("--feat_key", type=str, default="0", choices=["0", "1", "2", "3"])
    p.add_argument("--prefix_grid", type=int, default=8)

    # Freeze controls
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--no_freeze_encoder", action="store_false", dest="freeze_encoder")
    p.add_argument("--freeze_adapter", action="store_true")
    p.add_argument("--no_freeze_adapter", action="store_false", dest="freeze_adapter")
    p.set_defaults(freeze_encoder=True, freeze_adapter=True)

    p.add_argument("--freeze_encoder_bn", action="store_true")
    p.add_argument("--no_freeze_encoder_bn", action="store_false", dest="freeze_encoder_bn")
    p.set_defaults(freeze_encoder_bn=True)

    p.add_argument("--encoder_train_regex", type=str, default="")

    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target", type=str, default="qkvomlp",
                   help="qv|qkv|qkvo|qkvomlp (default)")

    # Optim/Trainer
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--adapter_lr", type=float, default=1e-4)
    p.add_argument("--encoder_lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--max_grad_norm", type=float, default=0.3)

    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--deepspeed", type=str, default=None)

    return p.parse_args()


def _pick_lora_targets(mode: str) -> List[str]:
    mode = (mode or "").lower()
    if mode == "qv":
        return ["q_proj", "v_proj"]
    if mode == "qkv":
        return ["q_proj", "k_proj", "v_proj"]
    if mode == "qkvo":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    # default: qkv + o + mlp
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _count_trainable(model: nn.Module) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    rank0 = int(os.environ.get("RANK", "0")) == 0

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LLM
    model_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    td = _torch_dtype_from_arg(args.torch_dtype)
    if td != "auto":
        model_kwargs["torch_dtype"] = td
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    llm = AutoModelForCausalLM.from_pretrained(args.llm_name, **model_kwargs)

    # LoRA
    if args.use_lora:
        targets = _pick_lora_targets(args.lora_target)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
        )
        llm = get_peft_model(llm, lora_cfg)

    # Vision encoder
    enc_cfg = DEMEncoderConfig()
    pyramid = ResNetPyramidBackbone(name="resnet50", pretrained=args.backbone_pretrained)
    encoder = DEMVisionBackbone(
        pyramid_backbone=pyramid,
        cfg=enc_cfg,
        disable_dem2=args.disable_dem2,
        disable_dem3=args.disable_dem3,
        disable_dem4=args.disable_dem4,
        disable_dem5=args.disable_dem5,
    )

    if args.encoder_ckpt:
        key = args.encoder_ckpt_key.strip() or None
        load_encoder_ckpt_by_suffix(encoder, args.encoder_ckpt, map_location=args.ckpt_map_location, ckpt_key=key)

    # Adapter
    llm_dim = llm.get_input_embeddings().weight.shape[1]
    adapter = DAAdapter(DAAdapterConfig(in_channels=encoder.out_channels, llm_dim=llm_dim))
    load_adapter_from_stage1(
        adapter,
        args.stage1_ckpt,
        map_location=args.ckpt_map_location,
        key=args.stage1_adapter_key,
        strict=args.adapter_strict,
    )

    # Freeze / unfreeze
    if args.freeze_encoder:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
    else:
        encoder.train()
        for n, p in encoder.named_parameters():
            p.requires_grad = True
        if args.encoder_train_regex.strip():
            rgx = re.compile(args.encoder_train_regex.strip())
            for n, p in encoder.named_parameters():
                p.requires_grad = bool(rgx.search(n))
        if args.freeze_encoder_bn:
            freeze_batchnorm_stats(encoder)

    if args.freeze_adapter:
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad = False
    else:
        adapter.train()
        for p in adapter.parameters():
            p.requires_grad = True

    # Wrap
    mm_model = VisionPrefixQwen(
        encoder=encoder,
        adapter=adapter,
        llm=llm,
        feat_key=args.feat_key,
        prefix_grid=args.prefix_grid,
        freeze_encoder_bn=args.freeze_encoder_bn,
    )

    # Data
    train_ds = JsonlSFTDataset(args.train_jsonl)
    if rank0:
        print(f"[data] train size = {len(train_ds)}", flush=True)

    collator = DataCollatorSFT(
        tokenizer=tokenizer,
        enc_cfg=enc_cfg,
        image_size=args.image_size,
        max_text_len=args.max_text_len,
    )

    # TrainingArguments
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=[],
        deepspeed=args.deepspeed,
    )

    # Trainer (tokenizer kwarg compatibility)
    import inspect as _inspect
    _trainer_kwargs = dict(
        model=mm_model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
    )
    _sig = _inspect.signature(Trainer.__init__)
    if "tokenizer" in _sig.parameters:
        _trainer_kwargs["tokenizer"] = tokenizer

    trainer = MultiGroupTrainer(**_trainer_kwargs)
    trainer._adapter_lr = args.adapter_lr
    trainer._encoder_lr = args.encoder_lr

    if rank0:
        trn, tot = _count_trainable(mm_model)
        print(f"[trainable] {trn:,} / {tot:,} ({100.0*trn/tot:.4f}%)", flush=True)
        print("[train] calling trainer.train() ...", flush=True)

    trainer.train()

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)

        torch.save({"adapter": adapter.state_dict()}, Path(args.output_dir) / "da_adapter.pt")
        if any(p.requires_grad for p in encoder.parameters()):
            torch.save({"encoder": encoder.state_dict()}, Path(args.output_dir) / "dem_encoder.pt")

        # Save LoRA/LLM
        if args.use_lora and hasattr(llm, "save_pretrained"):
            llm.save_pretrained(Path(args.output_dir) / "lora_llm")

        print(f"[save] outputs written to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
