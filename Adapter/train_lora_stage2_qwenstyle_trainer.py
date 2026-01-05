#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-style LoRA fine-tuning for a custom concrete-domain Vision Encoder + Adapter + Qwen LLM.

Key goals vs. the original DDP script:
  - Keep your custom Vision Encoder + Adapter modules, but (optionally) FREEZE them and train ONLY LoRA on the LLM.
  - Use the Transformers Trainer stack (the same "shape" used by most Qwen/Transformers fine-tuning recipes),
    so you can rely on mature features: torchrun DDP, gradient accumulation, logging/saving strategies, Deepspeed, etc.
  - Preserve your current dataset format: JSONL with keys: image_path, system, user, assistant.

Run (multi-GPU):
  torchrun --nproc_per_node 8 train_lora_stage2_qwenstyle_trainer.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --train_jsonl /path/to/train.jsonl \
    --output_dir /path/to/out \
    --freeze_vision --freeze_adapter \
    --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --num_train_epochs 1 --learning_rate 2e-4 --bf16

Notes:
  - If you want to also train Adapter and/or Encoder, pass --no-freeze_adapter and/or --no-freeze_vision
    plus corresponding --adapter_lr / --encoder_lr.
"""

from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import math
import argparse
from dataclasses import dataclass

import numpy as np
from PIL import Image
from typing import Any, Dict, List, Optional

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
from peft import LoraConfig, get_peft_model

# ---- Your custom modules (keep the import paths consistent with your project) ----
# These are referenced in your current script.
from models.dem_encoder import DEMEncoderConfig, DEMVisionBackbone
from models.backbone import ResNetPyramidBackbone
from models.da_adapter import DAAdapter, DAAdapterConfig


# -----------------------------
# Dataset & Collator
# -----------------------------
class JsonlVLDataset(Dataset):
    """
    Expects each line in JSONL has:
      - image_path: str
      - system: str
      - user: str
      - assistant: str
    """
    def __init__(self, jsonl_path: str):
        super().__init__()
        self.items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


# -------------------------
# Utils: image preprocessing (match DEMEncoderConfig)
# -------------------------
def pil_resize_square(img: Image.Image, size: int) -> Image.Image:
    """Resize to fixed-size square."""
    return img.resize((size, size), resample=Image.BICUBIC)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to (3,H,W) float tensor in [0,1]."""
    arr = np.array(img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def normalize(t: torch.Tensor, mean, std) -> torch.Tensor:
    """Channel-wise normalize."""
    m = torch.tensor(mean, dtype=t.dtype).view(3, 1, 1)
    s = torch.tensor(std, dtype=t.dtype).view(3, 1, 1)
    return (t - m) / s


@dataclass
class DataCollatorVisionPrefix:
    tokenizer: Any
    image_size: int = 448
    max_length: int = 2048
    ignore_index: int = -100
    image_mean: tuple = (0.485, 0.456, 0.406)
    image_std: tuple = (0.229, 0.224, 0.225)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        attention_mask_list: List[torch.Tensor] = []
        images_list: List[torch.Tensor] = []

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # Qwen tokenizers usually have pad, but be defensive.
            self.tokenizer.pad_token = self.tokenizer.eos_token
            pad_id = self.tokenizer.pad_token_id

        for ex in batch:
            system = ex.get("system", "")
            user = ex["user"]
            assistant = ex["assistant"]
            img_path = ex["image_path"]

            # Chat template (Qwen-style) with exact assistant-span masking:
            #   1) tokenize "system+user (+generation prompt)" to get the boundary
            #   2) tokenize full (including assistant answer) for input_ids/labels
            sys_user_msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            full_msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]

            sys_user_prompt = self.tokenizer.apply_chat_template(
                sys_user_msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_prompt = self.tokenizer.apply_chat_template(
                full_msgs,
                tokenize=False,
                add_generation_prompt=False,
            )

            sys_user_ids = self.tokenizer(
                sys_user_prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )["input_ids"][0]
            tok = self.tokenizer(
                full_prompt,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = tok["input_ids"][0]
            attention_mask = tok["attention_mask"][0]

            labels = input_ids.clone()
            prompt_len = min(sys_user_ids.size(0), labels.size(0))
            labels[:prompt_len] = self.ignore_index

            img = Image.open(img_path).convert("RGB")
            img = pil_resize_square(img, self.image_size)
            img_tensor = normalize(pil_to_tensor(img), self.image_mean, self.image_std)

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
            images_list.append(img_tensor)

        # Pad to max in batch
        max_len = max(x.size(0) for x in input_ids_list)
        def pad_1d(x: torch.Tensor, value: int) -> torch.Tensor:
            if x.size(0) == max_len:
                return x
            return F.pad(x, (0, max_len - x.size(0)), value=value)

        input_ids = torch.stack([pad_1d(x, pad_id) for x in input_ids_list])
        labels = torch.stack([pad_1d(x, self.ignore_index) for x in labels_list])
        attention_mask = torch.stack([pad_1d(x, 0) for x in attention_mask_list])
        images = torch.stack(images_list)  # (B,3,H,W)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "images": images,
        }


# -----------------------------
# Model: your encoder + adapter, LoRA only on LLM
# -----------------------------
class VisionPrefixQwen(nn.Module):
    def __init__(
        self,
        base_model: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: str,
        freeze_vision: bool,
        freeze_adapter: bool,
        stage1_adapter_ckpt: Optional[str],
        stage1_adapter_map_location: str,
        encoder_ckpt: Optional[str],
        encoder_map_location: str,
        image_size: int = 448,
        max_vision_tokens: int = 196,
        vision_dim: int = 512,
        adapter_dim: int = 512,
        adapter_out_dim: Optional[int] = None,
        torch_dtype: str = "auto",
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()

        # 1) Load LLM
        model_kwargs = {}
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self.llm = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        self.config = self.llm.config
        self.hidden_size = getattr(self.llm.config, "hidden_size", None)

        # 2) Your vision encoder + adapter
        enc_cfg = DEMEncoderConfig()
        pyramid = ResNetPyramidBackbone(name="resnet50")
        self.encoder = DEMVisionBackbone(pyramid_backbone=pyramid, cfg=enc_cfg,)
        out_dim = adapter_out_dim if adapter_out_dim is not None else self.hidden_size
        if out_dim is None:
            raise ValueError("Could not infer LLM hidden size; please set --adapter_out_dim explicitly.")
        llm_dim=self.llm.get_input_embeddings().weight.shape[1]
        self.adapter = DAAdapter(DAAdapterConfig(in_channels=self.encoder.out_channels, llm_dim=llm_dim)).to(device)

        # Load stage-1 Adapter weights (optional)
        if stage1_adapter_ckpt:
            sd = torch.load(stage1_adapter_ckpt, map_location=stage1_adapter_map_location)
            self.adapter.load_state_dict(sd, strict=True)

        # Load encoder weights (optional)
        if encoder_ckpt:
            sd = torch.load(encoder_ckpt, map_location=encoder_map_location)
            self.encoder.load_state_dict(sd, strict=False)

        # 3) Apply LoRA ONLY to the LLM
        if lora_rank and lora_rank > 0:
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(self.llm, lora_cfg)
            self.llm.print_trainable_parameters()

        # 4) Freeze vision / adapter if requested
        if freeze_vision:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        if freeze_adapter:
            for p in self.adapter.parameters():
                p.requires_grad = False
            self.adapter.eval()

        # Always disable KV cache during training
        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = False

    @torch.no_grad()
    def _vision_forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns prefix embeddings shaped (B, V, H).
        """
        feats = self.encoder(images)  # expects dict with "tokens": (B, V, adapter_dim)
        tokens = feats["tokens"]
        embeds = self.adapter(tokens)  # (B, V, hidden_size)
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Prefix the text embeddings with visual embeddings, then compute standard causal LM loss.
        """
        # Text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        if images is not None:
            # allow gradients only if vision/adapter trainable
            if any(p.requires_grad for p in self.encoder.parameters()) or any(p.requires_grad for p in self.adapter.parameters()):
                vision_embeds = self.adapter(self.encoder(images)["tokens"])
            else:
                vision_embeds = self._vision_forward(images)

            inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

            # Extend attention + labels
            B, V, _ = vision_embeds.shape
            vision_attn = torch.ones((B, V), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([vision_attn, attention_mask], dim=1)

            if labels is not None:
                vision_labels = torch.full((B, V), -100, device=labels.device, dtype=labels.dtype)
                labels = torch.cat([vision_labels, labels], dim=1)
        else:
            inputs_embeds = text_embeds

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        # Transformers Trainer expects loss in dict/attr
        return {"loss": out.loss}

    def save_pretrained(self, save_directory: str, save_mm: bool = False, **kwargs):
        """
        Save:
          - LoRA adapter (or full LLM) via llm.save_pretrained(...)
          - optionally your frozen MM weights to mm.pt (so you can reproduce exact inference)
        """
        os.makedirs(save_directory, exist_ok=True)
        self.llm.save_pretrained(os.path.join(save_directory, "llm"), **kwargs)
        if save_mm:
            torch.save(
                {
                    "encoder": self.encoder.state_dict(),
                    "adapter": self.adapter.state_dict(),
                },
                os.path.join(save_directory, "mm.pt"),
            )


class MultiGroupTrainer(Trainer):
    """
    Trainer that supports different LRs for:
      - LLM (LoRA)
      - Adapter
      - Encoder
    If Adapter/Encoder are frozen, they won't be in the optimizer.
    """
    def __init__(self, *args, llm_lr: float, adapter_lr: float, encoder_lr: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._llm_lr = llm_lr
        self._adapter_lr = adapter_lr
        self._encoder_lr = encoder_lr

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        # Build param groups by module ownership
        llm_params, adapter_params, encoder_params = [], [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("llm."):
                llm_params.append(p)
            elif n.startswith("adapter."):
                adapter_params.append(p)
            elif n.startswith("encoder."):
                encoder_params.append(p)
            else:
                # fallback: treat as LLM
                llm_params.append(p)

        groups = []
        if llm_params:
            groups.append({"params": llm_params, "lr": self._llm_lr})
        if adapter_params:
            groups.append({"params": adapter_params, "lr": self._adapter_lr})
        if encoder_params:
            groups.append({"params": encoder_params, "lr": self._encoder_lr})

        if not groups:
            raise RuntimeError("No trainable parameters found. Check your freeze flags / LoRA config.")

        from torch.optim import AdamW
        self.optimizer = AdamW(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.weight_decay)
        return self.optimizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--max_length", type=int, default=2048)

    # Base model
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--attn_implementation", type=str, default=None, help="e.g., flash_attention_2 (if available)")

    # Stage-1 weights
    p.add_argument("--stage1_adapter_ckpt", type=str, default=None)
    p.add_argument("--stage1_adapter_map_location", type=str, default="cpu")
    p.add_argument("--encoder_ckpt", type=str, default=None)
    p.add_argument("--encoder_map_location", type=str, default="cpu")

    # Freeze controls (default: freeze both, train LoRA only)
    p.add_argument("--freeze_vision", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze_adapter", action=argparse.BooleanOptionalAction, default=True)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # For Qwen-family models, "all-linear" is common when LoRA is applied ONLY to the LLM.
    p.add_argument("--lora_target_modules", type=str, default="all-linear")

    # Optim
    p.add_argument("--learning_rate", type=float, default=2e-4, help="LR for LLM (LoRA)")
    p.add_argument("--adapter_lr", type=float, default=2e-4)
    p.add_argument("--encoder_lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)

    # Trainer
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--save_mm", action="store_true", help="Also save encoder+adapter state_dict to mm.pt")
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # Precision
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    # Deepspeed (optional)
    p.add_argument("--deepspeed", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = JsonlVLDataset(args.train_jsonl)
    enc_cfg = DEMEncoderConfig()
    collator = DataCollatorVisionPrefix(
        tokenizer=tokenizer,
        image_size=args.image_size,
        max_length=args.max_length,
        image_mean=getattr(enc_cfg, "image_mean", (0.485, 0.456, 0.406)),
        image_std=getattr(enc_cfg, "image_std", (0.229, 0.224, 0.225)),
    )

    model = VisionPrefixQwen(
        base_model=args.base_model,
        image_size=args.image_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        freeze_vision=args.freeze_vision,
        freeze_adapter=args.freeze_adapter,
        stage1_adapter_ckpt=args.stage1_adapter_ckpt,
        stage1_adapter_map_location=args.stage1_adapter_map_location,
        encoder_ckpt=args.encoder_ckpt,
        encoder_map_location=args.encoder_map_location,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    # TrainingArguments: set remove_unused_columns=False so "images" survives into model.forward(...)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,  # used if you don't override optimizer; we override anyway
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
        report_to="none",
        deepspeed=args.deepspeed,
    )

    trainer = MultiGroupTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        llm_lr=args.learning_rate,
        adapter_lr=args.adapter_lr,
        encoder_lr=args.encoder_lr,
    )

    trainer.train()

    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if args.save_mm and trainer.is_world_process_zero():
        # Save multimodal weights for exact reproducibility.
        model.save_pretrained(args.output_dir, save_mm=True)

    # Also write a small manifest for reproducibility
    if trainer.is_world_process_zero():
        with open(os.path.join(args.output_dir, "run_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
