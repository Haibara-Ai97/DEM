import argparse
import json
import os
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

from models.dem_encoder import DEMEncoderConfig, DEMVisionBackbone
from models.backbone import ResNetPyramidBackbone
from models.da_adapter import DAAdapter, DAAdapterConfig

# Avoid tokenizers fork warnings in multi-worker settings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# -------------------------
# Utils: image preprocessing (match DEMEncoderConfig)
# -------------------------
def pil_resize_square(img: Image.Image, size: int) -> Image.Image:
    """
    Resize the image to the fixed-size square.
    """
    return img.resize((size, size), resample=Image.BICUBIC)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL image to Tensor.
    """
    arr = np.array(img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def normalize(t: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    """
    Construct mean/std tensor and reshape it to (3,1,1) to channel-wise broadcasting.
    """
    m = torch.tensor(mean, dtype=t.dtype).view(3, 1, 1)
    s = torch.tensor(std, dtype=t.dtype).view(3, 1, 1)
    return (t - m) / s


# -------------------------
# Dataset
# -------------------------
class JsonlSFTDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        ex = self.items[idx]
        img = Image.open(ex["image_path"]).convert("RGB")
        return ex, img


def build_input_and_labels(tokenizer, system: str, user: str, assistant: str, max_len: int):
    """
    Prefer tokenizer.apply_chat_template if available.
    Create labels masking system+user (prompt) tokens, only learn assistant tokens.
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

        # truncate
        full_ids = full_ids[:max_len]
        attn = torch.ones_like(full_ids, dtype=torch.long)

        labels = full_ids.clone()
        # mask prompt region (system+user and assistant prefix)
        pl = min(prompt_ids.numel(), labels.numel())
        labels[:pl] = -100
        return full_ids, attn, labels

    # Fallback: simple plain text (less ideal but runnable)
    text = f"System: {system}\nUser: {user}\nAssistant: {assistant}"
    enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    attn = enc["attention_mask"][0]
    labels = input_ids.clone()
    # mask everything before "Assistant:" approximately
    # (fallback is approximate; prefer apply_chat_template)
    return input_ids, attn, labels


def collate_sft(
    batch,
    tokenizer,
    enc_cfg: DEMEncoderConfig,
    image_size: int,
    max_text_len: int,
):
    exs, imgs = zip(*batch)

    # images -> encoder input tensor
    enc_imgs = [pil_resize_square(im, image_size) for im in imgs]
    image_tensor = torch.stack(
        [normalize(pil_to_tensor(im), enc_cfg.image_mean, enc_cfg.image_std) for im in enc_imgs], dim=0
    )

    # build token sequences
    input_ids_list, attn_list, labels_list = [], [], []
    meta = []
    for ex in exs:
        inp, attn, lab = build_input_and_labels(
            tokenizer,
            ex.get("system", ""),
            ex["user"],
            ex["assistant"],
            max_text_len,
        )
        input_ids_list.append(inp)
        attn_list.append(attn)
        labels_list.append(lab)
        meta.append({"task": ex.get("task", ""), "split": ex.get("split", "")})

    # pad
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
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
        "meta": meta,
    }


# -------------------------
# Model: vision prefix + Qwen
# -------------------------
class VisionPrefixQwen(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        adapter: nn.Module,
        llm: nn.Module,
        feat_key: str = "0",
        prefix_grid: int = 8,   # prefix tokens = prefix_grid^2
    ):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.llm = llm
        self.feat_key = feat_key
        self.prefix_grid = prefix_grid

    def forward(self, images, input_ids, attention_mask, labels):
        # images: (B,3,H,W)
        feats = self.encoder(images)  # OrderedDict[str, (B,C,Hf,Wf)]
        feat = feats[self.feat_key]   # (B,C,Hf,Wf)

        # fixed-length vision tokens for stable training
        feat = F.adaptive_avg_pool2d(feat, output_size=(self.prefix_grid, self.prefix_grid))
        vtok = self.adapter(feat)  # (B, N, D), N=prefix_grid^2

        # text embeddings
        tok_emb = self.llm.get_input_embeddings()(input_ids)  # (B,L,D)
        vtok = vtok.to(dtype=tok_emb.dtype)

        # concat prefix + text
        inputs_embeds = torch.cat([vtok, tok_emb], dim=1)

        # attention mask / labels for prefix region
        B, N, D = vtok.shape
        prefix_mask = torch.ones((B, N), device=attention_mask.device, dtype=attention_mask.dtype)
        attn = torch.cat([prefix_mask, attention_mask], dim=1)

        prefix_labels = torch.full((B, N), -100, device=labels.device, dtype=labels.dtype)
        lab = torch.cat([prefix_labels, labels], dim=1)

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=lab)
        return out


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



def _init_distributed(args):
    """
    Initialize torch.distributed from environment variables (torchrun / torch.distributed.launch).
    """
    args.distributed = False
    args.rank = 0
    args.world_size = 1

    # torchrun sets: RANK, LOCAL_RANK, WORLD_SIZE
    if "WORLD_SIZE" in os.environ:
        try:
            args.world_size = int(os.environ["WORLD_SIZE"])
        except ValueError:
            args.world_size = 1

    if args.world_size > 1:
        args.distributed = True
        args.rank = int(os.environ.get("RANK", "0"))
        # prefer env LOCAL_RANK; fall back to argparse --local_rank
        if "LOCAL_RANK" in os.environ:
            args.local_rank = int(os.environ["LOCAL_RANK"])
        elif getattr(args, "local_rank", -1) != -1:
            args.local_rank = int(args.local_rank)
        else:
            args.local_rank = 0

        # backend
        if args.dist_backend:
            backend = args.dist_backend
        else:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        dist.init_process_group(backend=backend, init_method="env://")
        dist.barrier()


def _is_main_process(args) -> bool:
    return (not getattr(args, "distributed", False)) or getattr(args, "rank", 0) == 0


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--valid_jsonl", type=str, default="")
    ap.add_argument("--output_dir", type=str, default="checkpoints/stage2_qwen25")

    ap.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--stage1_ckpt", type=str, required=True,
                    help="Stage-1 checkpoint containing adapter weights (key 'adapter')")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--max_text_len", type=int, default=512)

    ap.add_argument("--feat_key", type=str, default="0", choices=["0","1","2","3"])
    ap.add_argument("--prefix_grid", type=int, default=8, help="vision prefix tokens = prefix_grid^2")

    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--use_lora", action="store_true")

    # LoRA params
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # training params
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2, help="Per-GPU batch size (DDP).")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--adapter_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=20)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--backbone_pretrained", action="store_true", default=True)
    ap.add_argument("--no_backbone_pretrained", action="store_false", dest="backbone_pretrained")

    ap.add_argument("--freeze_llm", action="store_true",
                    help="Freeze base LLM params (useful when not using LoRA).")

    ap.add_argument("--encoder_ckpt", type=str, default="",
                    help="Path to pretrained DEM-Encoder checkpoint used in stage1")
    ap.add_argument("--encoder_ckpt_key", type=str, default="encoder",
                    help="Key name for encoder state dict if ckpt is a dict")
    ap.add_argument("--disable_dem2", action="store_true")
    ap.add_argument("--disable_dem3", action="store_true")
    ap.add_argument("--disable_dem4", action="store_true")
    ap.add_argument("--disable_dem5", action="store_true")

    # Distributed / memory helpers
    ap.add_argument("--local_rank", type=int, default=-1,
                    help="Used by torch.distributed.launch. With torchrun, LOCAL_RANK is taken from env.")
    ap.add_argument("--dist_backend", type=str, default="",
                    help="Distributed backend: nccl (GPU) / gloo (CPU). Default: auto.")
    ap.add_argument("--ddp_find_unused_parameters", action="store_true",
                    help="Enable if you hit DDP errors about unused parameters (slower).")
    ap.add_argument("--grad_checkpointing", action="store_true",
                    help="Enable LLM gradient checkpointing to reduce activation memory (slower).")

    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN", None)

    # Distributed init (torchrun)
    _init_distributed(args)

    # per-rank seed for better shuffling determinism
    _seed_everything(args.seed + getattr(args, "rank", 0))

    # Device / dtype
    if torch.cuda.is_available():
        local_rank = getattr(args, "local_rank", 0) if getattr(args, "distributed", False) else 0
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    if _is_main_process(args):
        os.makedirs(args.output_dir, exist_ok=True)

    def log(msg: str):
        if _is_main_process(args):
            print(msg, flush=True)

    # To avoid races on first download/cache when using torchrun
    if getattr(args, "distributed", False) and not _is_main_process(args):
        dist.barrier()

    # Tokenizer + LLM
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, use_fast=True, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=hf_token,
    )

    if getattr(args, "distributed", False) and _is_main_process(args):
        dist.barrier()

    # Optional: gradient checkpointing (reduces activation memory)
    if args.grad_checkpointing and hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable()
        # Disable KV cache during training (required by some models when checkpointing)
        if hasattr(llm.config, "use_cache"):
            llm.config.use_cache = False

    # Apply LoRA to LLM (recommended to mitigate OOM)
    if args.use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        llm = get_peft_model(llm, lora_cfg)

    # Freeze base LLM params if requested (keeps LoRA trainable if enabled)
    if args.freeze_llm:
        for n, p in llm.named_parameters():
            if args.use_lora and ("lora_" in n or "lora" in n):
                continue
            p.requires_grad = False

    llm.to(device)

    # Vision: DEM-Encoder
    enc_cfg = DEMEncoderConfig()
    pyramid = ResNetPyramidBackbone(name="resnet50", pretrained=args.backbone_pretrained)
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

    # Adapter: load from Stage-1
    llm_dim = llm.get_input_embeddings().weight.shape[1]
    adapter = DAAdapter(DAAdapterConfig(in_channels=encoder.out_channels, llm_dim=llm_dim)).to(device)

    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "adapter" in ckpt):
        raise ValueError("Stage-1 checkpoint must be a dict and contain key 'adapter'.")
    state = ckpt["adapter"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    adapter.load_state_dict(state, strict=True)

    if args.freeze_encoder:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

    # Combined multimodal model
    model = VisionPrefixQwen(
        encoder=encoder,
        adapter=adapter,
        llm=llm,
        feat_key=args.feat_key,
        prefix_grid=args.prefix_grid,
    ).to(device)

    # Wrap with DDP (multi-GPU)
    if getattr(args, "distributed", False):
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=args.ddp_find_unused_parameters,
        )

    def set_train_mode():
        model.train()
        if args.freeze_encoder:
            # keep frozen encoder in eval mode (BN/Dropout frozen)
            enc = model.module.encoder if hasattr(model, "module") else model.encoder
            enc.eval()

    # Datasets / Samplers
    train_ds = JsonlSFTDataset(args.train_jsonl)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if getattr(args, "distributed", False) else None

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_sft(b, tokenizer, enc_cfg, args.image_size, args.max_text_len),
        drop_last=False,
    )

    valid_dl = None
    valid_sampler = None
    if args.valid_jsonl:
        valid_ds = JsonlSFTDataset(args.valid_jsonl)
        valid_sampler = DistributedSampler(valid_ds, shuffle=False) if getattr(args, "distributed", False) else None
        valid_dl = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            collate_fn=lambda b: collate_sft(b, tokenizer, enc_cfg, args.image_size, args.max_text_len),
            drop_last=False,
        )

    # Optimizer: allow different LR for adapter vs (LLM/encoder/LoRA) params
    # Note: with DDP, access parameters on the wrapped module transparently.
    adapter_ref = model.module.adapter if hasattr(model, "module") else model.adapter
    encoder_ref = model.module.encoder if hasattr(model, "module") else model.encoder

    adapter_params = [p for p in adapter_ref.parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and (not n.startswith("module.adapter.") and not n.startswith("adapter."))]

    optim = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": adapter_params, "lr": args.adapter_lr},
        ],
        weight_decay=args.weight_decay,
    )

    # Scheduler (steps are per-rank; consistent across ranks under DDP)
    total_steps = math.ceil(len(train_dl) / args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp_fp16 = (args.fp16 and device.type == "cuda")
    use_amp_bf16 = (args.bf16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp_fp16)

    set_train_mode()
    global_step = 0

    def run_eval():
        if valid_dl is None:
            return
        model.eval()

        loss_sum = 0.0
        count = 0

        with torch.no_grad():
            for batch in valid_dl:
                images = batch["images"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=(use_amp_fp16 or use_amp_bf16),
                    dtype=(torch.float16 if use_amp_fp16 else torch.bfloat16),
                ):
                    out = model(images, input_ids, attention_mask, labels)
                    loss = out.loss.detach().float()

                loss_sum += loss.item()
                count += 1

        # aggregate across ranks
        if getattr(args, "distributed", False):
            t = torch.tensor([loss_sum, count], device=device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            loss_sum = float(t[0].item())
            count = int(t[1].item())

        set_train_mode()
        if _is_main_process(args):
            log(f"[valid] loss={loss_sum/max(1,count):.4f}")

    for epoch in range(args.epochs):
        set_train_mode()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_dl:
            images = batch["images"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            is_accum_step = ((global_step + 1) % args.grad_accum != 0)
            sync_ctx = model.no_sync() if (getattr(args, "distributed", False) and is_accum_step) else nullcontext()

            with sync_ctx:
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=(use_amp_fp16 or use_amp_bf16),
                    dtype=(torch.float16 if use_amp_fp16 else torch.bfloat16),
                ):
                    out = model(images, input_ids, attention_mask, labels)
                    loss = out.loss / args.grad_accum

                if use_amp_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                if use_amp_fp16:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optim)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optim.step()

                optim.zero_grad(set_to_none=True)
                sched.step()

            if _is_main_process(args) and (global_step % args.log_every == 0):
                log(f"epoch={epoch+1} step={global_step} loss={(loss.item()*args.grad_accum):.4f}")

            global_step += 1

        # eval & save each epoch
        run_eval()

        if _is_main_process(args):
            save_dir = Path(args.output_dir) / f"epoch{epoch+1}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save adapter (always)
            adapter_to_save = (model.module.adapter if hasattr(model, "module") else model.adapter).state_dict()
            torch.save({"adapter": adapter_to_save}, save_dir / "da_adapter.pt")

            # Save encoder if trainable
            if any(p.requires_grad for p in encoder_ref.parameters()):
                torch.save(encoder_ref.state_dict(), save_dir / "dem_encoder.pt")

            # Save LoRA / LLM (if enabled)
            if args.use_lora and hasattr(llm, "save_pretrained"):
                llm.save_pretrained(save_dir / "lora_llm")

            tokenizer.save_pretrained(save_dir / "tokenizer")

            log(f"Saved to: {save_dir}")

        if getattr(args, "distributed", False):
            dist.barrier()

    if getattr(args, "distributed", False):
        dist.barrier()
        dist.destroy_process_group()

    log("Done.")


if __name__ == "__main__":
    main()
