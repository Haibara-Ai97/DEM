#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate the VisionPrefixQwen-style model on the concrete-defect QA jsonl.

End-to-end inference pipeline:
  image -> DEM encoder -> pooled grid -> DAAdapter -> vision-prefix embeddings
  (vision prefix + text prompt embeddings) -> LLM.generate -> textual answer

Supported tasks (from your dataset-generation script):
  - yesno
  - multilabel
  - count
  - grid
  - json

Outputs:
  - prints aggregate metrics
  - optionally writes per-example predictions to a jsonl

Assumes the following modules are importable in your repo:
  from models.dem_encoder import DEMEncoderConfig, DEMVisionBackbone
  from models.backbone import ResNetPyramidBackbone
  from models.da_adapter import DAAdapter, DAAdapterConfig
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except Exception:
    PeftModel = None  # type: ignore

from models.dem_encoder import DEMEncoderConfig, DEMVisionBackbone
from models.backbone import ResNetPyramidBackbone
from models.da_adapter import DAAdapter, DAAdapterConfig


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
    m = torch.tensor(mean, dtype=t.dtype, device=t.device).view(3, 1, 1)
    s = torch.tensor(std, dtype=t.dtype, device=t.device).view(3, 1, 1)
    return (t - m) / s


# -------------------------
# Prompt building
# -------------------------
def build_prompt_ids(tokenizer, system: str, user: str, max_text_len: int) -> torch.Tensor:
    """
    Return 1D input_ids for the prompt (system + user + generation prompt).
    """
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )[0]
    else:
        text = f"System: {system}\nUser: {user}\nAssistant:"
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_text_len).input_ids[0]
    if ids.numel() > max_text_len:
        ids = ids[:max_text_len]
    return ids


# -------------------------
# Model wrapper for eval generation
# -------------------------
class VisionPrefixGenerator(nn.Module):
    def __init__(self, encoder: nn.Module, adapter: nn.Module, llm: nn.Module, feat_key: str, prefix_grid: int):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.llm = llm
        self.feat_key = feat_key
        self.prefix_grid = prefix_grid

    @torch.no_grad()
    def build_inputs_embeds(
        self,
        images: torch.Tensor,                # (B,3,H,W)
        input_ids: torch.Tensor,             # (B,L)
        attention_mask: torch.Tensor,        # (B,L)
    ):
        feats = self.encoder(images)
        if self.feat_key not in feats:
            raise KeyError(f"feat_key={self.feat_key} not found. Available: {list(feats.keys())}")
        feat = feats[self.feat_key]  # (B,C,Hf,Wf)

        pooled = F.adaptive_avg_pool2d(feat, (self.prefix_grid, self.prefix_grid))  # (B,C,g,g)
        vtok = self.adapter(pooled)  # (B, N, D)
        B, N, _ = vtok.shape

        tok_emb = self.llm.get_input_embeddings()(input_ids)  # (B,L,D)
        inputs_embeds = torch.cat([vtok, tok_emb], dim=1)     # (B,N+L,D)

        prefix_mask = torch.ones((B, N), dtype=attention_mask.dtype, device=attention_mask.device)
        attn_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        return inputs_embeds, attn_mask, N

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
    ):
        """
        We call `llm.generate(inputs_embeds=...)`.
        Note: The returned sequences contain dummy ids for the prefix+prompt part.
              So we always decode only the tail: sequences[:, in_len:].
        """
        inputs_embeds, attn_mask, _ = self.build_inputs_embeds(images, input_ids, attention_mask)
        B, in_len, _ = inputs_embeds.shape

        dummy_id = pad_token_id if pad_token_id is not None else eos_token_id
        if dummy_id is None:
            dummy_id = 0

        dummy_input_ids = torch.full((B, in_len), int(dummy_id), dtype=torch.long, device=inputs_embeds.device)

        gen = self.llm.generate(
            input_ids=dummy_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            num_beams=1,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        return gen, in_len


# -------------------------
# Output normalization/parsing (align with your dataset spec)
# -------------------------
YES_SET = {"yes", "y", "true"}
NO_SET = {"no", "n", "false"}
DEFAULT_DEFECT_VOCAB = ["crack", "spalling", "honeycomb", "hole", "exposed rebar", "seepage"]

_re_int = re.compile(r"[-+]?\d+")
_re_cell = re.compile(r"r\s*(\d+)\s*c\s*(\d+)", re.IGNORECASE)

def norm_yesno(text: str) -> Optional[str]:
    t = text.strip().lower()
    m = re.search(r"(yes|no|true|false|y|n)\b", t)
    if not m:
        return None
    w = m.group(1)
    if w in YES_SET:
        return "Yes"
    if w in NO_SET:
        return "No"
    return None

def norm_multilabel(text: str, vocab: Sequence[str]) -> Optional[List[str]]:
    t = text.strip().lower()
    if not t:
        return None
    if t == "none" or t.startswith("none"):
        return []
    parts = [p.strip() for p in t.split(",")]
    labels = []
    for p in parts:
        if not p:
            continue
        best = None
        for v in vocab:
            if p == v:
                best = v
                break
        if best is None:
            for v in vocab:
                if v in p or p in v:
                    best = v
                    break
        if best is not None:
            labels.append(best)
    seen, out = set(), []
    for x in labels:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def norm_count(text: str) -> Optional[int]:
    m = _re_int.search(text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None

def norm_cell(text: str) -> Optional[str]:
    m = _re_cell.search(text)
    if not m:
        return None
    r = int(m.group(1)); c = int(m.group(2))
    return f"r{r}c{c}"

def parse_json_strict(text: str) -> Optional[Dict[str, Any]]:
    t = text.strip()
    if not t:
        return None
    if "{" in t and "}" in t:
        t = t[t.find("{"): t.rfind("}") + 1]
    try:
        return json.loads(t)
    except Exception:
        return None

def json_schema_ok(obj: Any, vocab: Sequence[str]) -> bool:
    if not isinstance(obj, dict):
        return False
    if "defects" not in obj or "overall_condition" not in obj:
        return False
    if not isinstance(obj["defects"], list):
        return False
    if obj["overall_condition"] not in {"good", "fair", "poor"}:
        return False
    for d in obj["defects"]:
        if not isinstance(d, dict):
            return False
        for k in ("type", "count", "severity", "primary_cell"):
            if k not in d:
                return False
        if not isinstance(d["type"], str) or d["type"] not in set(vocab):
            return False
        if not isinstance(d["count"], int):
            return False
        if d["severity"] not in {"small", "medium", "large"}:
            return False
        if not isinstance(d["primary_cell"], str) or _re_cell.search(d["primary_cell"]) is None:
            return False
    return True


# -------------------------
# Metrics
# -------------------------
@dataclass
class YesNoMetrics:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    invalid: int = 0

    def update(self, pred: Optional[str], gold: str) -> None:
        if pred not in ("Yes", "No"):
            self.invalid += 1
            return
        if gold == "Yes":
            if pred == "Yes": self.tp += 1
            else: self.fn += 1
        else:
            if pred == "No": self.tn += 1
            else: self.fp += 1

    def accuracy(self) -> float:
        n = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / n if n else 0.0

    def balanced_acc(self) -> float:
        tpr = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        tnr = self.tn / (self.tn + self.fp) if (self.tn + self.fp) else 0.0
        return 0.5 * (tpr + tnr)

    def macro_f1(self) -> float:
        def f1(p, r): return 2*p*r/(p+r) if (p+r) else 0.0
        prec_yes = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        rec_yes  = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1_yes = f1(prec_yes, rec_yes)
        prec_no = self.tn / (self.tn + self.fn) if (self.tn + self.fn) else 0.0
        rec_no  = self.tn / (self.tn + self.fp) if (self.tn + self.fp) else 0.0
        f1_no = f1(prec_no, rec_no)
        return 0.5 * (f1_yes + f1_no)

    def valid_rate(self) -> float:
        n_all = self.tp + self.tn + self.fp + self.fn + self.invalid
        n_valid = self.tp + self.tn + self.fp + self.fn
        return n_valid / n_all if n_all else 0.0


@dataclass
class MultiLabelMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    invalid: int = 0
    ex_f1_sum: float = 0.0
    ex_n: int = 0

    def update(self, pred: Optional[List[str]], gold: str) -> None:
        gold_labels = [] if gold.strip().lower() == "none" else [x.strip().lower() for x in gold.split(",")]
        gold_set = set([g for g in gold_labels if g])
        if pred is None:
            self.invalid += 1
            return
        pred_set = set(pred)

        self.tp += len(pred_set & gold_set)
        self.fp += len(pred_set - gold_set)
        self.fn += len(gold_set - pred_set)

        inter = len(pred_set & gold_set)
        denom = len(pred_set) + len(gold_set)
        f1 = (2 * inter / denom) if denom else 1.0
        self.ex_f1_sum += f1
        self.ex_n += 1

    def micro_f1(self) -> float:
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        return (2*p*r/(p+r)) if (p+r) else 0.0

    def example_f1(self) -> float:
        return self.ex_f1_sum / self.ex_n if self.ex_n else 0.0

    def valid_rate(self) -> float:
        n_all = self.ex_n + self.invalid
        return self.ex_n / n_all if n_all else 0.0


@dataclass
class CountMetrics:
    n: int = 0
    invalid: int = 0
    abs_err_sum: float = 0.0
    sq_err_sum: float = 0.0
    exact: int = 0
    within1: int = 0

    def update(self, pred: Optional[int], gold: str) -> None:
        try:
            g = int(gold.strip())
        except Exception:
            return
        if pred is None:
            self.invalid += 1
            return
        self.n += 1
        e = pred - g
        self.abs_err_sum += abs(e)
        self.sq_err_sum += float(e * e)
        if pred == g: self.exact += 1
        if abs(e) <= 1: self.within1 += 1

    def mae(self) -> float:
        return self.abs_err_sum / self.n if self.n else 0.0

    def rmse(self) -> float:
        return math.sqrt(self.sq_err_sum / self.n) if self.n else 0.0

    def exact_acc(self) -> float:
        return self.exact / self.n if self.n else 0.0

    def within1_acc(self) -> float:
        return self.within1 / self.n if self.n else 0.0

    def valid_rate(self) -> float:
        n_all = self.n + self.invalid
        return self.n / n_all if n_all else 0.0


@dataclass
class GridMetrics:
    n: int = 0
    invalid: int = 0
    exact: int = 0
    within1: int = 0
    l1_sum: float = 0.0

    def update(self, pred: Optional[str], gold: str) -> None:
        if pred is None:
            self.invalid += 1
            return
        m1 = _re_cell.search(pred)
        m2 = _re_cell.search(gold)
        if not (m1 and m2):
            self.invalid += 1
            return
        pr, pc = int(m1.group(1)), int(m1.group(2))
        gr, gc = int(m2.group(1)), int(m2.group(2))
        d = abs(pr - gr) + abs(pc - gc)
        self.n += 1
        self.l1_sum += d
        if d == 0: self.exact += 1
        if d <= 1: self.within1 += 1

    def exact_acc(self) -> float:
        return self.exact / self.n if self.n else 0.0

    def within1_acc(self) -> float:
        return self.within1 / self.n if self.n else 0.0

    def mean_l1(self) -> float:
        return self.l1_sum / self.n if self.n else 0.0

    def valid_rate(self) -> float:
        n_all = self.n + self.invalid
        return self.n / n_all if n_all else 0.0


@dataclass
class JsonMetrics:
    n: int = 0
    parse_ok: int = 0
    schema_ok: int = 0
    overall_acc: int = 0
    type_f1_sum: float = 0.0
    count_mae_sum: float = 0.0
    sev_acc_sum: float = 0.0
    cell_acc_sum: float = 0.0

    def update(self, pred_obj: Optional[Dict[str, Any]], gold_str: str, vocab: Sequence[str]) -> None:
        self.n += 1
        gold_obj = parse_json_strict(gold_str)
        if gold_obj is None:
            return
        if pred_obj is None:
            return
        self.parse_ok += 1
        if not json_schema_ok(pred_obj, vocab):
            return
        self.schema_ok += 1

        if pred_obj.get("overall_condition") == gold_obj.get("overall_condition"):
            self.overall_acc += 1

        gt = {d["type"]: d for d in gold_obj.get("defects", []) if isinstance(d, dict) and "type" in d}
        pr = {d["type"]: d for d in pred_obj.get("defects", []) if isinstance(d, dict) and "type" in d}

        gt_types = set(gt.keys())
        pr_types = set(pr.keys())
        inter = len(gt_types & pr_types)
        denom = len(gt_types) + len(pr_types)
        type_f1 = (2 * inter / denom) if denom else 1.0
        self.type_f1_sum += type_f1

        if gt_types:
            abs_errs = []
            sev_hits = 0
            cell_hits = 0
            for t in gt_types:
                gt_d = gt[t]
                pr_d = pr.get(t)

                gt_count = int(gt_d.get("count", 0))
                pr_count = int(pr_d.get("count", 0)) if (pr_d is not None and isinstance(pr_d.get("count", None), int)) else 0
                abs_errs.append(abs(pr_count - gt_count))

                gt_sev = gt_d.get("severity", None)
                pr_sev = pr_d.get("severity", None) if pr_d is not None else None
                if pr_sev == gt_sev:
                    sev_hits += 1

                gt_cell = norm_cell(str(gt_d.get("primary_cell", "")))
                pr_cell = norm_cell(str(pr_d.get("primary_cell", ""))) if pr_d is not None else None
                if (gt_cell is not None) and (pr_cell is not None) and (gt_cell.lower() == pr_cell.lower()):
                    cell_hits += 1

            self.count_mae_sum += float(sum(abs_errs)) / len(abs_errs)
            self.sev_acc_sum += sev_hits / len(gt_types)
            self.cell_acc_sum += cell_hits / len(gt_types)
        else:
            self.count_mae_sum += 0.0
            self.sev_acc_sum += 1.0
            self.cell_acc_sum += 1.0

    def parse_rate(self) -> float:
        return self.parse_ok / self.n if self.n else 0.0

    def schema_rate(self) -> float:
        return self.schema_ok / self.n if self.n else 0.0

    def overall_condition_acc(self) -> float:
        return self.overall_acc / self.schema_ok if self.schema_ok else 0.0

    def type_f1(self) -> float:
        return self.type_f1_sum / self.schema_ok if self.schema_ok else 0.0

    def count_mae(self) -> float:
        return self.count_mae_sum / self.schema_ok if self.schema_ok else 0.0

    def severity_acc(self) -> float:
        return self.sev_acc_sum / self.schema_ok if self.schema_ok else 0.0

    def cell_acc(self) -> float:
        return self.cell_acc_sum / self.schema_ok if self.schema_ok else 0.0


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--llm_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", None))

    ap.add_argument("--adapter_ckpt", type=str, required=True, help="Path to da_adapter.pt (dict with key 'adapter').")
    ap.add_argument("--encoder_ckpt", type=str, default="", help="Optional path to dem_encoder.pt (state_dict).")
    ap.add_argument("--lora_dir", type=str, default="", help="Optional LoRA directory (PEFT).")

    ap.add_argument("--feat_key", type=str, default="3")
    ap.add_argument("--prefix_grid", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--max_text_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--max_new_tokens_json", type=int, default=256)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--defect_vocab", type=str, default=",".join(DEFAULT_DEFECT_VOCAB))
    ap.add_argument("--save_pred_jsonl", type=str, default="")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.fp16 and args.bf16:
        raise ValueError("Choose at most one: --fp16 or --bf16")

    use_amp = (device.type == "cuda") and (args.fp16 or args.bf16)
    amp_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    tok = AutoTokenizer.from_pretrained(args.llm_name, trust_remote_code=True, token=args.hf_token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_name, trust_remote_code=True, token=args.hf_token,
        low_cpu_mem_usage=True,
    )
    if args.lora_dir:
        if PeftModel is None:
            raise RuntimeError("peft not installed but --lora_dir is provided.")
        llm = PeftModel.from_pretrained(llm, args.lora_dir)
    llm.eval().to(device)

    # encoder
    enc_cfg = DEMEncoderConfig()
    backbone = ResNetPyramidBackbone(name="resnet50", pretrained=False)
    encoder = DEMVisionBackbone(
        cfg=enc_cfg,
        pyramid_backbone=backbone,
        disable_dem2=False,
        disable_dem3=False,
        disable_dem4=False,
        disable_dem5=False,
    )
    if args.encoder_ckpt:
        sd = torch.load(args.encoder_ckpt, map_location="cpu")
        if isinstance(sd, dict) and any(isinstance(v, torch.Tensor) for v in sd.values()):
            encoder.load_state_dict(sd, strict=False)
        elif isinstance(sd, dict) and "encoder" in sd and isinstance(sd["encoder"], dict):
            encoder.load_state_dict(sd["encoder"], strict=False)
        else:
            encoder.load_state_dict(sd, strict=False)
    encoder.eval().to(device)

    # adapter
    llm_dim = llm.get_input_embeddings().weight.shape[1]
    adapter = DAAdapter(DAAdapterConfig(in_channels=encoder.out_channels, llm_dim=llm_dim))
    ck = torch.load(args.adapter_ckpt, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "adapter" in ck and isinstance(ck["adapter"], dict)):
        raise ValueError("adapter_ckpt must be a dict with key 'adapter'.")
    state = {k.replace("module.", ""): v for k, v in ck["adapter"].items()}
    adapter.load_state_dict(state, strict=True)
    adapter.eval().to(device)

    model = VisionPrefixGenerator(encoder=encoder, adapter=adapter, llm=llm, feat_key=args.feat_key, prefix_grid=args.prefix_grid)
    model.eval().to(device)

    defect_vocab = [x.strip().lower() for x in args.defect_vocab.split(",") if x.strip()]

    yesno_all = YesNoMetrics()
    yesno_by_defect: Dict[str, YesNoMetrics] = defaultdict(YesNoMetrics)
    multilabel_m = MultiLabelMetrics()
    count_m = CountMetrics()
    grid_m = GridMetrics()
    json_m = JsonMetrics()

    out_f = None
    if args.save_pred_jsonl:
        Path(args.save_pred_jsonl).parent.mkdir(parents=True, exist_ok=True)
        out_f = open(args.save_pred_jsonl, "w", encoding="utf-8")

    batch_ex, batch_imgs, batch_ids = [], [], []

    def flush():
        nonlocal batch_ex, batch_imgs, batch_ids
        if not batch_ex:
            return

        Lmax = max(t.numel() for t in batch_ids)
        B = len(batch_ids)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        input_ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((B, Lmax), dtype=torch.long)
        for i, ids in enumerate(batch_ids):
            l = ids.numel()
            input_ids[i, :l] = ids
            attn_mask[i, :l] = 1

        images = torch.stack(batch_imgs, dim=0)

        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        images = images.to(device)

        max_new = args.max_new_tokens
        if any(ex.get("task") == "json" for ex in batch_ex):
            max_new = max(max_new, args.max_new_tokens_json)

        with autocast_ctx:
            seqs, in_len = model.generate(
                images=images,
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )

        print("[DEBUG] seq_len=", seqs.shape[1], "in_len=", in_len, "tail_len=", seqs.shape[1]-in_len)
        gen_part = seqs[:, in_len:]
        texts = tok.batch_decode(gen_part, skip_special_tokens=True)
        print("[DEBUG] pred_raw=", repr(texts[0]))
        print("[DEBUG] pred_yesno_norm=", norm_yesno(texts[0]))

        for ex, pred_text in zip(batch_ex, texts):
            task = ex.get("task", "")
            gold = ex.get("assistant", "")
            meta = ex.get("meta", {}) or {}

            rec = dict(ex)
            rec["pred_text"] = pred_text

            if task == "yesno":
                pred = norm_yesno(pred_text)
                yesno_all.update(pred, gold)
                d = meta.get("defect") or "unknown"
                yesno_by_defect[d].update(pred, gold)
                rec["pred_norm"] = pred
            elif task == "multilabel":
                pred = norm_multilabel(pred_text, defect_vocab)
                multilabel_m.update(pred, gold)
                rec["pred_norm"] = pred
            elif task == "count":
                pred = norm_count(pred_text)
                count_m.update(pred, gold)
                rec["pred_norm"] = pred
            elif task == "grid":
                pred = norm_cell(pred_text)
                grid_m.update(pred, gold)
                rec["pred_norm"] = pred
            elif task == "json":
                pred_obj = parse_json_strict(pred_text)
                json_m.update(pred_obj, gold, defect_vocab)
                rec["pred_norm"] = pred_obj
            else:
                rec["pred_norm"] = None

            if out_f is not None:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        batch_ex, batch_imgs, batch_ids = [], [], []

    for ex in iter_jsonl(args.jsonl):
        img_path = ex.get("image_path", "")
        if not img_path:
            continue

        img = Image.open(img_path).convert("RGB")
        img = pil_resize_square(img, args.image_size)
        timg = pil_to_tensor(img)
        timg = normalize(timg, enc_cfg.image_mean, enc_cfg.image_std)

        ids = build_prompt_ids(tok, ex.get("system", ""), ex.get("user", ""), args.max_text_len)

        batch_ex.append(ex)
        batch_imgs.append(timg)
        batch_ids.append(ids)

        if len(batch_ex) >= args.batch_size:
            flush()

    flush()
    if out_f is not None:
        out_f.close()

    print("========== EVAL RESULTS ==========")

    # yesno
    total_yesno = yesno_all.tp + yesno_all.tn + yesno_all.fp + yesno_all.fn + yesno_all.invalid
    if total_yesno > 0:
        print("[yesno] valid_rate={:.4f}  acc={:.4f}  bacc={:.4f}  macro_f1={:.4f}  invalid={}".format(
            yesno_all.valid_rate(), yesno_all.accuracy(), yesno_all.balanced_acc(), yesno_all.macro_f1(), yesno_all.invalid
        ))
        rows = []
        for d, m in yesno_by_defect.items():
            n = m.tp + m.tn + m.fp + m.fn + m.invalid
            rows.append((n, d, m))
        rows.sort(reverse=True, key=lambda x: x[0])
        print("[yesno] per-defect (top 20 by support):")
        for n, d, m in rows[:20]:
            print("  - {:<14s} n={:<5d} acc={:.3f} bacc={:.3f} macro_f1={:.3f} valid={:.3f}".format(
                d, n, m.accuracy(), m.balanced_acc(), m.macro_f1(), m.valid_rate()
            ))

    # multilabel
    total_ml = multilabel_m.ex_n + multilabel_m.invalid
    if total_ml > 0:
        print("[multilabel] valid_rate={:.4f}  micro_f1={:.4f}  example_f1={:.4f}  invalid={}".format(
            multilabel_m.valid_rate(), multilabel_m.micro_f1(), multilabel_m.example_f1(), multilabel_m.invalid
        ))

    # count
    total_c = count_m.n + count_m.invalid
    if total_c > 0:
        print("[count] valid_rate={:.4f}  exact_acc={:.4f}  within1_acc={:.4f}  MAE={:.4f}  RMSE={:.4f}  invalid={}".format(
            count_m.valid_rate(), count_m.exact_acc(), count_m.within1_acc(), count_m.mae(), count_m.rmse(), count_m.invalid
        ))

    # grid
    total_g = grid_m.n + grid_m.invalid
    if total_g > 0:
        print("[grid] valid_rate={:.4f}  exact_acc={:.4f}  within1_acc={:.4f}  mean_L1={:.4f}  invalid={}".format(
            grid_m.valid_rate(), grid_m.exact_acc(), grid_m.within1_acc(), grid_m.mean_l1(), grid_m.invalid
        ))

    # json
    if json_m.n > 0:
        print("[json] parse_rate={:.4f}  schema_rate={:.4f}  overall_acc={:.4f}  type_f1={:.4f}  count_MAE={:.4f}  severity_acc={:.4f}  cell_acc={:.4f}".format(
            json_m.parse_rate(), json_m.schema_rate(), json_m.overall_condition_acc(),
            json_m.type_f1(), json_m.count_mae(), json_m.severity_acc(), json_m.cell_acc()
        ))


if __name__ == "__main__":
    main()
