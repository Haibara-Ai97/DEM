# scripts/train_stage1_adapter_alignment.py
from __future__ import annotations
import argparse, os, math, random
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import CLIPModel, CLIPProcessor

from models.backbone import SimplePyramidBackbone
from models.da_adapter import DAAdapter, DAAdapterConfig

# 你上传的 DEMVisionBackbone：建议把上传文件重命名为 models/dem_encoder.py
from models.dem_encoder import DEMEncoderConfig, DEMVisionBackbone  # noqa


# -------------------------
# 数据
# -------------------------
class ImageCsvDataset(Dataset):
    def __init__(self, csv_path: str):
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert "image_path" in df.columns
        self.paths = df["image_path"].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        return Image.open(self.paths[idx]).convert("RGB")


def pil_resize_square(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), resample=Image.BICUBIC)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.uint8)           # (H,W,3)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def normalize(t: torch.Tensor, mean: Tuple[float,float,float], std: Tuple[float,float,float]) -> torch.Tensor:
    m = torch.tensor(mean, dtype=t.dtype).view(3,1,1)
    s = torch.tensor(std, dtype=t.dtype).view(3,1,1)
    return (t - m) / s


def collate_fn(batch_imgs: List[Image.Image], clip_processor: CLIPProcessor, enc_cfg: DEMEncoderConfig, image_size: int):
    # 1) encoder input: resize + toTensor + normalize(enc mean/std)
    enc_imgs = [pil_resize_square(im, image_size) for im in batch_imgs]
    enc_t = torch.stack([normalize(pil_to_tensor(im), enc_cfg.image_mean, enc_cfg.image_std) for im in enc_imgs], dim=0)

    # 2) CLIP input: 交给 CLIPProcessor（标准 CLIP 预处理）
    clip_enc = clip_processor(images=batch_imgs, return_tensors="pt")
    clip_t = clip_enc["pixel_values"]  # (B,3,H,W)

    return enc_t, clip_t


# -------------------------
# CLIP teacher：patch / text embeds
# -------------------------
@torch.no_grad()
def clip_patch_embeds(clip: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    返回 (B, N, D_clip_embed) patch embedding（已归一化）
    """
    out = clip.vision_model(pixel_values=pixel_values, return_dict=True)
    h = out.last_hidden_state                            # (B,1+N,Dv)
    h = clip.vision_model.post_layernorm(h)
    patch = h[:, 1:, :]                                  # (B,N,Dv)
    patch = clip.visual_projection(patch)                # (B,N,De)
    patch = F.normalize(patch, dim=-1)
    return patch


@torch.no_grad()
def clip_text_embeds(clip: CLIPModel, processor: CLIPProcessor, phrases: List[str], device: torch.device) -> torch.Tensor:
    enc = processor(text=phrases, return_tensors="pt", padding=True, truncation=True).to(device)
    t = clip.get_text_features(**enc)                    # (V,De)
    t = F.normalize(t, dim=-1)
    return t


# -------------------------
# token sampling + loss
# -------------------------
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
    # v,s: (M,D) normalized
    logits = (v @ s.t()) / max(temperature, 1e-6)
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--domain_vocab", type=str, required=True)
    ap.add_argument("--llm_phrase_pt", type=str, required=True)  # 由 precompute_llm_phrase_embeds.py 生成
    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch16")
    ap.add_argument("--output_dir", type=str, default="checkpoints/stage1")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--win", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--feat_key", type=str, default="0", choices=["0","1","2","3"])  # 选择 U2/U3/U4/U5
    ap.add_argument("--align_to_clip_grid", action="store_true")
    ap.add_argument("--train_encoder", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # 可选：快速 ablation / 加速
    ap.add_argument("--disable_dem2", action="store_true")
    ap.add_argument("--disable_dem3", action="store_true")
    ap.add_argument("--disable_dem4", action="store_true")
    ap.add_argument("--disable_dem5", action="store_true")

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取领域词表
    phrases = [l.strip() for l in open(args.domain_vocab, "r", encoding="utf-8") if l.strip()]
    assert len(phrases) > 0

    # 读取 LLM phrase embeds（LLM embedding space）
    payload = torch.load(args.llm_phrase_pt, map_location="cpu")
    llm_phrases = payload["phrases"]
    llm_phrase_embeds = payload["embeds"]                 # (V, d_llm) float32
    assert llm_phrases == phrases, "domain_vocab.txt 与 llm_phrase_pt 的 phrases 不一致，请重新预计算。"
    llm_phrase_embeds = F.normalize(llm_phrase_embeds.to(device), dim=-1)
    llm_dim = llm_phrase_embeds.size(-1)

    # CLIP teacher
    clip = CLIPModel.from_pretrained(args.clip_name).eval().to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_name)
    for p in clip.parameters():
        p.requires_grad = False
    clip_text = clip_text_embeds(clip, clip_processor, phrases, device)  # (V, D_clip)

    # DEM-Encoder（使用你的设计）
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

    # 只训练 Adapter（默认）；如需联合训练 Encoder，用 --train_encoder
    if not args.train_encoder:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

    # DA-Adapter：in_channels = encoder.out_channels（=cfg.C=256）
    adapter = DAAdapter(DAAdapterConfig(in_channels=encoder.out_channels, llm_dim=llm_dim)).to(device)
    adapter.train()

    params = list(adapter.parameters()) + (list(encoder.parameters()) if args.train_encoder else [])
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # DataLoader
    ds = ImageCsvDataset(args.train_csv)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_fn(b, clip_processor, enc_cfg, args.image_size),
    )

    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for enc_x, clip_x in pbar:
            enc_x = enc_x.to(device, non_blocking=True)
            clip_x = clip_x.to(device, non_blocking=True)

            # 1) CLIP patch tokens（teacher）
            with torch.no_grad():
                patch = clip_patch_embeds(clip, clip_x)                 # (B, Nc, Dc)
            B, Nc, Dc = patch.shape
            side = int(math.sqrt(Nc))
            assert side * side == Nc, "CLIP patch token 非正方网格，请改写 grid 推断。"

            # 2) DEM-Encoder multi-scale -> pick one feature map
            if args.train_encoder:
                feats: OrderedDict = encoder(enc_x)
            else:
                with torch.no_grad():
                    feats: OrderedDict = encoder(enc_x)

            feat = feats[args.feat_key]                                 # (B, C=256, H, W)

            # 3) 对齐到 CLIP patch grid（推荐打开）
            if args.align_to_clip_grid:
                feat = F.interpolate(feat, size=(side, side), mode="bilinear", align_corners=False)

            _, C, H, W = feat.shape
            assert H * W == Nc if args.align_to_clip_grid else True

            # 4) DA-Adapter -> V tokens（LLM space）
            with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
                V = adapter(feat)                                       # (B, N, d_llm)
                V = F.normalize(V, dim=-1)

                # 5) patch->vocab topk（CLIP space）
                sims = torch.einsum("bnd,vd->bnv", patch, clip_text)     # (B, Nc, Vocab)
                topv, topi = torch.topk(sims, k=min(args.topk, sims.size(-1)), dim=-1)  # (B,Nc,K)
                w = F.softmax(topv, dim=-1)                              # (B,Nc,K)

                # 6) 构造语义目标 S（LLM space）
                s_k = llm_phrase_embeds[topi]                            # (B,Nc,K,d_llm)
                S = (w.unsqueeze(-1) * s_k).sum(dim=-2)                  # (B,Nc,d_llm)
                S = F.normalize(S, dim=-1)

                # 若未对齐到 CLIP grid，则需要把 V resize/或把 S reshape 插值到 V 的 H×W
                if V.size(1) != S.size(1):
                    # 将 S 还原为 (B,d,side,side) 插值到 (H,W) 再展平
                    Sd = S.transpose(1,2).reshape(B, llm_dim, side, side)
                    Sd = F.interpolate(Sd, size=(H, W), mode="bilinear", align_corners=False)
                    S = Sd.flatten(2).transpose(1,2).contiguous()
                    S = F.normalize(S, dim=-1)

                # 7) window sampling，减小 InfoNCE 矩阵规模
                idx = window_sample_indices(H, W, win=args.win, seed=args.seed + global_step).to(device)
                V_s = V[:, idx, :].reshape(-1, llm_dim)
                S_s = S[:, idx, :].reshape(-1, llm_dim)

                loss = symmetric_infonce(V_s, S_s, temperature=args.temperature)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", N=H*W, K=idx.numel())

        # save ckpt each epoch
        ckpt = {
            "adapter": adapter.state_dict(),
            "args": vars(args),
        }
        if args.train_encoder:
            ckpt["encoder"] = encoder.state_dict()
        torch.save(ckpt, os.path.join(args.output_dir, f"stage1_epoch{epoch+1}.pt"))

    print(f"Done. Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    from tqdm import tqdm
    main()
