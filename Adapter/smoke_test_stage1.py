# smoke_test_stage1.py
from __future__ import annotations

import os, sys, math, shutil, argparse, runpy, random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from transformers import CLIPModel, CLIPProcessor

# 依赖你的工程代码
from dem.models.backbone import SimplePyramidBackbone
from dem.models.da_adapter import DAAdapter, DAAdapterConfig
from dem.models.dem_encoder import DEMEncoderConfig, DEMVisionBackbone  # 你的上传文件（重命名为 dem_encoder.py）

# 复用你 precompute 脚本的核心逻辑（避免再写一次）
from transformers import AutoTokenizer, AutoModelForCausalLM


def _make_random_images(workdir: Path, n: int = 2, size: int = 256) -> list[str]:
    img_dir = workdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        im = Image.fromarray(arr, mode="RGB")
        p = img_dir / f"rand_{i}.jpg"
        im.save(p, quality=95)
        paths.append(str(p))
    return paths


def _write_csv(workdir: Path, image_paths: list[str]) -> str:
    import pandas as pd
    csv_path = workdir / "train_images.csv"
    pd.DataFrame({"image_path": image_paths}).to_csv(csv_path, index=False)
    return str(csv_path)


def _write_vocab(workdir: Path) -> str:
    vocab_path = workdir / "domain_vocab.txt"
    vocab = [
        "crack",
        "efflorescence",
        "water seepage",
        "spalling",
    ]
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    return str(vocab_path)


def _precompute_llm_phrase_embeds(llm_name: str, domain_vocab_path: str, out_pt: str):
    phrases = [l.strip() for l in open(domain_vocab_path, "r", encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    W = model.get_input_embeddings().weight.detach()  # (Vocab, d_llm)

    embs = []
    for p in phrases:
        ids = tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        e = W[ids].mean(dim=0)
        embs.append(e)
    embs = torch.stack(embs, dim=0)
    embs = F.normalize(embs, dim=-1)

    payload = {"phrases": phrases, "embeds": embs.to(torch.float32), "llm_name": llm_name}
    torch.save(payload, out_pt)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def _normalize(t: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, dtype=t.dtype).view(3, 1, 1)
    s = torch.tensor(std, dtype=t.dtype).view(3, 1, 1)
    return (t - m) / s


def _window_sample_indices(h: int, w: int, win: int = 2, seed: int = 123) -> torch.Tensor:
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


def _symmetric_infonce(v: torch.Tensor, s: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = (v @ s.t()) / max(temperature, 1e-6)
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


@torch.no_grad()
def _clip_patch_embeds(clip: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    兼容旧版 transformers：vision_model.forward 不支持 return_dict
    返回 (B, N, D_clip_embed) patch embedding（已归一化）
    """
    out = clip.vision_model(pixel_values=pixel_values)  # 不传 return_dict

    # 兼容：旧版通常返回 tuple，新版可能返回带属性的输出对象
    if isinstance(out, (tuple, list)):
        hidden = out[0]  # last_hidden_state: (B, 1+N, Dv)
    else:
        hidden = out.last_hidden_state

    # post_layernorm 在 CLIPVisionTransformer 中通常存在
    if hasattr(clip.vision_model, "post_layernorm"):
        hidden = clip.vision_model.post_layernorm(hidden)

    patch = hidden[:, 1:, :]                 # 去 CLS -> (B, N, Dv)
    patch = clip.visual_projection(patch)    # -> (B, N, De)
    patch = F.normalize(patch, dim=-1)
    return patch



@torch.no_grad()
def _clip_text_embeds(clip: CLIPModel, processor: CLIPProcessor, phrases: list[str], device) -> torch.Tensor:
    enc = processor(text=phrases, return_tensors="pt", padding=True, truncation=True).to(device)
    t = clip.get_text_features(**enc)
    return F.normalize(t, dim=-1)


def sanity_forward_backward(workdir: Path, device: torch.device, align_to_clip_grid: bool = True):
    # 准备最小数据
    img_paths = _make_random_images(workdir, n=2, size=256)
    vocab_path = _write_vocab(workdir)
    llm_pt = workdir / "llm_phrase_embeds.pt"

    # 用小 LLM 预计算 phrase embeds（仅验证流程）
    _precompute_llm_phrase_embeds("distilgpt2", vocab_path, str(llm_pt))
    payload = torch.load(str(llm_pt), map_location="cpu")
    phrases = payload["phrases"]
    llm_phrase = F.normalize(payload["embeds"].to(device), dim=-1)
    llm_dim = llm_phrase.size(-1)

    # CLIP teacher（标准 CLIP）
    clip_name = "openai/clip-vit-base-patch16"
    clip = CLIPModel.from_pretrained(clip_name).eval().to(device)
    processor = CLIPProcessor.from_pretrained(clip_name)
    for p in clip.parameters():
        p.requires_grad = False

    clip_text = _clip_text_embeds(clip, processor, phrases, device)

    # DEM-Encoder + Adapter
    enc_cfg = DEMEncoderConfig()
    pyramid = SimplePyramidBackbone()
    encoder = DEMVisionBackbone(pyramid_backbone=pyramid, cfg=enc_cfg).to(device)

    # 验证“冻结 encoder”
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    adapter = DAAdapter(DAAdapterConfig(in_channels=encoder.out_channels, llm_dim=llm_dim)).to(device)
    adapter.train()

    # 构造一批图像输入
    imgs = [Image.open(p).convert("RGB") for p in img_paths]

    # encoder 输入（按 enc_cfg mean/std）
    enc_imgs = [im.resize((224, 224), resample=Image.BICUBIC) for im in imgs]
    enc_x = torch.stack([_normalize(_pil_to_tensor(im), enc_cfg.image_mean, enc_cfg.image_std) for im in enc_imgs], dim=0).to(device)

    # clip 输入（processor）
    clip_x = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)

    # 1) CLIP patch
    with torch.no_grad():
        patch = _clip_patch_embeds(clip, clip_x)  # (B,Nc,Dc)
    B, Nc, Dc = patch.shape
    side = int(math.sqrt(Nc))
    assert side * side == Nc, "CLIP patch token 不是正方网格，需改写 grid 推断。"

    # 2) DEM features
    with torch.no_grad():
        feats = encoder(enc_x)                    # OrderedDict
    feat = feats["0"]                             # 默认用 U2，对应你脚本 feat_key="0"
    # 3) 对齐到 CLIP 网格
    if align_to_clip_grid:
        feat = F.interpolate(feat, size=(side, side), mode="bilinear", align_corners=False)

    # 4) Adapter -> V
    V = adapter(feat)                             # (B,N,d_llm)
    V = F.normalize(V, dim=-1)

    # 5) topk 检索 -> S
    sims = torch.einsum("bnd,vd->bnv", patch, clip_text)
    topv, topi = torch.topk(sims, k=min(4, sims.size(-1)), dim=-1)
    w = F.softmax(topv, dim=-1)
    s_k = llm_phrase[topi]                        # (B,N,K,d_llm)
    S = (w.unsqueeze(-1) * s_k).sum(dim=-2)
    S = F.normalize(S, dim=-1)

    # 6) 采样 + loss
    H = W = side
    idx = _window_sample_indices(H, W, win=2, seed=123).to(device)
    V_s = V[:, idx, :].reshape(-1, llm_dim)
    S_s = S[:, idx, :].reshape(-1, llm_dim)
    loss = _symmetric_infonce(V_s, S_s, temperature=0.07)

    # backward
    for p in adapter.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()

    # 检查：loss 有限、adapter 有梯度、encoder 无梯度（冻结）
    assert torch.isfinite(loss).item(), "Loss is not finite."
    has_adapter_grad = any([(p.grad is not None) and torch.isfinite(p.grad).all().item() for p in adapter.parameters()])
    assert has_adapter_grad, "Adapter has no valid gradients."

    has_encoder_grad = any([(p.grad is not None) for p in encoder.parameters()])
    assert not has_encoder_grad, "Encoder should be frozen but has gradients."

    print("[OK] sanity_forward_backward passed.")
    print(f"  feat: {tuple(feat.shape)}  V: {tuple(V.shape)}  S: {tuple(S.shape)}  loss: {loss.item():.4f}")


def run_tiny_training(workdir: Path, device: torch.device):
    """
    直接调用你的 Stage-1 训练脚本，确保能跑完并落盘 ckpt。
    """
    # 准备最小数据
    img_paths = _make_random_images(workdir, n=2, size=256)
    csv_path = _write_csv(workdir, img_paths)
    vocab_path = _write_vocab(workdir)

    llm_pt = workdir / "llm_phrase_embeds.pt"
    _precompute_llm_phrase_embeds("distilgpt2", vocab_path, str(llm_pt))

    out_dir = workdir / "ckpt_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 用 runpy 运行脚本 main，等价于命令行 python scripts/train_stage1_adapter_alignment.py ...
    sys.argv = [
        "train.py",
        "--train_csv", csv_path,
        "--domain_vocab", vocab_path,
        "--llm_phrase_pt", str(llm_pt),
        "--output_dir", str(out_dir),
        "--epochs", "1",
        "--batch_size", "2",
        "--lr", "1e-4",
        "--topk", "2",
        "--win", "2",
        "--temperature", "0.07",
        "--image_size", "224",
        "--feat_key", "0",
        "--align_to_clip_grid",
        # CPU 验证默认不启用 amp
    ]
    print("[RUN] tiny training via train_stage1_adapter_alignment.py ...")
    runpy.run_module("train", run_name="__main__")

    ckpt = out_dir / "stage1_epoch1.pt"
    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    print(f"[OK] tiny training finished, checkpoint: {ckpt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=str, default="_smoke_stage1")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--run_train", action="store_true", help="是否额外跑一次极小训练并检查ckpt落盘")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    sanity_forward_backward(workdir, device, align_to_clip_grid=True)

    if args.run_train:
        run_tiny_training(workdir, device)

    print("[PASS] Stage-1 pipeline smoke test done.")


if __name__ == "__main__":
    main()
