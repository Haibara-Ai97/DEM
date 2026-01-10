from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_name", type=str, required=True)
    ap.add_argument("--domain_vocab", type=str, required=True)
    ap.add_argument("--out_pt", type=str, default="data/llm_phrase_embeds.pt")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16","bfloat16","float32"])
    args = ap.parse_args()

    # Read the phrase list from the vocabulary file
    phrases = [l.strip() for l in open(args.domain_vocab, "r", encoding="utf-8") if l.strip()]
    assert len(phrases) > 0

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    # load tokenizer
    tok = AutoTokenizer.from_pretrained(args.llm_name, use_fast=True)
    # load LLM
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_name,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    W = model.get_input_embeddings().weight.detach()  # (Vocab, d_llm)

    # Compute a vector for each phrase
    #
    embs = []
    for p in phrases:
        ids = tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        e = W[ids].mean(dim=0)                         # phrase pooling: mean
        embs.append(e)
    embs = torch.stack(embs, dim=0)                    # (V, d_llm)
    embs = F.normalize(embs, dim=-1)

    payload = {
        "phrases": phrases,
        "embeds": embs.to(torch.float32),              # 保存 float32 更稳
        "llm_name": args.llm_name,
    }
    torch.save(payload, args.out_pt)
    print(f"Saved: {args.out_pt}, shape={tuple(embs.shape)}")


if __name__ == "__main__":
    main()
