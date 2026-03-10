#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple single-turn inference with Qwen2.5-VL-7B-Instruct.

Examples:
  python -m scripts.vlm_baselines.qwen25vl_infer \
    --image https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
    --text_file /path/to/prompt.txt

  python -m scripts.vlm_baselines.qwen25vl_infer \
    --image /path/to/image.jpg \
    --text_file /path/to/prompt.txt \
    --system "你是一个混凝土缺陷巡检助手"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Call Qwen2.5-VL-7B directly with an image path/URL + text prompt.")
    ap.add_argument("--image", type=str, required=True, help="Image path (local file) or URL.")
    ap.add_argument("--text_file", type=str, required=True, help="Path to a UTF-8 txt file containing the user prompt (supports multi-line text).")
    ap.add_argument("--system", type=str, default="你是一个有帮助的视觉问答助手。", help="Optional system prompt.")
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="HF model id.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    return ap.parse_args()




def load_text_prompt(text_file: str) -> str:
    p = Path(text_file)
    if not p.exists():
        raise FileNotFoundError(f"Text file not found: {text_file}")
    prompt = p.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError(f"Text file is empty: {text_file}")
    return prompt

def main() -> None:
    args = parse_args()

    if not args.image.startswith(("http://", "https://")) and not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    user_text = load_text_prompt(args.text_file)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": args.system}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # qwen_vl_utils is required by Qwen2/2.5-VL to parse message image/video fields.
    from qwen_vl_utils import process_vision_info  # type: ignore

    image_inputs, video_inputs = process_vision_info(messages)
    if video_inputs:
        model_inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
    else:
        model_inputs = processor(text=[prompt], images=image_inputs, return_tensors="pt", padding=True)

    if device == "cpu":
        model = model.to(device)
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    do_sample = args.temperature > 0
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
        )

    output_ids = generated_ids[:, model_inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(answer.strip())


if __name__ == "__main__":
    main()