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
    ap.add_argument(
        "--save_visual_dir",
        type=str,
        default="",
        help="Optional output dir to save visual-encoder feature/attention tensors during inference.",
    )
    return ap.parse_args()




def load_text_prompt(text_file: str) -> str:
    p = Path(text_file)
    if not p.exists():
        raise FileNotFoundError(f"Text file not found: {text_file}")
    prompt = p.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError(f"Text file is empty: {text_file}")
    return prompt


def _save_visual_artifacts(
    model: Qwen2_5_VLForConditionalGeneration,
    model_inputs: dict[str, torch.Tensor],
    save_dir: str,
) -> None:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pixel_values = model_inputs.get("pixel_values")
    image_grid_thw = model_inputs.get("image_grid_thw")
    if pixel_values is None:
        print("[warn] save_visual_dir is set but no pixel_values found in model_inputs, skip saving.")
        return

    torch.save(pixel_values.detach().cpu(), out_dir / "pixel_values.pt")
    if image_grid_thw is not None:
        torch.save(image_grid_thw.detach().cpu(), out_dir / "image_grid_thw.pt")

    visual = getattr(model, "visual", None)
    if visual is None:
        print("[warn] model.visual not found, only saved pixel_values/image_grid_thw.")
        return

    vis_kwargs: dict[str, torch.Tensor] = {"pixel_values": pixel_values}
    if image_grid_thw is not None:
        vis_kwargs["grid_thw"] = image_grid_thw

    with torch.no_grad():
        try:
            vis_out = visual(
                **vis_kwargs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
        except TypeError:
            vis_out = visual(**vis_kwargs)

    hidden_states = getattr(vis_out, "hidden_states", None)
    attentions = getattr(vis_out, "attentions", None)
    if torch.is_tensor(vis_out):
        torch.save(vis_out.detach().cpu(), out_dir / "vision_output.pt")

    if hidden_states is not None:
        hidden_cpu = tuple(h.detach().cpu() for h in hidden_states)
        torch.save(hidden_cpu, out_dir / "vision_hidden_states.pt")
        torch.save(hidden_cpu[-1], out_dir / "vision_last_hidden_state.pt")

        if image_grid_thw is not None and hidden_cpu[-1].ndim == 3:
            t, h, w = image_grid_thw[0].detach().cpu().tolist()
            if t * h * w == hidden_cpu[-1].shape[1]:
                feat_2d = hidden_cpu[-1][0].view(t, h, w, -1).mean(dim=0).permute(2, 0, 1).contiguous()
                torch.save(feat_2d, out_dir / "vision_feature_map_tmean.pt")

    if attentions is not None:
        attn_cpu = tuple(a.detach().cpu() for a in attentions)
        torch.save(attn_cpu, out_dir / "vision_attentions.pt")
        torch.save(attn_cpu[-1], out_dir / "vision_last_layer_attention.pt")

    print(f"[info] visual artifacts saved to: {out_dir.resolve()}")

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

    if args.save_visual_dir:
        _save_visual_artifacts(model, model_inputs, args.save_visual_dir)

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
