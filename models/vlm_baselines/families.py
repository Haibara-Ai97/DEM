from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
)

from peft import PeftModel


def safe_auto_processor_from_pretrained(model_id: str, **kwargs):
    """Call AutoProcessor.from_pretrained with best-effort compatibility across Transformers versions."""
    try:
        return AutoProcessor.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("use_fast", None)
        return AutoProcessor.from_pretrained(model_id, **kwargs)


def _base_messages(system_text: str) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    if system_text:
        msgs.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
    return msgs


def build_messages_prompt(sample: Dict[str, Any], family: str) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Returns:
      messages (for apply_chat_template)
      images_payload (for processor), format depends on family:
        - qwen2*: uses qwen_vl_utils.process_vision_info(messages) => images_payload is ignored here (empty)
        - others: images_payload contains PIL images (loaded later in collator)
    """
    system_text = sample.get("system", "")
    user_text = sample["user"]
    img_path = sample["image_path"]

    msgs = _base_messages(system_text)

    if family in ("qwen2_vl", "qwen2_5_vl"):
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": user_text},
                ],
            }
        )
        return msgs, []

    if family == "llava_1_5":
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image"},
                ],
            }
        )
        return msgs, [img_path]

    if family == "idefics2":
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        )
        return msgs, [img_path]

    if family == "phi3v":
        placeholder = "<|image_1|>\n"
        msgs.append({"role": "user", "content": placeholder + user_text})
        return msgs, [img_path]

    raise ValueError(f"Unknown family: {family}")


def build_messages_full(sample: Dict[str, Any], family: str) -> Tuple[List[Dict[str, Any]], List[Any]]:
    msgs, imgs = build_messages_prompt(sample, family)
    assistant_text = sample["assistant"]
    if family == "phi3v":
        msgs.append({"role": "assistant", "content": assistant_text})
    else:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
    return msgs, imgs


def infer_lora_targets(model: torch.nn.Module, family: str) -> List[str]:
    """Infer likely attention/MLP projection names for LoRA insertion by model family."""
    patterns_by_family = {
        "qwen2_vl": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "qwen2_5_vl": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "llava_1_5": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "idefics2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "phi3v": ["qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj", "dense", "fc1", "fc2", "down_proj", "up_proj"],
    }
    patterns = patterns_by_family.get(family, ["q_proj", "k_proj", "v_proj", "o_proj"])

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        for pat in patterns:
            if name.endswith(pat) or (("." + pat + ".") in ("." + name + ".")) or name.split(".")[-1] == pat:
                targets.append(name.split(".")[-1])
                break
    targets = sorted(set(targets))

    if not targets:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                targets.append(name.split(".")[-1])
        targets = sorted(set(targets))

    if not targets:
        raise RuntimeError("Could not infer LoRA targets (no Linear layers found).")

    return targets


def load_model_and_processor(
    model_id: str,
    family: str,
    adapter_dir: Optional[str],
    bf16: bool,
    qwen_min_pixels: int = 0,
    qwen_max_pixels: int = 0,
) -> Tuple[Any, Any, Any]:
    torch_dtype = torch.bfloat16 if bf16 else torch.float16

    if family == "llava_1_5":
        from transformers import LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=qwen_min_pixels or None,
            max_pixels=qwen_max_pixels or None,
        )
        tokenizer = processor.tokenizer
    elif family in ("qwen2_vl", "qwen2_5_vl", "idefics2"):
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = getattr(processor, "tokenizer", AutoTokenizer.from_pretrained(model_id, trust_remote_code=True))
    elif family == "phi3v":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = processor.tokenizer
    else:
        raise ValueError(f"Unknown family: {family}")

    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval().cuda()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model, processor, tokenizer
