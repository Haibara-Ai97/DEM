# VLM Baseline LoRA Runner (Concrete QA JSONL)

VLM baseline scripts have been archived into `dem/vlm_baselines/`:
- `python -m dem.vlm_baselines.train` : LoRA SFT for several mainstream 4–8B VLMs/MLLMs
- `python -m dem.vlm_baselines.eval`  : unified evaluation for tasks (yesno/multilabel/count/grid/json)

## Supported baseline models (IDs)

Recommended 4–5 baselines that are widely used and actively maintained:

1. `Qwen/Qwen2.5-VL-7B-Instruct`  (family: `qwen2_5_vl`)
2. `Qwen/Qwen2-VL-7B-Instruct`    (family: `qwen2_vl`)
3. `llava-hf/llava-1.5-7b-hf`     (family: `llava_1_5`)
4. `HuggingFaceM4/idefics2-8b`    (family: `idefics2`)
5. `microsoft/Phi-3.5-vision-instruct` (family: `phi3v`, smaller but strong efficiency baseline)

## Dependencies

Minimum:
- torch
- transformers
- peft
- accelerate
- pillow

Additional for Qwen2/2.5-VL:
- qwen-vl-utils

## Example training commands (multi-GPU)

Assume you have:
- train jsonl: `qa_train.jsonl`
- valid jsonl: `qa_valid.jsonl`

### Qwen2.5-VL
torchrun --nproc_per_node=8 -m dem.vlm_baselines.train \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --family qwen2_5_vl \
  --train_jsonl qa_train.jsonl \
  --valid_jsonl qa_valid.jsonl \
  --output_dir runs/qwen25vl_lora \
  --bf16 --gradient_checkpointing \
  --per_device_train_batch_size 1 --grad_accum 8 \
  --max_length 512

### LLaVA-1.5
torchrun --nproc_per_node=8 -m dem.vlm_baselines.train \
  --model_id llava-hf/llava-1.5-7b-hf \
  --family llava_1_5 \
  --train_jsonl qa_train.jsonl \
  --valid_jsonl qa_valid.jsonl \
  --output_dir runs/llava15_lora \
  --bf16 --gradient_checkpointing \
  --per_device_train_batch_size 1 --grad_accum 8

## Example evaluation

python -m dem.vlm_baselines.eval \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --family qwen2_5_vl \
  --adapter_dir runs/qwen25vl_lora \
  --jsonl qa_test.jsonl \
  --split test

## Config presets

Unified defaults live in `configs/vlm_baselines/default.yaml`. You can use them via:

```bash
python -m dem.vlm_baselines.train --config configs/vlm_baselines/default.yaml --model_key qwen2_5_vl ...
python -m dem.vlm_baselines.eval --config configs/vlm_baselines/default.yaml --model_key qwen2_5_vl ...
```
