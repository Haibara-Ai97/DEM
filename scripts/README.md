# Scripts Overview

This directory contains the latest training, fine-tuning, evaluation, and data-prep scripts after consolidation. Each folder groups scripts by workflow stage.

## Stage 1 (encoder + adapter alignment)

- `stage1/train_stage1.py`
  - **Purpose**: Train DEM encoder + DA adapter with cache-based supervision for stage-1 alignment.
  - **Command**:
    ```bash
    python -m scripts.stage1.train_stage1 --config configs/stage1/default.yaml
    ```

- `stage1/smoke_test_stage1.py`
  - **Purpose**: Quick smoke test to validate stage-1 pipeline dependencies (CLIP, cache, model wiring).
  - **Command**:
    ```bash
    python -m scripts.stage1.smoke_test_stage1 --workdir _smoke_stage1
    ```

## Stage 2 (LLM/VLM fine-tuning)

- `stage2/train_lora_stage2_qwenstyle_trainer_v5_joint_encoder.py`
  - **Purpose**: Stage-2 LoRA fine-tuning with optional joint tuning of DEM encoder + DA adapter.
  - **Command**:
    ```bash
    torchrun --nproc_per_node=8 scripts/stage2/train_lora_stage2_qwenstyle_trainer_v5_joint_encoder.py \
      --train_jsonl /path/to/train.jsonl \
      --output_dir /path/to/output \
      --llm_name Qwen/Qwen2.5-7B-Instruct \
      --stage1_ckpt /path/to/stage1.pt \
      --encoder_ckpt /path/to/encoder_best.pth \
      --use_lora \
      --no_freeze_encoder \
      --encoder_lr 2e-5 \
      --adapter_lr 1e-4 \
      --learning_rate 2e-4 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --max_text_len 512 \
      --bf16
    ```

## VLM Baselines

- `vlm_baselines/train.py`
  - **Purpose**: LoRA SFT for mainstream VLM baselines (Qwen2/2.5-VL, LLaVA-1.5, Idefics2, Phi3V).
  - **Command**:
    ```bash
    torchrun --nproc_per_node=8 -m scripts.vlm_baselines.train \
      --config configs/vlm_baselines/default.yaml \
      --model_key qwen2_5_vl \
      --train_jsonl /path/train.jsonl \
      --valid_jsonl /path/valid.jsonl \
      --output_dir runs/qwen25vl_lora
    ```

- `vlm_baselines/eval.py`
  - **Purpose**: Unified evaluation for baseline VLM tasks (yesno/multilabel/count/grid/json).
  - **Command**:
    ```bash
    python -m scripts.vlm_baselines.eval \
      --config configs/vlm_baselines/default.yaml \
      --model_key qwen2_5_vl \
      --adapter_dir runs/qwen25vl_lora \
      --jsonl /path/test.jsonl \
      --split test
    ```

## Data Preparation

- `data/make_stage1_csv_from_yolo.py`
  - **Purpose**: Build stage-1 CSV index from YOLO-format datasets.
  - **Command**:
    ```bash
    python -m scripts.data.make_stage1_csv_from_yolo --data_root /path/to/yolo --output_csv stage1.csv
    ```

- `data/build_stage1_clip_cache.py`
  - **Purpose**: Precompute CLIP cache for stage-1 alignment.
  - **Command**:
    ```bash
    python -m scripts.data.build_stage1_clip_cache --config configs/stage1/default.yaml
    ```

- `data/precompute_llm_phrase_embeds.py`
  - **Purpose**: Precompute LLM phrase embeddings for domain vocabulary.
  - **Command**:
    ```bash
    python -m scripts.data.precompute_llm_phrase_embeds --vocab /path/domain_vocab.txt --output /path/llm_phrase_embeds.pt
    ```

- `data/build_stage2_sft_from_yolo.py`
  - **Purpose**: Build stage-2 SFT JSONL from YOLO-format annotations.
  - **Command**:
    ```bash
    python -m scripts.data.build_stage2_sft_from_yolo --data_root /path/to/yolo --output_jsonl stage2_sft.jsonl
    ```

## Detection

- `detection/train_det.py`
  - **Purpose**: Train Faster R-CNN detection baselines and DEM-encoder variants.
  - **Command**:
    ```bash
    python -m scripts.detection.train_det --config apps/detection/configs/default.yaml --data_root /path/dataset
    ```

- `detection/eval_det.py`
  - **Purpose**: Evaluate detection models on YOLO-format datasets (COCO metrics).
  - **Command**:
    ```bash
    python -m scripts.detection.eval_det --data_root /path/dataset --split valid --ckpt outputs/model_best.pth
    ```

## MLLM Evaluation

- `eval/eval_mllm.py`
  - **Purpose**: End-to-end evaluation for DEM + DAAdapter + LLM pipeline on QA tasks.
  - **Command**:
    ```bash
    python -m scripts.eval.eval_mllm --config /path/eval_config.yaml
    ```