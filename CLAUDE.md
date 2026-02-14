# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fine-tuning DeepSeek Janus-Pro-7B (multimodal vision-language model) for visual counting and pointing tasks using the Pixmo dataset. Uses LoRA for parameter-efficient fine-tuning, with WandB for experiment tracking.

## Environment Setup

```bash
conda create -n janus-pro-lora python=3.10 -y
conda activate janus-pro-lora
pip install -r requirements.txt
```

Key dependencies: torch 2.2.2, transformers 4.48.1, peft 0.14.0, wandb, accelerate.

## Training Commands

Training is launched via shell scripts in `script/`:

```bash
# Single-GPU (falls back to pdb debugger)
bash script/train_counting_0.sh

# Multi-GPU with LoRA r=128 (primary configuration)
bash script/train_counting_transformer_lora128.sh

# Pointing task
bash script/point/train_points_transformer_ONLY.sh
```

Multi-GPU runs use `accelerate launch`. GPU selection is via `CUDA_VISIBLE_DEVICES` in the scripts. The `.env` file contains `WANDB_API_KEY`.

### Key training arguments (passed to `train_counting.py` / `train_points.py`)

- `--tuning_mode`: `transformer_lora` | `transformer` | `transformer_ONLY` | `full`
- `--model_path`: HF model ID (default `deepseek-ai/Janus-Pro-7B`)
- `--data_path`: HF dataset ID (e.g. `heez/pixmo-point-count-gen-und`)
- `--task`: `counting` (uses `question_count`/`answer_count` columns) or `pointing` (uses `question`/`answer` columns)
- `--lora_r`, `--lora_alpha`: LoRA hyperparameters

## Linting

The `transformers/` subdirectory uses Ruff (line-length 119, target py310). The main project code does not have a configured linter.

## Architecture

### Model Pipeline

```
Image → SigLIP Vision Encoder → MLP Aligner/Projector → [merged with text embeddings] → LLaMA Language Model → Output
```

### Key Classes

- **`MultiModalityCausalLM`** (`janus/models/modeling_vlm.py`): Base model combining vision encoder, aligner, and LLaMA language model. Registered as `MultiModalityCausalLM` in HF AutoModel.
- **`EnhancedMultiModalModel`** (`train_counting.py`, `train_points.py`): Subclass adding a `forward()` method that replaces text token embeddings at image positions with vision-aligned features via `image_token_masks`. Each training script has its own copy.
- **`VLChatProcessor`** (`janus/models/processing_vlm.py`): Handles conversation formatting and batch tokenization via `batchify()`.
- **`StreamingDatasetWrapper`** (`train_counting.py`): Wraps HF datasets to download images on-the-fly from URLs, with fallback to random sampling on failure.
- **`ValidationCallback`** (`train_counting.py`): HF Trainer callback for periodic validation, logs accuracy/samples to WandB and saves checkpoints.

### Training Data Flow

1. `collate_fn` formats each sample as a conversation (`<User>` with `<image_placeholder>` + question, `<Assistant>` with answer)
2. `VLChatProcessor.batchify()` tokenizes and produces `input_ids`, `pixel_values`, `image_token_masks`
3. Labels are masked so loss is computed only on assistant response tokens (everything before the assistant split point gets `label = -100`)

### LoRA Configuration

Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj`. Applied to the language model only. Scripts test ranks of 4, 16, and 128.

### Fine-tuning Modes

- `transformer_lora`: LoRA on language model transformer layers (primary mode)
- `transformer`: Full fine-tune of language model, freeze vision
- `transformer_ONLY`: Fine-tune transformer, freeze everything else
- `full`: Fine-tune entire model

## Project Layout

- `train_counting.py` / `train_points.py` — Main training entry points for counting and pointing tasks
- `janus/models/` — Model architecture (vision encoder, projector, multimodal model, processor)
- `janus/utils/` — Conversation templates and image I/O
- `script/` — Training launch scripts with hyperparameter configurations
- `analysis/` — Dataset analysis scripts and visualizations
- `checkpoints/` — Saved model checkpoints (gitignored)
- `transformers/` — Vendored Hugging Face transformers fork
- `default_config.yaml` — Accelerate config (multi-GPU, bf16 mixed precision)

## Notes

- Comments in the codebase are primarily in Korean
- Hardware requirement: ~32GB GPU VRAM minimum for 7B model fine-tuning
- The `janus/__init__.py` patches `collections` module for Python 3.10+ compatibility
