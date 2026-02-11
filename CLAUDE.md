# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lumina-Dimoo is a unified multimodal vision-language model based on the LLaDA (Large Language Diffusion with mAda) architecture. It supports multiple tasks: text-to-image generation, multimodal understanding (MMU), OCR, spatial reasoning, and pointing/counting. The model uses VQ-VAE for image tokenization and MaskGit parallel decoding for generation.

**Base model:** `Alpha-VLLM/Lumina-DiMOO` (HuggingFace)
**Branch note:** `lumina_fsdp_lora` supports FSDP + LoRA simultaneous usage. Previously these were mutually exclusive (`fsdp` + no LoRA, or `none` + LoRA).

## Environment Setup

```bash
conda env create -f lumina.yaml   # Python 3.10, PyTorch 2.3.1
conda activate lumina_dimoo
export PYTHONNOUSERSITE=1          # Avoid user-site-packages conflicts
```

## Common Commands

### Training (Distributed)

All training uses `torch.distributed.run`. The main unified training script is `train/train_unified.py`.

```bash
# Unified multi-task training (generation + understanding + OCR + pointing + position)
python -m torch.distributed.run --nproc_per_node=<N_GPUS> --master_port=29504 \
    train/train_unified.py \
    --batch_size 1 --accum_iter 32 --epochs 999 \
    --lr 2e-5 --precision bf16 --checkpointing \
    --data_parallel fsdp \     # or 'none' for non-FSDP
    --use_lora --lora_rank 128 \
    --use_wandb --wandb_project "lumina-dimoo" \
    --init_from Alpha-VLLM/Lumina-DiMOO \
    --data_config configs/data.yaml \
    --output_dir output/<exp_name>
```

Task-specific scripts: `train/train_generation.py`, `train/train_ocr.py`, `train/train_pointing.py`, `train/train_position.py`. Shell wrappers in `train/*.sh`.

### Inference

```bash
# Text-to-image (single/batch)
python inference/inference_t2i.py --checkpoint <path> --height 512 --width 512 --timesteps 64 --cfg_scale 4.0 --seed 65513 --vae_ckpt Alpha-VLLM/Lumina-DiMOO --output_dir output/

# Batch from prompt file
python inference/inference_t2i_batch.py --prompt_file <path_to_prompts.txt> ...

# Multi-GPU inference
python inference/inference_t2i_multigpu.py ...
torchrun --nproc_per_node=<N> inference/inference_t2i_ddp.py ...

# Multimodal understanding
python inference/inference_mmu.py ...
```

### Evaluation (`evaluation_scripts/`)

All evaluation scripts are in `evaluation_scripts/`, organized by task.

```bash
# Understanding - Spatial (single GPU)
python evaluation_scripts/2d_spatial/evaluate_blink_spatial.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --lora_ckpt_path <lora_dir> --output_dir <output> \
    --steps 20 --gen_length 20 --block_length 20
python evaluation_scripts/2d_spatial/evaluate_cvbench_spatial.py ...

# Understanding - Counting (multi-GPU)
torchrun --nproc_per_node=<N> evaluation_scripts/counting/evaluate_pixmo_multigpu.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --lora_ckpt_path <lora_dir> --output_dir <output> \
    --steps 20 --gen_length 20 --block_length 20
torchrun --nproc_per_node=<N> evaluation_scripts/counting/evaluate_cvbench_counting_multigpu.py ...

# Generation - GenEval (3-stage pipeline)
bash evaluation_scripts/counting/01_geneval_inference_multigpu.sh  # 1) generate images
bash evaluation_scripts/counting/02_geneval_eval_specific_model.sh # 2) evaluate with external tool
python evaluation_scripts/counting/03_summarize_jsonl_results_and_make_confusion_mat.py # 3) aggregate results
```

**Evaluation benchmarks:**

| Task | Script | Dataset | Metric | Multi-GPU |
|------|--------|---------|--------|-----------|
| BLINK Spatial | `2d_spatial/evaluate_blink_spatial.py` | `BLINK-Benchmark/BLINK` | Accuracy (A/B) | X |
| CVBench Spatial | `2d_spatial/evaluate_cvbench_spatial.py` | `nyu-visionx/CV-Bench` (Relation) | Accuracy | X |
| CVBench Counting | `counting/evaluate_cvbench_counting.py` | `nyu-visionx/CV-Bench` (Counting) | Accuracy | O |
| Pixmo Counting | `counting/evaluate_pixmo.py` | `Jiwon-Kang/pixmo-count-filtered-imgContained` | MAD | O |
| GenEval | `counting/01→02→03` pipeline | Text prompts → generated images | Counting accuracy | O |

**Common eval arguments:** `--checkpoint`, `--vae_ckpt`, `--output_dir`, `--steps`, `--gen_length`, `--block_length`, `--lora_ckpt_path`, `--seed`

### Resume Training

Pass `--wandb_run_id <run_id>` and `--resume_path <checkpoint_dir>` to resume from a checkpoint.

## Architecture

### Model Stack

```
LLaDAForMultiModalGeneration (model/modeling_xllmx_dimoo.py)
  └── LLaDAModelLM (model/modeling_llada.py)
        └── LLaDAConfig (model/configuration_llada.py)
```

- `LLaDAForMultiModalGeneration`: Wrapper that handles variable-length batch padding, attention mask creation, and cross-entropy loss computation. Supports LoRA wrapping/unwrapping.
- `LLaDAModelLM`: Core transformer with activation checkpointing and FSDP block-level wrapping.

### Key Modules

| Directory | Purpose |
|-----------|---------|
| `model/` | Model definitions (LLaDA transformer + multimodal wrapper) |
| `train/` | Training scripts per task + `train_unified.py` for multi-task |
| `inference/` | Inference scripts for T2I, I2I, MMU |
| `evaluation_scripts/` | Evaluation scripts by task: `2d_spatial/`, `counting/`, `ocr/` |
| `generators/` | Generation algorithms (MaskGit decoding, text understanding, I2I) |
| `utils/` | Image encoding/decoding (VQ-VAE), generation utils (cosine schedule, Gumbel sampling), prompt templates |
| `xllmx/` | Distributed training framework (solver, dataset, sampler, checkpointing, LR scheduling) |
| `data/` | Data item processors for dataset loading |
| `configs/` | YAML data configs (T2I, MMU, I2I data paths) |
| `pre_tokenizer/` | Offline pre-tokenization scripts |
| `VLMEvalKit/` | Git submodule for standardized VLM evaluation benchmarks |

### Training Framework (`xllmx/`)

- `solvers/finetune/finetune.py` — `FinetuneSolverBase`: handles distributed setup (FSDP/DDP), checkpoint management, logging, LR scheduling.
- `data/dataset.py` — `FinetuneConversationDataset`: multi-source dataset from YAML config with cache-on-disk support.
- `data/sampler.py` — `FinetuneDistSampler`: distributed sampler for DDP/FSDP.
- `data/item_processor.py` — `ItemProcessorBase`: abstract class for per-item data transformation.

### Special Tokens (from `config.py`)

Defined in `SPECIAL_TOKENS` dict: `mask_token` (126336), `newline_token` (126084), `image_token_offset` (126356), `answer_start`/`answer_end` (126354/126355), `boi`/`eoi` (126349/126350). Padding token is 126339 (defined in training scripts).

### Generation Pipeline

Image generation uses MaskGit parallel decoding (`generators/image_generation_generator.py`):
1. Start with fully masked image token sequence
2. Iteratively predict tokens, keep high-confidence predictions, remask low-confidence ones
3. Cosine noise schedule controls masking ratio per step
4. Classifier-free guidance (CFG) with configurable scale
5. VQ-VAE decodes final token sequence to RGB image

### Image Tokenization (`utils/image_utils.py`)

Images are encoded via VQ-VAE into discrete tokens. Newline tokens (`126084`) are inserted between rows at fixed intervals (`newline_every=16`). The `encode_img_with_breaks` and `decode_vq_to_image` functions handle this conversion.

## Key Training Arguments

- `--data_parallel fsdp|none` — FSDP sharding vs no sharding
- `--use_lora` / `--lora_rank` / `--lora_target_modules` — LoRA fine-tuning config
- `--wo_lm_head` — Exclude LM head from LoRA (common pattern)
- `--checkpointing` — Activation checkpointing for memory savings
- `--precision bf16` — Mixed precision training
- `--mode gen|und|both` — Task mode for pointing/position training
- `--task counting|pointing|position|ocr` — Task selection for unified training
- `--validation_interval` / `--save_iteration_interval` — Validation and checkpoint frequency

## WandB Integration

Projects: `lumina-generation-exp`, `lumina-dimoo`, `lumina-pointing`. Use `--use_wandb --wandb_project <name> --wandb_run_name <name>` to enable.
