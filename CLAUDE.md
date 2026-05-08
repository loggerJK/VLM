# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMaDA (Multimodal Large Diffusion Language Models) is a unified multimodal diffusion foundation model that handles text generation, multimodal understanding (MMU), and text-to-image (T2I) generation in a single architecture. The backbone is LLaDA (a masked diffusion LLM), extended with a MAGVIT-v2 VQ-VAE for image tokenization.

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `transformers==4.46.0`, `accelerate`, `deepspeed`, `omegaconf`, `webdataset`, `wandb`, `lightning`, `gradio`.

## Common Commands

**Gradio demo:**
```bash
python app.py
```

**Text generation inference:**
```bash
python generate.py
```

**Multimodal understanding inference** (results logged to wandb):
```bash
wandb login
python3 inference_mmu.py \
  config=configs/mmada_demo.yaml \
  mmu_image_root=./mmu_validation \
  mmu_prompts_file=./mmu_validation/prompts_with_vqa.json
```

**Text-to-image inference:**
```bash
python3 inference_t2i.py config=configs/mmada_demo.yaml batch_size=1 \
  validation_prompts_file=validation_prompts/text2image_prompts.txt \
  guidance_scale=3.5 generation_timesteps=15 mode='t2i'
```

**Training** (always launched via `accelerate launch`):
```bash
# Stage 1.1 - ImageNet pretraining
accelerate launch --config_file <accel_cfg> --main_process_port=8888 \
  training/train_mmada.py config=configs/mmada_pretraining_stage1_llada_instruct.yaml

# Stage 1.2 - Image-Text dataset
accelerate launch --config_file <accel_cfg> --main_process_port=8888 \
  training/train_mmada_stage2.py config=configs/mmada_pretraining_stage2_llada_instruct.yaml

# Stage 1.3 - Text instruction following
accelerate launch --config_file <accel_cfg> --main_process_port=8888 \
  training/train_mmada_stage3.py config=configs/mmada_pretraining_stage3_llada_instruct.yaml

# Stage 2.1 - Mix-CoT text-only
accelerate launch --config_file <accel_cfg> --main_process_port=8888 \
  training/train_mmada_cot_sft.py config=configs/mmada_pretraining_stage3_llada_instruct_512_cot.yaml

# Stage 2.2 - Mix-CoT with multimodal reasoning
accelerate launch --config_file <accel_cfg> --main_process_port=8888 \
  training/train_mmada_stage4.py config=configs/mmada_pretraining_stage4_llada_instruct.yaml
```

**VLM Evaluation:**
```bash
cd evaluation/VLMEvalKit && pip install -e .
CUDA_VISIBLE_DEVICES=0 python run.py --data {dataset_name} --model MMaDA-MixCoT
# Multi-GPU:
torchrun --nproc-per-node=8 --master-port=54321 run.py --data {dataset_name} --model MMaDA-MixCoT
```

**LLM Evaluation:**
```bash
cd evaluation/lm && pip install lm-eval && bash eval.sh
```

## Architecture

### Core Model Stack

- **`models/modeling_llada.py`** — `LLaDAModelLM`: the base masked diffusion transformer (LLaDA), a bidirectional transformer variant operating over masked token sequences.
- **`models/modeling_mmada.py`** — `MMadaModelLM(LLaDAModelLM)`: extends LLaDA with T2I generation (`t2i_generate`), MMU generation (`mmu_generate`, `mmu_generate_fast`), and training forward passes (`forward_process`, `forward_process_with_r2i`). Registered with HuggingFace AutoModel.
- **`models/modeling_magvitv2.py`** — `MAGVITv2`: VQGAN image tokenizer/detokenizer. Encodes 512×512 images into 1024 discrete VQ tokens (codebook size 8192) used as the image vocabulary.
- **`models/configuration_llada.py`** — `LLaDAConfig`, `MMadaConfig`: model configs. `MMadaConfig` stores `llm_vocab_size`, `codebook_size`, `num_vq_tokens`, etc.
- **`models/sampling.py`** — Noise schedules (cosine, linear) and `mask_by_random_topk` for MaskGIT-style iterative decoding.

### Unified Token Vocabulary

The model uses a single vocabulary spanning:
- LLM text tokens (`llm_vocab_size` ≈ 126,464)
- Special control tokens: `<|soi|>`, `<|eoi|>`, `<|t2i|>`, `<|mmu|>`, `<|r2i|>`, etc. (defined in `training/prompting_utils.py` and `models/__init__.py`)
- Image VQ tokens: offset by `llm_vocab_size` (total `new_vocab_size` ≈ 134,656)
- Mask token ID: `126336`

### Prompting & Tokenization

**`training/prompting_utils.py`** — `UniversalPrompting` wraps the LLaDA tokenizer and manages the unified input format. It inserts task-specific control tokens (`<|t2i|>`, `<|mmu|>`, `<|soi|>`, `<|eoi|>`, etc.) before/after image and text regions. All training scripts instantiate `UniversalPrompting` before building batches.

### Training Data Pipeline

**`training/data.py`** — `Text2ImageDataset`: webdataset-based loader for image-text pairs (SA-1B, CC12M, LAION). Handles external captions via JSON/CSV sidecar files.

**`training/imagenet_dataset.py`** — `ImageNetDataset`: for Stage 1.1.

**`parquet/my_dataset.py`** — `RefinedWebDataset`, `ChatDataset`, `VQADataset`: parquet-based loaders for text (FalconRefinedWeb), instruction chat, and VQA data.

Training batches mix T2I, LM, and MMU samples within a single forward pass. Batch composition is controlled by `batch_size_t2i`, `batch_size_lm`, `batch_size_mmu` in YAML configs.

### Training Loss

`forward_process` in `MMadaModelLM` computes three losses jointly:
- `loss_t2i`: cross-entropy on masked image token positions (full-sequence bidirectional attention via `t2i_masks`)
- `loss_lm`: masked diffusion loss on text positions, normalized by `p_mask` and answer length
- `loss_mmu`: same as `loss_lm` but for image-conditioned text answers

Final loss = `t2i_coeff * loss_t2i + lm_coeff * loss_lm + mmu_coeff * loss_mmu`.

### Configuration

All scripts use `omegaconf`-based YAML configs loaded via `training/utils.py::get_config()`. CLI overrides use `key=value` syntax (e.g., `config=configs/mmada_demo.yaml batch_size=1`). Config files live in `configs/`; accelerate configs live in `accelerate_configs/`.

### Accelerate / DeepSpeed

Multi-GPU training uses HuggingFace Accelerate with optional DeepSpeed ZeRO-2/ZeRO-3. Pre-built accelerate configs in `accelerate_configs/` cover 1-GPU, 1-node-8-GPU, and 8-node-8-GPU setups.

## Key Checkpoints (HuggingFace)

- `Gen-Verse/MMaDA-8B-Base` — after pretraining + instruction tuning
- `Gen-Verse/MMaDA-8B-MixCoT` — after Mix-CoT fine-tuning
- `showlab/magvitv2` — MAGVIT-v2 VQ model (always loaded separately from the LLM)
- Tokenizer: `GSAI-ML/LLaDA-8B-Instruct`
