# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LoRA fine-tuning framework for **DeepSeek Janus-Pro-7B**, a unified multimodal model supporting both **image understanding** (counting, pointing) and **text-to-image generation**. Training uses HuggingFace Trainer with PEFT/LoRA, WandB logging, and multi-GPU DDP.

## Common Commands

### Environment Setup
```bash
conda create -n janus-pro-lora python=3.10 -y && conda activate janus-pro-lora
pip install -r requirements.txt
modelscope download --model deepseek-ai/Janus-Pro-7B --local_dir ./Janus-Pro-7B
```

### Training
```bash
# Via shell script (accepts MODE TASK [RESUME_CKPT] arguments)
bash script/train_counting_transformer_lora128.sh und counting
bash script/train_counting_transformer_lora128.sh gen
bash script/train_counting_transformer_lora128.sh both counting
bash script/train_counting_transformer_lora128.sh both counting ./checkpoints/.../step-500  # resume

# Generation-only (dedicated script)
bash script/train_counting_gen_transformer_lora128.sh

# Direct launch (multi-GPU)
accelerate launch --num_processes 2 train_counting.py \
    --task counting --mode und \
    --data_path ./data/pixmo_processed \
    --model_path deepseek-ai/Janus-Pro-7B \
    --tuning_mode transformer_lora \
    --batch_size 2 --lr 1e-4 --epochs 3

# --task: counting, ocr (data domain — orthogonal to mode)
# --mode: und (understanding), gen (generation), both
```

### Smoke Test (3 modes)
```bash
bash script/test_three_modes.sh   # runs und/gen/both for 3 steps each
```

### VQ Encode/Decode Test
```bash
python test_scripts/test_vq_encode_decode.py \
    --model_path deepseek-ai/Janus-Pro-7B \
    --image_path <test_image.png> \
    --output_dir output/debug_vq
```

## Architecture

### Training Entry Points
- **`train_counting.py`** — Main training script (--task and --mode are orthogonal)
  - `--task {counting,ocr}` — Data domain (determines dataset format/loading)
  - `--mode {und,gen,both}` — Training mode (understanding, generation, or both)
- **`train_points.py`** — Pointing task variant with coordinate regression
- **`script/`** — Shell launchers with preset hyperparameters (accept `MODE TASK [RESUME_CKPT]` args)

### Model (`janus/models/`)
- `modeling_vlm.py` — Base `MultiModalityCausalLM` with config classes
- `clip_encoder.py` / `siglip_vit.py` — SigLIP vision encoder (understanding)
- `vq_model.py` — VQ-VAE with codebook_size=16384 (generation, encodes images to 576 discrete tokens)
- `projector.py` — MLP aligner between vision and language features
- `processing_vlm.py` — `VLChatProcessor` for tokenizing multimodal conversations

### Key Class: `EnhancedMultiModalModel` (in `train_counting.py`)
Wraps the base model with a custom `forward()` that branches based on input:
- **Understanding path**: image → SigLIP → aligner → merged with text embeddings → LLM → text output
- **Generation path**: text → LLM embeddings → gen_embed/gen_aligner → gen_head → 576 VQ token logits (16384-way classification)

### Data Flow
- `StreamingDatasetWrapper` — On-the-fly image loading from URLs
- `collate_fn()` — Understanding batches (tokenize Q&A, create image masks)
- `collate_fn_generation()` — Generation batches (VQ-encode images to 576 tokens, create gen_token_mask)
- `collate_fn_both()` — Routes samples to the appropriate collator based on field presence

### Trainable Parameter Strategy

| Component | Understanding | Generation/Both |
|-----------|:---:|:---:|
| Language Model (Llama) | **LoRA** | **LoRA** |
| gen_head, gen_embed, gen_aligner | Frozen | **Trainable** |
| vision_model, aligner, VQ-VAE | Frozen | Frozen |

### Checkpoints
```
checkpoints/<run_name>/
├── adapter_config.json / adapter_model.safetensors  # LoRA weights
├── gen_components.pt                                 # gen_head + gen_embed + gen_aligner (generation/both only)
└── final_model/
```

### Validation
`ValidationCallback` runs at `--log_freq` step intervals:
- Understanding: evaluates accuracy on val split, logs predictions to WandB
- Generation: autoregressively generates images from 4 fixed prompts, logs to WandB

## Key Arguments (`train_counting.py`)
- `--task {counting,ocr}` — Data domain (determines dataset format, orthogonal to mode)
- `--mode {und,gen,both}` — Training mode (understanding, generation, or both)
- `--tuning_mode {transformer_lora,transformer,transformer_ONLY,full,lora}` — What to train
- `--lora_r` / `--lora_alpha` — LoRA rank and scaling (defaults: 16, 32)
- `--data_path` — Understanding dataset path (required for mode=und/both)
- `--gen_data_path` — Generation dataset path, HF dataset or local (required for mode=gen/both)
- `--gen_img_size` — VQ encoding image size (default: 384)
- `--max_steps` — Max training steps (-1 for unlimited)
- `--resume_checkpoint` — Path to checkpoint directory for resuming training
- `--use_8bit_adam` — Use 8-bit AdamW from bitsandbytes (flag)
- `--run_name` — WandB run name override

## Notes
- GPU VRAM: 32GB+ recommended
- Training uses bf16 (falls back to fp16)
- `ddp_find_unused_parameters=True` is required for gen/both modes
- Generation dataset samples need `image` + one of `text`/`caption`/`prompt` fields
- Understanding dataset samples need `image` + `question_count`/`question` + answer fields
- Project language: comments and docs are mixed Korean/English
