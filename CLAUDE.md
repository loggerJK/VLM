# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BLIP3-o is a unified multimodal model that combines autoregressive language modeling (Qwen2.5-VL) with diffusion-based image generation. It diffuses semantically rich CLIP image features (not VAE features or raw pixels). Available in 4B and 8B parameter variants.

Paper: http://arxiv.org/abs/2505.09568

## Setup & Installation

```bash
# setup.sh creates the environment
conda create -n blip3o python=3.11 -y
conda activate blip3o
uv pip install --upgrade pip setuptools
uv pip install -r requirements.txt
uv pip install -e .
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```

Key version constraints: PyTorch 2.3.0, transformers 4.51.3, xformers 0.0.26.post1, timm 0.6.13. The `diffusers` dependency is installed from a custom fork (JiuhaiChen/diffusers).

## Common Commands

### Training (single node, 8 GPUs)
```bash
bash run.sh          # Pretraining from Qwen2.5-VL-7B-Instruct
bash sft.sh          # Supervised fine-tuning from a pretrained BLIP3o checkpoint
```
Both scripts call `torchrun --nproc_per_node=8 blip3o/train/train_mem.py` with DeepSpeed ZeRO-1. Edit environment variables at the top of each script: `HF_HOME`, `OUTPUT_FOLDER`, `IMG_FOLDER`.

### Training (multi-node via SLURM)
```bash
sbatch slurm.sh      # 4 nodes x 8 GPUs, uses srun + torchrun
```

### Inference (text-to-image generation)
```bash
python inference.py   # Edit model_path inside the script
```
Loads the model via `load_pretrained_model()`, then uses a custom DiffusionPipeline (`pipeline_llava_gen`) from the `diffusion-decoder` subdirectory of the model checkpoint.

### Evaluation
```bash
# Image understanding (lmms-eval)
cd eval/lmms-eval && pip install -e .
bash understanding_eval.sh  # Set --model_args pretrained="your/model/path" and --tasks

# Image generation (GenEval/DBP/WISE)
bash eval/geneval/generation.sh  # Set HF_HOME, model path, N_CHUNKS
# Then run GenEval toolkit for scoring
```

## Architecture

### Module Organization (`blip3o/`)

The model uses a **mixin-based design** with multiple inheritance:

- **`model/blip3o_arch.py`** — Core meta-model classes (`blip3oMetaModel`, `blip3oMetaForCausalLM`). Defines the multimodal forward pass, initializes vision towers, the diffusion transformer (DiT), latent queries, noise scheduler, and projectors.
- **`model/language_model/blip3o_qwen.py`** — Primary model: combines `Qwen2_5_VLForConditionalGeneration` + `blip3oMetaModel`. This is the production model class.
- **`model/language_model/blip3o_qwen_inference.py`** — Inference-optimized variant of the Qwen model.
- **`model/language_model/blip3o_llama.py`** — Experimental LLaMA-3 backend.

### Key Components

| Component | Files | Purpose |
|-----------|-------|---------|
| Vision encoders (understanding) | `model/multimodal_encoder/` | EVA-CLIP, SigLIP, OpenCLIP, ImageBind |
| Generation vision tower | `model/multimodal_encoder/` | EVA-CLIP E-14-plus encodes target images for diffusion |
| Multimodal projectors | `model/multimodal_projector/builder.py` | Linear or MLP (2x/3x gelu) projections between vision and LLM spaces |
| Diffusion transformer | `model/nextdit_crossattn.py`, `model/lumina_nextdit2d.py` | NextDiT with cross-attention; Flow Matching scheduler |
| Training | `train/train.py`, `train/train_mem.py` | Data loading (WebDataset), argument parsing, training loop |
| Custom trainer | `train/blip3o_trainer.py` | Extends HuggingFace Trainer with per-group LR scheduling |
| Conversation templates | `conversation.py` | 13 templates; `qwen` is the primary one used |
| Model builder | `model/builder.py` | `load_pretrained_model()` — main entry point for loading checkpoints |
| Multimodal utils | `mm_utils.py` | Image tokenization, processing, patching |
| Constants | `constants.py` | Special token IDs (IMAGE_TOKEN_IDX=151667, UND_IMAGE_TOKEN_IDX=151655), default image tokens |

### Data Flow

1. **Understanding**: Image → Vision encoder → Multimodal projector → LLM input embeddings → Autoregressive generation
2. **Generation**: Text prompt → LLM processes tokens → Learnable latent queries attend to LLM hidden states → DiT denoises CLIP features via Flow Matching → Diffusion decoder (VAE) → Output image

### Training Arguments

Critical model-specific arguments (beyond standard HuggingFace TrainingArguments):
- `--gen_vision_tower`: Generation encoder (e.g., `eva-clip-E-14-plus`)
- `--gen_projector_type` / `--mm_projector_type`: Projector architecture
- `--n_query`: Number of latent queries for generation (64 default in scripts, 729 for SigLIP)
- `--gen_pooling`: Feature pooling strategy (`early_pool2d_4`, `pool2d_3`, `seq_3`, etc.)
- `--data_type "mix"`: Blend understanding and generation training data
- `--version qwen`: Conversation template to use

### DeepSpeed Configs

Located in `deepspeed_scripts/`: `zero1.json` (default), `zero2.json`, `zero3.json`, `zero3_offload.json`.

## Branches

- **main**: Base BLIP3o (Qwen2.5-VL + EVA-CLIP)
- **BLIP3o-NEXT**: Main development branch (PR target)
- **Qwen3-Siglip2**: SigLIP2 + Qwen3 variant with flexible training strategies
