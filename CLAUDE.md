# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BAGEL is a multimodal foundation model (7B active / 14B total parameters) using a **Mixture-of-Transformer-Experts (MoT)** architecture for unified visual understanding and generation. It combines a Qwen2.5 LLM backbone, SigLIP vision encoder (ViT), and Flux-based VAE in a single model that handles text-to-image generation, image editing, and vision-language understanding.

## Common Commands

### Environment Setup
```bash
conda create -n bagel python=3.10 -y && conda activate bagel
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

### Training (distributed with torchrun)
```bash
torchrun \
  --nnodes=$num_nodes --node_rank=$node_rank --nproc_per_node=8 \
  --master_addr=$master_addr --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --vae_path $vae_path --vit_path $vit_path --llm_path $llm_path \
  --use_flex True --max_latent_size 64
```

Fine-tuning adds: `--finetune_from_hf True --lr 2e-5 --finetune-from-ema True --resume-model-only True`

For debugging, use smaller token budgets: `--expected_num_tokens 10240 --max_num_tokens 11520 --max_num_tokens_per_sample 10240`

### Inference (Gradio UI)
```bash
python app.py --model_path $model_path
```

### Evaluation
Individual benchmark scripts live in `scripts/eval/`. Each requires setting `$model_path` and `$output_path`. Some benchmarks (MathVista, WISE, KRIS, RISE, ImgEdit, GEdit) need `$openai_api_key` for GPT-based judging.

## Architecture

### Three-encoder unified model
The core `Bagel` class (`modeling/bagel/bagel.py`) orchestrates:
- **LLM** (`modeling/bagel/qwen2_navit.py`): Qwen2.5 with MoT/MoE decoder layer variants (`Qwen2MoTDecoderLayer`, `Qwen2MoEDecoderLayer`, `Qwen2DecoderLayer`)
- **ViT** (`modeling/bagel/siglip_navit.py`): SigLIP vision encoder for understanding tasks — outputs projected to LLM dimension via MLP connector
- **VAE** (`modeling/autoencoder.py`): Flux.1-dev autoencoder for latent-space image generation

The model processes interleaved multimodal token sequences. Understanding tasks use ViT features with CE loss; generation tasks use VAE latents with MSE loss. Attention masks support causal, full, and noise modes for different token types within the same sequence.

### Training pipeline
`train/pretrain_unified_navit.py` is the single entry point. It uses PyTorch FSDP (`HYBRID_SHARD` strategy) via helpers in `train/fsdp_utils.py`. Key features:
- **FLEX packing** (`data/dataset_base.py:PackedDataset`): variable-length sequences packed to maximize GPU utilization with sparse attention masks
- **EMA** (decay 0.9999) maintained alongside the training model
- **Dual loss**: CE for language/understanding tokens + MSE for generation latent tokens
- Activation checkpointing and gradient clipping (max_norm=1.0)

### Data system
Dataset config is YAML-based (`data/configs/example.yaml`). Three dataset types registered in `data/dataset_info.py`:
- `T2IIterableDataset` — Parquet-based text-to-image pairs
- `SftJSONLIterableDataset` — JSONL multi-turn vision-language conversations
- `UnifiedEditIterableDataset` — Parquet-based image editing (source + instruction + target)

All extend `DistributedIterableDataset` for multi-rank/worker sharding. The pipeline: raw data → transforms (`data/transforms.py`) → tokenize/patchify (`data/data_utils.py`) → pack into batches (`data/dataset_base.py`).

To add a new dataset: subclass `DistributedIterableDataset`, register it in `DATASET_REGISTRY` in `data/dataset_info.py`, add its info to `DATASET_INFO`, and reference it in the YAML config.

### Inference
`InterleaveInferencer` (`inferencer.py`) handles KV-cache-based generation with classifier-free guidance (CFG) for both text and image conditioning. `app.py` provides the Gradio web UI with quantization support (NF4/INT8 via bitsandbytes).

## Key Training Notes

- `max_latent_size=64` is required when fine-tuning from the released BAGEL checkpoint (otherwise OOB errors)
- `num_used_data` sum across datasets must exceed `NUM_GPUS × NUM_WORKERS`
- For T2I-only fine-tuning: `--visual_und=False`. For VLM-only: `--visual_gen=False`
- VAE is frozen by default (`freeze_vae=True`); LLM and ViT are trainable
- Logging is rank-0 only; W&B integration via `--wandb_project` / `WANDB_API_KEY`

## No Build System / No Tests

There is no `setup.py`, `pyproject.toml`, or test suite. The project uses direct module imports. Models are loaded via HuggingFace Hub's `snapshot_download()` or local paths.
