# Lumina-DiMOO

Unified multimodal vision-language model based on the LLaDA architecture. Supports text-to-image generation, multimodal understanding (MMU), OCR, spatial reasoning, and pointing/counting.

**Base model:** [`Alpha-VLLM/Lumina-DiMOO`](https://huggingface.co/Alpha-VLLM/Lumina-DiMOO)

## Project Structure

```
Lumina-DiMOO/
├── model/                  # Model definitions (LLaDA transformer + multimodal wrapper)
├── train/                  # Training scripts
│   ├── train_unified.py    #   Main unified multi-task training script
│   ├── train_unified_ocr.sh#   OCR training shell wrapper (unified)
│   ├── train_generation.py #   Generation-only training
│   ├── train_ocr.py        #   OCR-only training
│   └── train_pointing.py   #   Pointing-only training
├── inference/              # Inference scripts (T2I, I2I, MMU)
├── evaluation_scripts/     # Evaluation benchmarks (see below)
├── generators/             # Generation algorithms (MaskGit, text understanding)
├── utils/                  # Image encoding/decoding (VQ-VAE), prompt templates
├── xllmx/                  # Distributed training framework (FSDP, checkpointing, LR)
├── data/                   # Data item processors
├── configs/                # YAML data configs
├── pre_tokenizer/          # Offline pre-tokenization scripts
└── VLMEvalKit/             # Git submodule for VLM evaluation benchmarks
```

## Setup

```bash
conda env create -f lumina.yaml   # Python 3.10, PyTorch 2.3.1
conda activate lumina_dimoo
export PYTHONNOUSERSITE=1
```

## Training

All training uses `torch.distributed.run`. The unified script `train/train_unified.py` supports multiple tasks via `--task`.

```bash
python -m torch.distributed.run --nproc_per_node=<N_GPUS> --master_port=29504 \
    train/train_unified.py \
    --task ocr \
    --batch_size 1 --accum_iter 32 --epochs 999 \
    --lr 1e-5 --precision bf16 \
    --data_parallel fsdp --use_lora --lora_rank 128 \
    --init_from Alpha-VLLM/Lumina-DiMOO \
    --data_config configs/data.yaml \
    --use_wandb --wandb_project "lumina-dimoo" \
    --output_dir output/<exp_name>
```

### Training Modes

| Mode | `--data_parallel` | `--use_lora` | Description |
|------|-------------------|--------------|-------------|
| LoRA only | `none` | Yes | Single-node LoRA fine-tuning |
| FSDP only | `fsdp` | No | Full-parameter distributed training |
| FSDP + LoRA | `fsdp` | Yes | Memory-efficient distributed LoRA (`lumina_fsdp_lora` branch) |

### Resume

`--wandb_run_id <run_id>` 와 `--resume_path <checkpoint_dir>` 인자로 resume 가능.

## Evaluation (`evaluation_scripts/`)

평가 스크립트는 task별로 정리되어 있음.

```
evaluation_scripts/
├── 2d_spatial/             # Spatial relation (BLINK, CVBench, GQA)
├── counting/               # Counting (Pixmo, CVBench) + GenEval pipeline
└── ocr/                    # (planned)
```

### Understanding Evaluation

```bash
# Spatial - BLINK (single GPU)
python evaluation_scripts/2d_spatial/evaluate_blink_spatial.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --lora_ckpt_path <lora_dir> --output_dir <output> \
    --steps 20 --gen_length 20 --block_length 20

# Counting - Pixmo (multi-GPU)
torchrun --nproc_per_node=<N> evaluation_scripts/counting/evaluate_pixmo_multigpu.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --lora_ckpt_path <lora_dir> --output_dir <output> \
    --steps 20 --gen_length 20 --block_length 20
```

### Generation Evaluation (GenEval 3-Stage Pipeline)

1. `01_geneval_inference_multigpu.sh` — 텍스트 프롬프트로 이미지 생성
2. `02_geneval_eval_specific_model.sh` — 외부 도구로 생성된 이미지의 카운팅 평가
3. `03_summarize_jsonl_results_and_make_confusion_mat.py` — 결과 집계, 혼동행렬 생성

### Benchmark Summary

| Benchmark | Task | Dataset | Metric | Multi-GPU |
|-----------|------|---------|--------|-----------|
| BLINK Spatial | Spatial relation | `BLINK-Benchmark/BLINK` | Accuracy | - |
| CVBench Spatial | Spatial relation | `nyu-visionx/CV-Bench` | Accuracy | - |
| CVBench Counting | Object counting | `nyu-visionx/CV-Bench` | Accuracy | O |
| Pixmo Counting | Object counting | `Jiwon-Kang/pixmo-count-filtered-imgContained` | MAD | O |
| GenEval | Generation counting | Text prompts | Counting accuracy | O |

## Inference

```bash
# Text-to-image
python inference/inference_t2i.py \
    --checkpoint <path> --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --height 512 --width 512 --timesteps 64 --cfg_scale 4.0 --seed 65513 \
    --output_dir output/

# Multimodal understanding
python inference/inference_mmu.py ...
```

## Architecture

```
LLaDAForMultiModalGeneration (model/modeling_xllmx_dimoo.py)
  └── LLaDAModelLM (model/modeling_llada.py)
        └── LLaDAConfig (model/configuration_llada.py)
```

- **LLaDAForMultiModalGeneration**: Variable-length batch padding, attention masks, loss. LoRA wrapping/unwrapping 지원.
- **LLaDAModelLM**: Core transformer. Activation checkpointing + FSDP block-level wrapping.
- **VQ-VAE**: 이미지 토크나이제이션 (discrete tokens). 행 사이에 newline token 삽입 (`utils/image_utils.py`).
- **MaskGit**: Parallel decoding for image generation. Cosine noise schedule + CFG (`generators/image_generation_generator.py`).
