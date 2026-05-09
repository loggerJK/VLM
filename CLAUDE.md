# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Work Log Convention

저장소 루트의 `CONTEXT.md`에 작업 내역을 누적 기록합니다. **모든 작업이 끝날 때마다** `CONTEXT.md`의 맨 위에 새 항목(날짜 + 목표 + 변경 파일 + 실행 방법 + 검증 메모)을 추가하세요. 기존 항목은 절대 삭제하지 말고 위에 쌓는 방식으로 유지합니다.

## Project Overview

Qwen3-VL is Alibaba's vision-language model repository containing fine-tuning frameworks, utility packages, evaluation benchmarks, and demo notebooks for Qwen2-VL, Qwen2.5-VL, and Qwen3-VL model families. Models range from 2B to 235B parameters.

## Repository Structure

- **qwen-vl-finetune/**: Training framework using DeepSpeed + torchrun
- **qwen-vl-utils/**: Installable Python package for vision processing (image/video loading, resizing, frame extraction)
- **evaluation/**: Benchmark scripts (MMMU, VideoMME, MathVision, RealWorldQA, ODinW-13)
- **cookbooks/**: Jupyter notebook examples for various tasks (OCR, grounding, video understanding, etc.)
- **web_demo_mm.py**: Gradio-based web demo with vLLM or HuggingFace backend
- **docker/**: Dockerfile for vLLM-based deployment

## Common Commands

### Installation
```bash
pip install qwen-vl-utils                    # Utility package
pip install -r requirements_web_demo.txt     # Web demo dependencies
# Training deps: torch==2.6.0 torchvision==0.21.0 transformers==4.57.0.dev0
#   deepspeed==0.17.1 flash_attn==2.7.4.post1 accelerate==1.7.0 peft==0.17.1
```

### Training (SFT)
```bash
# Launch distributed training (edit script to set model path, dataset, GPU count)
bash qwen-vl-finetune/scripts/sft_7b.sh      # 7B model
bash qwen-vl-finetune/scripts/sft_32b.sh     # 32B model
bash qwen-vl-finetune/scripts/sft_30a3b.sh   # 30B MoE (Qwen3-VL)
bash qwen-vl-finetune/scripts/sft_30a3b_lora.sh  # LoRA variant

# Training entry point (called by scripts above via torchrun):
# qwen-vl-finetune/qwenvl/train/train_qwen.py
```

### Evaluation
```bash
# MMMU benchmark
bash evaluation/mmmu/infer_instruct.sh   # Run inference
bash evaluation/mmmu/eval_instruct.sh    # Score results with judge model

# VideoMME benchmark
bash evaluation/VideoMME/infer_instruct.sh
bash evaluation/VideoMME/eval_instruct.sh
```

### Web Demo
```bash
python web_demo_mm.py --checkpoint-path Qwen/Qwen3-VL-235B-A22B-Instruct --backend vllm --server-port 7860
```

### Build qwen-vl-utils package
```bash
cd qwen-vl-utils && pip install -e .
```

## Architecture

### Training Pipeline (qwen-vl-finetune/)
- **Entry point**: `qwenvl/train/train_qwen.py` — sets up model, data, and launches HuggingFace Trainer
- **Custom Trainer**: `qwenvl/train/trainer.py` — extends HF Trainer with flash attention loss computation and safe saving for LoRA/DeepSpeed
- **Arguments**: `qwenvl/train/argument.py` — dataclasses for model, data, and training args; controls which components to tune (`tune_mm_vision`, `tune_mm_mlp`, `tune_mm_llm`)
- **Data processing**: `qwenvl/data/data_processor.py` — handles JSON→tokenized training examples; processes images/videos with pixel budgets; supports sequence packing
- **Dataset registry**: `qwenvl/data/__init__.py` — datasets registered as dicts with `annotation_path` and `data_path`; sampling rates via "dataset%50" syntax
- **DeepSpeed configs**: `scripts/zero2.json`, `zero3.json`, `zero3_offload.json`

### Vision Utilities (qwen-vl-utils/)
- **Core module**: `src/qwen_vl_utils/vision_process.py`
- **Key functions**: `process_vision_info()` (main entry), `fetch_image()`, `fetch_video()`, `smart_resize()`, `smart_nframes()`
- Image sources: file paths, URLs, base64 strings, PIL objects
- Video processing: FPS-based frame extraction with configurable min/max frames

### Evaluation Pattern
All benchmarks follow: run inference with vLLM → save JSONL results → score with GPT judge model via API (DashScope or OpenAI-compatible).

## Training Data Format

```json
{"image": "path/to/image.jpg", "conversations": [{"from": "human", "value": "<image>\nQuestion"}, {"from": "gpt", "value": "Answer"}]}
{"video": "path/to/video.mp4", "conversations": [{"from": "human", "value": "<video>\nQuestion"}, {"from": "gpt", "value": "Answer"}]}
```

Multi-image: `"image": ["img1.jpg", "img2.jpg"]` with multiple `<image>` tokens in the human message.

## Key Training Parameters

- **Component tuning**: `--tune_mm_vision` (ViT), `--tune_mm_mlp` (projector), `--tune_mm_llm` (language model) — each can be toggled independently
- **LoRA**: `--lora_enable True --lora_r 128 --lora_alpha 256`
- **Pixel budgets**: `--min_pixels` / `--max_pixels` control image resolution (affects memory)
- **Video frames**: `--max_nframes` controls max frames extracted from video
- **Sequence packing**: `--pack_data True` for efficient multi-example batching (use `qwenvl/tools/pack_data.py` to preprocess)
