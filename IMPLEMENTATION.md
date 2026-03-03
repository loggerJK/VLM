# BLIP3o Counting + OCR Synthetic Training — Implementation Summary

## Overview

BLIP3o의 기존 T2I 학습 파이프라인에 **counting/OCR synthetic generation** 데이터셋 지원을 추가했다.
Lumina-DiMOO, BAGEL, Janus와 Fair Comparison을 위해 동일한 LoRA/8bit AdamW/validation/NLTK 메트릭 컨벤션을 맞췄다.

---

## Quick Start

### 환경 설정

```bash
conda activate blip3o
source ./.env    # WANDB_API_KEY 로드
```

### Counting 학습

```bash
# Generation only (기본)
bash train_counting.sh gen

# Generation + Understanding
bash train_counting.sh both

# 특정 체크포인트에서 Resume
bash train_counting.sh gen output/counting_gen/epoch0_step-500
```

### OCR Synthetic 학습

```bash
# Generation only (기본)
bash train_ocr_synthetic.sh gen

# Generation + Understanding
bash train_ocr_synthetic.sh both

# 특정 체크포인트에서 Resume
bash train_ocr_synthetic.sh gen output/ocr_synthetic_gen/epoch0_step-500
```

### Auto-Resume

두 스크립트 모두 `--auto_resume`가 기본 활성화. output_dir에서 `epoch{N}_step-{S}` 패턴의 최신 체크포인트를 자동으로 찾아 학습을 재개한다. 별도 인자 없이 같은 명령어를 다시 실행하면 됨.

---

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `blip3o/ocr_metrics.py` | **New** | OCR 평가 메트릭 (WER, CER, METEOR, BLEU, Edit Distance, P/R/F1) |
| `blip3o/ocr_render.py` | **New** | Markdown → 이미지 렌더링 (wkhtmltoimage) |
| `blip3o/train/train.py` | **Modified** | 핵심 학습 코드: 데이터셋, LoRA, validation callback, checkpoint/resume |
| `sft.sh` | **Modified** | 새 argument 문서화 |
| `train_counting.sh` | **New** | Counting 학습 shell script |
| `train_ocr_synthetic.sh` | **New** | OCR synthetic 학습 shell script |
| `.env` | **New** | WANDB_API_KEY (Lumina-DiMOO에서 복사) |

---

## Architecture: 주요 변경사항

### 1. New Arguments

**ModelArguments**:
```
--use_lora                  # LoRA 활성화 (default: False)
--lora_r 128                # LoRA rank
--lora_alpha 32             # LoRA alpha
--lora_dropout 0.05         # LoRA dropout
--lora_target_modules "q_proj,k_proj,v_proj,o_proj"  # Qwen attention projections
--resume_from_checkpoint    # 명시적 체크포인트 경로
--auto_resume               # output_dir에서 최신 체크포인트 자동 탐색
```

**DataArguments**:
```
--task counting|ocr_synthetic      # 태스크 유형
--mode gen|und|both                # 학습 모드
--data_type counting|ocr_synthetic|mix  # 데이터 라우팅 (task와 자동 동기화)
--hf_dataset_path                  # HuggingFace 데이터셋 경로
--ocr_num_samples 200000           # OCR 샘플 수
--ocr_image_width 512              # OCR 이미지 크기
```

**TrainingArguments**:
```
--validation_interval 250   # understanding validation 주기 (steps)
--log_freq 250              # generation validation 주기 (steps)
--validation_samples 100    # validation 샘플 수
--optim adamw_bnb_8bit      # 8-bit AdamW (Janus convention)
```

### 2. New Datasets

| Dataset | Class | Source | Usage |
|---------|-------|--------|-------|
| Counting | `CountingGenDataset` | `heez/pixmo-point-count-gen-und` (train split, descriptions 있는 행) | Generation training |
| OCR Synthetic | `OCRSyntheticGenDataset` | `agentlans/high-quality-english-sentences` + `ocr_render` | Generation training |

두 데이터셋 모두 기존 T2I 파이프라인과 동일한 방식으로 처리:
- Conversation: `human: "Please generate image based on: {caption}"` → `gpt: "<image>"`
- Image: EVA-CLIP preprocessor → gen_image tensor

### 3. LoRA

- `peft` 라이브러리의 `LoraConfig` + `get_peft_model()` 사용
- **Qwen2.5-VL backbone의 attention projections에만 적용** (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- DiT, down_projector, latent_queries는 기존대로 직접 학습 (LoRA 미적용)

### 4. ValidationCallback

`transformers.TrainerCallback`을 상속한 `ValidationCallback`이 N steps마다 자동 실행:

**Understanding Validation** (`validation_interval` steps마다):
- Counting: Qwen2.5-VL `model.generate()` → accuracy + MAD
- OCR: Qwen2.5-VL `model.generate()` → WER/CER/METEOR/BLEU/Edit Distance/P/R/F1

**Generation Validation** (`log_freq` steps마다):
- 5개 프롬프트로 이미지 생성 → WandB에 `wandb.Image` 로깅

**Checkpoint Saving** (`save_steps` steps마다):
```
epoch{N}_step-{S}/
├── adapter_config.json           # LoRA config
├── adapter_model.safetensors     # LoRA adapter weights
├── gen_components.pt             # DiT + down_projector + latent_queries
├── tokenizer files
```

### 5. Resume Logic

1. `--resume_from_checkpoint PATH` (명시적) 또는 `--auto_resume` (자동)
2. LoRA adapter 로드: `PeftModel.from_pretrained()`
3. Non-LoRA components 복원: `gen_components.pt` → DiT + down_projector + latent_queries
4. Step/epoch 파싱: WandB 연동

---

## Conventions (Fair Comparison Table)

| Item | Lumina | BAGEL | Janus | **BLIP3o** |
|------|--------|-------|-------|------------|
| LoRA rank | 128 | 128 | 128 | **128** |
| LoRA alpha | 32 | 256 | 32 | **32** |
| LoRA targets | q,k,v,attn_out... | q,k,v,o | q,k,v,o,up,down,gate | **q,k,v,o_proj** |
| Optimizer | AdamW8bit | AdamW8bit | adamw_bnb_8bit | **adamw_bnb_8bit** |
| LR | 1e-5 | 1e-4 | 4e-5 | **4e-5** |
| Batch (effective) | 128 | packed | 128 | **128** (1x16x8) |
| Val interval | 500 | 500 | 250 | **250** |
| Save interval | 500 | 2000 | 500 | **500** |
| Ckpt naming | epoch{N}-iter-step | epoch{N}-step{S} | epoch{N}_step-{S} | **epoch{N}_step-{S}** |
| OCR Metrics | WER,CER,METEOR,BLEU,ED,P/R/F1 | same | same | **same** |

---

## WandB Logged Metrics

### Counting Task
```
val/accuracy        # Counting 정확도
val/mad             # Mean Absolute Difference
val/generated_images  # 생성 이미지 (wandb.Image)
```

### OCR Task
```
val/ocr_wer          # Word Error Rate
val/ocr_cer          # Character Error Rate
val/ocr_meteor       # METEOR score
val/ocr_bleu         # BLEU-2 score
val/ocr_edit_distance  # Edit Distance
val/ocr_precision    # Word-level Precision
val/ocr_recall       # Word-level Recall
val/ocr_f1           # Word-level F1
val/generated_images   # 생성 이미지
```

---

## Troubleshooting

### `wkhtmltoimage not found` (OCR task)
```bash
sudo apt-get install wkhtmltopdf
```

### NLTK data missing
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

### Resume 후 loss가 크게 점프하는 경우
- Optimizer state는 resume 시 복원되지 않음 (model weights만 복원)
- Learning rate warmup이 다시 시작되므로 초기 몇 step은 loss 변동이 있을 수 있음
- 정상적인 경우 수십 step 내에 이전 수준으로 수렴
