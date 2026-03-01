# train_counting.py - Image Generation 학습 기능 통합 변경사항

## 개요

기존 `train_counting.py`는 image understanding(counting, pointing)만 학습 가능했음.
이번 변경으로 `--task generation` 및 `--task both` 모드를 추가하여, **understanding 단독 / generation 단독 / 동시 학습**을 모두 지원.

---

## 변경 파일 목록

| 파일 | 상태 |
|------|------|
| `train_counting.py` | Modified |
| `test_scripts/test_vq_encode_decode.py` | **New** |

---

## 신규 파일

### `test_scripts/test_vq_encode_decode.py`

Janus VQ-16 모델의 encode → decode 왕복 테스트 스크립트.

**검증 항목:**
- `indices.shape == [576]` (24×24)
- `indices` 값 범위: `[0, 16383]` (codebook_size=16384)
- 재구성 이미지가 원본과 시각적으로 유사한지 (PSNR 측정)
- fp32 / bf16 모두 NaN 없이 동작하는지
- `prepare_gen_img_embeds()` 정상 동작 여부

**실행:**
```bash
python test_scripts/test_vq_encode_decode.py \
    --model_path deepseek-ai/Janus-Pro-7B \
    --image_path <test_image.png> \
    --output_dir output/debug_vq_janus
```

---

## `train_counting.py` 상세 변경사항

### 1) 새로운 import 추가

```python
import torch.nn.functional as F                          # line 16
from transformers.modeling_outputs import CausalLMOutputWithPast  # line 20
```

### 2) `encode_image_to_vq_tokens()` 함수 (line 33-46)

```python
def encode_image_to_vq_tokens(vq_model, image, img_size=384):
    """PIL Image -> 576 VQ token IDs (LongTensor)"""
```

- PIL Image를 384×384로 resize → `[-1, 1]` 정규화
- VQ encoder로 인코딩하여 576개 discrete token IDs 반환
- `collate_fn_generation`에서 호출됨

### 3) `EnhancedMultiModalModel.forward()` 수정 (line 55-95)

**변경 전:** understanding만 처리 (pixel_values + image_token_masks)

**변경 후:** `gen_token_mask` 파라미터 추가, generation/understanding 자동 분기

```python
def forward(self, input_ids, attention_mask, pixel_values=None,
            image_token_masks=None, labels=None, gen_token_mask=None, **kwargs):

    if gen_token_mask is not None and gen_token_mask.any():
        return self._forward_generation(...)   # Generation 경로
    elif pixel_values is not None and image_token_masks is not None:
        ...                                     # Understanding 경로 (기존)
```

### 4) `_forward_generation()` 메서드 (line 97-129)

Generation forward path의 핵심 로직:

```
1) input_ids → language_model.get_input_embeddings() → base text embeddings
2) gen_token_mask 위치 → prepare_gen_img_embeds(GT VQ token IDs) → image embeddings 교체
   (Teacher forcing: 학습 시 GT 토큰을 입력으로 사용)
3) language_model.model(inputs_embeds) → hidden_states [batch, seq, hidden_dim]
4) gen_head(hidden_states) → gen_logits [batch, seq, 16384]
5) Shifted cross-entropy loss (next-token prediction, ignore_index=-100)
```

**반환:** `CausalLMOutputWithPast(loss=loss, logits=gen_logits)`

### 5) `collate_fn_generation()` 함수 (line 301-367)

Text-to-image 학습용 배치 구성:

```
시퀀스: [text_prompt_tokens] [<|Assistant|>] [boi] [vq_tok_0 ... vq_tok_575] [eoi]
Labels: [-100 .......................... -100 .. -100] [vq_tok_0 ... vq_tok_575] [eoi]
GenMask:[False ......................... False . False] [True ...... True ......] [False]
```

- 각 샘플의 image를 VQ encode하여 576 토큰으로 변환
- text는 `apply_sft_template_for_multi_turn_prompts`로 포맷
- Left-padding 적용 (understanding 관례와 일치)
- 반환: `{input_ids, labels, attention_mask, gen_token_mask}`

### 6) `collate_fn_both()` 함수 (line 369-394)

Multi-task collate: 배치 내 샘플을 자동 분류

| 필드 | 판별 |
|------|------|
| `question_count` / `question` 존재 | → Understanding (`collate_fn`) |
| `text` / `caption` / `prompt` 존재 | → Generation (`collate_fn_generation`) |

### 7) `ValidationCallback` 변경

#### `_save_gen_components()` (line 415-425)

Generation 학습 시 `gen_head`, `gen_embed`, `gen_aligner`의 state dict를 `gen_components.pt`로 별도 저장.

#### `on_step_begin()` 수정 (line 427-450)

- LoRA 체크포인트 저장 시 `gen_components.pt`도 함께 저장 (generation/both 태스크)
- Validation 분기:
  - `counting/pointing/both` → `self.validate()` (기존 counting 검증)
  - `generation/both` → `self.validate_generation()` (새 generation 검증)

#### `on_epoch_end()` 수정 (line 452-468)

- Epoch 종료 시에도 `gen_components.pt` 저장 로직 추가

#### `validate_generation()` 메서드 (line 582-673)

텍스트 프롬프트로 이미지를 autoregressive 생성하여 wandb에 로깅:

```
1) 프롬프트 토큰화 → [text_ids] + [boi]
2) 576 step autoregressive 생성:
   - 매 step: text_embed + gen_embed(generated tokens) → LLM → gen_head → argmax
3) 576 VQ tokens → vq_model.decode_code() → PIL Image
4) wandb.Image로 로깅
```

고정 프롬프트 4개 사용 (cat, car, sunset, dog).

### 8) argparse 추가 (line 691-696)

```python
--task {counting,pointing,generation,both}  # 기존 counting/pointing + 신규 generation/both
--gen_data_path <path>                       # Generation 데이터 경로 (HF dataset 또는 로컬)
--gen_img_size 384                           # VQ 인코딩용 이미지 크기
```

### 9) 데이터셋 로딩 분기 (line 706-730)

| Task | 데이터셋 |
|------|----------|
| `counting` / `pointing` | `--data_path`에서 understanding 데이터 로드 |
| `generation` | `--gen_data_path`에서 generation 데이터 로드 |
| `both` | 양쪽 모두 로드 → `ConcatDataset`으로 결합 |

### 10) Generation 파라미터 활성화 (line 807-823)

`--task generation` 또는 `--task both`일 때:

| Component | requires_grad |
|-----------|:------------:|
| `gen_head` | **True** |
| `gen_embed` | **True** |
| `gen_aligner` | **True** |
| `gen_vision_model` (VQ-VAE) | False (frozen) |

LoRA와 독립적으로 동작 — LoRA는 language_model에, gen 컴포넌트는 full parameter 학습.

### 11) DDP 설정 (line 843)

```python
ddp_find_unused_parameters=True  # generation/both 태스크일 때
```

Generation과 understanding이 서로 다른 모델 컴포넌트를 사용하므로, 미사용 파라미터 허용 필요.

### 12) Collator 분기 (line 853-864)

```python
if args.task in ["counting", "pointing"]:
    collator = collate_fn(...)            # Understanding
elif args.task == "generation":
    collator = collate_fn_generation(...) # Generation (VQ model 참조 필요)
elif args.task == "both":
    collator = collate_fn_both(...)       # Multi-task (자동 분기)
```

### 13) 최종 모델 저장 (line 898-907)

학습 완료 후 `final_model/gen_components.pt`로 gen 컴포넌트 별도 저장.

---

## 학습 가능한 파라미터 요약

| Component | counting/pointing | generation | both |
|-----------|:-----------------:|:----------:|:----:|
| vision_model (SigLIP) | Frozen | Frozen | Frozen |
| aligner | Frozen | Frozen | Frozen |
| language_model (Llama) | LoRA | LoRA | LoRA |
| gen_head | Frozen | **Trainable** | **Trainable** |
| gen_embed | Frozen | **Trainable** | **Trainable** |
| gen_aligner | Frozen | **Trainable** | **Trainable** |
| gen_vision_model (VQ) | Frozen | Frozen | Frozen |

---

## 실행 예시

```bash
# Generation 단독
python -m torch.distributed.run --nproc_per_node=4 train_counting.py \
    --task generation \
    --gen_data_path "your-hf-dataset/text-image-pairs" \
    --tuning_mode transformer_lora \
    --lora_r 16 --lora_alpha 32 \
    --batch_size 2 --lr 1e-4 --epochs 3

# Multi-task (Understanding + Generation)
python -m torch.distributed.run --nproc_per_node=4 train_counting.py \
    --task both \
    --data_path ./data/pixmo_counting \
    --gen_data_path "your-hf-dataset/text-image-pairs" \
    --tuning_mode transformer_lora \
    --batch_size 2 --lr 1e-4 --epochs 3

# 기존 counting (변경 없음)
python -m torch.distributed.run --nproc_per_node=4 train_counting.py \
    --task counting \
    --data_path ./data/pixmo_counting_filtered \
    --tuning_mode transformer_lora \
    --batch_size 2 --lr 1e-4 --epochs 3
```

---

## 체크포인트 구조

```
checkpoints/
├── epoch0_step-500/
│   ├── adapter_config.json      # LoRA config
│   ├── adapter_model.safetensors # LoRA weights
│   ├── gen_components.pt         # gen_head + gen_embed + gen_aligner (generation/both만)
│   └── ...
├── epoch1/
│   └── ...
└── final_model/
    ├── adapter_model.safetensors
    ├── gen_components.pt
    └── ...
```

---

## Generation 데이터셋 요구사항

Generation 데이터셋은 각 샘플에 다음 필드 중 하나 이상을 포함해야 함:

| 필드 | 타입 | 설명 |
|------|------|------|
| `image` | PIL Image | 학습 이미지 (또는 `image_url`로 다운로드) |
| `text` / `caption` / `prompt` | str | 이미지 설명 텍스트 |

Understanding 데이터셋과 구별: `question_count`/`question` 필드가 **없어야** generation으로 분류됨.
