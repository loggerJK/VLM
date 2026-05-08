# CONTEXT.md — MMaDA Audio Ablation (Counting 이식)

## Goal

Lumina_Dimoo (`/data/mm-llm-backbone_890/personal/sirius/audio_ablation/audio_lumina`) 의 **Counting** 학습 로직을 MMaDA에 이식한다.
Counting = 이미지 안의 객체 수를 세는 VQA 태스크 (Understanding only).

- 참고 코드베이스: `audio_lumina` (원본), `audio_bagel` (BAGEL 구현체)
- 베이스 모델: `Gen-Verse/MMaDA-8B-MixCoT`
- 베이스 훈련 스크립트: `train_mmada_stage2.py`
- 훈련 데이터: `heez/pixmo-point-count-gen-und` (HuggingFace Hub, `train` split)
- 평가 데이터: `Jiwon-Kang/pixmo-count-filtered-imgContained` (HuggingFace Hub, `validation` split, 501샘플)
- 메트릭: Accuracy (exact match), MAD (Mean Absolute Deviation), Confusion Matrix (0–20)

---

## 작업 내역

### 1. `parquet/my_dataset.py` — `CountingDataset` 추가

- 기존 `VQADataset`과 동일한 인터페이스
- `datasets.load_dataset("heez/pixmo-point-count-gen-und", split="train")` 로드
- `count_lower_limit` / `count_upper_limit` 로 범위 필터링 (기본 0–20)
- rank/world_size 샤딩 지원
- 이미지: `image_transform_squash(resolution)` 적용
- 텍스트: `tokenizer.apply_chat_template` 으로 Q&A 포맷 생성 (`**N**` 답변 포함)
- `collate_fn`에서 `answer` 키를 리스트로 유지 (학습 중 검증용)

### 2. `parquet/__init__.py` — `CountingDataset` export 추가

### 3. `training/train_mmada_counting.py` — Counting 전용 훈련 스크립트

- `train_mmada_stage2.py` 구조 기반
- MMU 데이터로더를 `CountingDataset`으로 교체
- T2I / LM 데이터는 Catastrophic Forgetting 방지 regularization 용도로 유지 (낮은 계수)
- `extract_number_fixed()` 유틸 내장 (Lumina/BAGEL 동일 로직)
- **`validate_counting()`**: `eval_every` 스텝마다 실행
  - `heez/pixmo-point-count-gen-und` `val_und` split (최대 100샘플) 사용
  - 생성 파라미터: `max_new_tokens=20, steps=20, block_length=20, temperature=0.0`
  - WandB 로깅: `counting/val_accuracy`, `counting/val_mad`, `counting/predictions`

### 4. `configs/mmada_counting_llada_instruct.yaml` — 훈련 설정

| 항목 | 값 |
|------|-----|
| `pretrained_model_path` | `Gen-Verse/MMaDA-8B-MixCoT` |
| `resolution` | 512 |
| `num_vq_tokens` | 1024 |
| `batch_size_mmu` | 4 (counting) |
| `mmu_coeff` | 1.0 |
| `t2i_coeff` | 0.1 (regularization) |
| `lm_coeff` | 0.05 (regularization) |
| `eval_every` | 500 steps |
| `max_train_steps` | 50000 |
| `learning_rate` | 2e-5 |
| `count_lower_limit / upper_limit` | 0 / 20 |

### 5. `evaluation/counting/evaluate_pixmo.py` — 단일 GPU 평가

- `Jiwon-Kang/pixmo-count-filtered-imgContained` validation split (501샘플) 평가
- 동일한 추론 파이프라인: VQ encode → `mmu_generate(max_new_tokens=20, steps=20, block_length=20)` → `extract_number_fixed()`
- 출력: `results.json`, `summary.json`, `confusion_matrix_0-20.png`, `confusion_matrix_full.png`

### 6. `evaluation/counting/evaluate_pixmo_multigpu.py` — 다중 GPU 평가

- `torchrun` 기반 분산 추론 (`i % world_size == rank` 샤딩)
- 중간 결과: `results_rank{n}.jsonl` (resume 지원)
- Rank 0에서 집계 후 최종 메트릭 계산 및 Confusion Matrix 저장

---

## 실행 방법

```bash
# 훈련 (단일 GPU 테스트)
accelerate launch --config_file accelerate_configs/1_gpu.yaml \
  training/train_mmada_counting.py \
  config=configs/mmada_counting_llada_instruct.yaml \
  training.max_train_steps=100 experiment.eval_every=50

# 훈련 (8 GPU)
accelerate launch --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
  --main_process_port=8888 \
  training/train_mmada_counting.py \
  config=configs/mmada_counting_llada_instruct.yaml

# 평가 (단일 GPU)
python evaluation/counting/evaluate_pixmo.py \
  --model_path Gen-Verse/MMaDA-8B-MixCoT \
  --vq_model_path showlab/magvitv2 \
  --output_dir evaluation_results/counting

# 평가 (다중 GPU)
torchrun --nproc-per-node=8 --master-port=54321 \
  evaluation/counting/evaluate_pixmo_multigpu.py \
  --model_path <checkpoint_path> \
  --vq_model_path showlab/magvitv2 \
  --output_dir evaluation_results/counting
```

---

## 파일 목록

| 파일 | 상태 |
|------|------|
| `parquet/my_dataset.py` | 수정 (`CountingDataset` 추가) |
| `parquet/__init__.py` | 수정 (export 추가) |
| `training/train_mmada_counting.py` | 신규 |
| `configs/mmada_counting_llada_instruct.yaml` | 신규 |
| `evaluation/counting/evaluate_pixmo.py` | 신규 |
| `evaluation/counting/evaluate_pixmo_multigpu.py` | 신규 |
| `CLAUDE.md` | 신규 (codebase 가이드) |

---

## Fix Log (Codex 검토 후 수정)

Codex 검토에서 4개 결함 발견 → Lumina_Dimoo 원본 동작 + `inference_mmu.py` 표준 패턴에 맞춰 정렬:

### Fix 1: `CountingDataset` answer 포맷 wrapping 제거

`parquet/my_dataset.py` — `answer_count`가 `"There are **7** ... in the image."` 같은 문장을 포함할 때 전체를 `**...**`로 다시 감싸서 markdown이 중첩되던 버그.

- 변경: `answer_count` 원본 그대로 사용 (Lumina와 동일)
- 정수 `count`만 폴백으로 있을 때만 `**N**` 포맷 적용

### Fix 2: Validation/Evaluation prompt를 `apply_chat_template` 기반으로 통일

`training/train_mmada_counting.py`, `evaluation/counting/evaluate_pixmo.py`, `evaluation/counting/evaluate_pixmo_multigpu.py`

- 이전: 수동 `<|start_header_id|>...` 문자열 + `<|sot|>` 추가 삽입 → BOS 2회 (학습 분포와 불일치)
- 수정: `inference_mmu.py:95-108` 패턴 사용 — `apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")` 후 `<|eoi|>` 직후에 바로 concat (별도 `<|sot|>` 없음)
- 결과: `<|mmu|> <|soi|> image <|eoi|> <bos> <header>user ...` (BOS 1회) — 학습과 일치

### Fix 3: MAD 계산에 parse 실패(-1) 포함

`training/train_mmada_counting.py`, `evaluation/counting/evaluate_pixmo.py`, `evaluation/counting/evaluate_pixmo_multigpu.py`

- 이전: `[abs(g - p) for ... if p >= 0]` → parse 실패 시 metric이 과소평가됨
- 수정 (Lumina `train_unified.py:1668`과 동일): `[abs(g - p) for ...]` — `-1` 예측도 deviation에 포함되어 잘못된 출력에 패널티 부여

### Fix 4: `CountingDataset`에 worker sharding 추가

`parquet/my_dataset.py` — `IterableDataset`에서 rank 샤딩만 하고 `torch.utils.data.get_worker_info()`를 사용하지 않아, 같은 rank의 16개 worker가 동일 indices를 반복하던 문제.

- 수정: `__iter__` 시작에 `worker_info.id::worker_info.num_workers` 슬라이싱으로 worker 단위 샤딩 추가
- 결과: rank-N의 worker-W는 `self.indices[W::num_workers]`만 처리 → 데이터 중복 제거
