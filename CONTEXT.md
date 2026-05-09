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

### Fix 6: `apply_chat_template` 사용 dataset 의 trailing assistant header strip 안정화 (CRITICAL)

`parquet/my_dataset.py` 의 `VQADataset` (line 339, 385-386), `CountingDataset` (line 470, 521-522), `RelPositionDataset` (line 610, 637-638) 세 곳 모두 `apply_chat_template(..., add_generation_prompt=True)` 결과의 trailing 빈 assistant header 를 `_asst_suffix = '...|>\n'` (개행 1개) 로 endswith strip 시도. 그러나:

1. LLaDA 토크나이저 (`GSAI-ML/LLaDA-8B-Instruct`) 의 chat_template jinja 는 `add_generation_prompt` 인자를 무시하고 무조건 `<|start_header_id|>assistant<|end_header_id|>\n\n` 을 끝에 매닮 (개행 **2개**). `if add_generation_prompt` 분기 자체가 없음.
2. 따라서 `endswith('...\n')` 가 False → strip 무력화 → trailing 헤더가 그대로 남음.
3. `UniversalPrompting.mmu_prompt()` 가 잡는 마지막 `<|end_header_id|>` 가 trailing 헤더 것으로 밀림 → 정답 문장이 prompt 영역으로 흡수, target 은 `\n + EOS padding` 만 남음.
4. Loss 는 EOS 예측만 학습해 정상적으로 감소하지만 task 학습은 0 에 가까움 — wandb metric 만 보면 발견 불가능한 silent failure.

**왜 원래 MMaDA 본 학습 경로 (`training/data.py`) 는 이 문제가 없는가**: 그쪽은 문자열을 manual 로 합성해 trailing 헤더 자체를 만들지 않음 → `mmu_prompt` 의 "마지막 `<|end_header_id|>` = 정답 직전 헤더" 전제가 자연스럽게 성립.

수정: 세 dataset 모두 endswith + 고정 suffix 매칭을 버리고 rfind 기반 robust strip 으로 교체. 헬퍼 attr 명을 의미별로 분리 (`_trailing_asst_header`, `_assistant_header`):

```python
self._trailing_asst_header = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
self._assistant_header = '<|start_header_id|>assistant<|end_header_id|>'
...
# Trailing 빈 assistant generation prompt 제거 (개행 개수에 robust)
idx = formatted_text.rfind(self._trailing_asst_header)
if idx != -1 and formatted_text[idx + len(self._trailing_asst_header):].strip() == '':
    formatted_text = formatted_text[:idx]
# Sanity: answer 앞 assistant header + 그 뒤 정답 텍스트가 있어야 함
if self._assistant_header not in formatted_text:
    continue
if not formatted_text.rsplit(self._assistant_header, 1)[-1].strip():
    continue
```

검증 (실측):
- `prompt_masks == 0` 기준 target_head 가 정답 문장 첫 단어부터 시작:
  - rel_position: `'\n\nThe tow truck appears to the top-right of the garter snake.<|endoftext|>...'`
  - counting: `'\n\nThere are **0** yellow face emoji's in the image.<|endoftext|>...'`
- 적용 전: `'\n<|endoftext|><|endoftext|>...'` (정답 문장은 prompt 로 흡수, target 은 EOS only)

**영향 범위 — Fix 6 적용 전 counting 학습 산출물은 사실상 정답 문장을 학습하지 못했을 가능성 매우 높음. fix 후 재학습 권장.**

### Fix 5 (counting train): validation gate barrier 비대칭 deadlock 해결

`training/train_mmada_counting.py:698-728` — 4 GPU 학습에서 step 5 의 counting validation 직후 stuck. 원인: `val_counting_items` 가 rank 0 에서만 로드되어 `need_eval = ... and val_counting_items` 가 rank 별로 다르게 평가됨. rank 0 만 분기 진입 → `wait_for_everyone()` 단독 호출 → 다음 step 의 backward all_reduce 와 mismatch.

- 수정: 게이트를 rank-symmetric 으로 (`need_eval = (global_step+1) % eval_every == 0`), `val_counting_items` 데이터 가용성 체크는 `is_main_process` 분기 안쪽으로 이동. `accelerator.wait_for_everyone()` 호출 2회 → 1회, 모든 conditional 바깥으로 hoist.

### Fix 7 (counting train): Lumina식 multi-GPU validation 이식

`/mnt/data6/jiwon/lumina/train/train_unified.py` 의 counting validation 패턴을 MMaDA counting train-time validation 에 최소 diff 로 이식.

- `training/train_mmada_counting.py`
  - `validate_counting()` 을 모든 rank 에서 호출하도록 변경.
  - validation 샘플을 `sample_idx % world_size == rank` 로 rank 별 stride sharding.
  - rank 별 local shard 길이를 기준으로 `tqdm(position=rank)` progress bar 표시. 샘플별 예측 print 도 모든 rank 가 자기 shard 결과를 출력.
  - 각 rank 의 local 결과를 `torch.distributed.all_gather_object()` 로 모은 뒤 rank 0 에서만 Accuracy / MAD / Confusion Matrix / WandB table 계산 및 로깅.
  - MMaDA 의 기존 prompt 구성 (`apply_chat_template` + `<|mmu|><|soi|>image<|eoi|>`) 은 유지.
  - validation 전역 RNG reseed 제거. `temperature=0.0` deterministic generation 이므로 validation 이 이후 training randomness 를 rank 별로 오염시키지 않게 함.
  - `validate_before_train` 추가: true 일 때 training 시작 전 step `global_step` 기준 baseline validation 수행.
  - epoch logging 추가: counting-only IterableDataset 에 대해 `225914 // total_batch_size` 기준 epoch progress 를 wandb/log 에 기록.
  - `answer_format` 전달 추가 (`sentence` / `number`).
- `parquet/my_dataset.py`
  - `CountingDataset(answer_format="sentence")` 인자 추가.
  - `answer_format="number"` 일 때 target 을 raw count string 으로 만들고, question 은 `"Response Example"` 앞부분만 사용.
- `configs/mmada_counting_llada_instruct.yaml`
  - `experiment.validate_before_train`
  - `dataset.params.answer_format`
  - `dataset.preprocessing.max_seq_length` 설명 주석 추가.
- `training/train_mmada_counting_autogpu_debug.sh`
  - 고정 absolute `cd` 제거.
  - debug batch 를 `TRAIN_BATCH_SIZE=1` 로 낮춤.
  - `experiment.name`, `answer_format`, `max_seq_length`, `validate_before_train`, `max_train_steps` override 추가.

검증:

```bash
python -m py_compile training/train_mmada_counting.py
git diff --check -- training/train_mmada_counting.py
```

---

## RelPosition 이식 (2025-05-08)

### Goal

Lumina_Dimoo (`audio_lumina`) 의 `task=rel_position, mode=und` 학습/평가 로직을 MMaDA 에 추가 이식. Counting 이식 패턴을 그대로 mirror.

- 데이터셋: `heez/relative-position-new` (HF Hub) — train 199400 / validation 100 / test 500
- 4-way 분류: `top-left | top-right | bottom-left | bottom-right` (균등 분포)
- 베이스 모델: `Gen-Verse/MMaDA-8B-MixCoT`
- 메트릭: Binary accuracy + per-class accuracy + 4×4 confusion matrix

### 작업 내역

#### 1. `parquet/my_dataset.py` — `RelPositionDataset` 추가

- `CountingDataset` 거의 동일 구조 (IterableDataset, rank+worker 샤딩, `apply_chat_template`, collate_fn)
- 데이터셋: `hf_load_dataset("heez/relative-position-new", split="train")`
- count 필터 / `descriptions` 필터 제거 (rel_position 데이터셋엔 해당 컬럼 없음)
- `item['question']` / `item['answer']` 직접 사용. fallback 로직 삭제 (단일 답 컬럼 신뢰 가능)
- `sample['answer']` 에 `position` 필드 (4 라벨) 저장

#### 2. `parquet/__init__.py` — `RelPositionDataset` export

#### 3. `training/train_mmada_rel_position.py` — RelPosition 전용 훈련 스크립트

- `train_mmada_counting.py` 기반 (Fix 5 적용본 사용)
- `extract_position(text)`: regex `(top-left|top-right|bottom-left|bottom-right)` 매칭
- `validate_rel_position()`: `eval_every` 마다 실행, validation split (max 100 샘플)
  - Generation: `mmu_generate(max_new_tokens=64, steps=64, block_length=64, temperature=0.0)`
  - GT 는 dataset 의 `position` 필드 직접 사용 (regex 불필요)
  - 메트릭: binary accuracy + 4×4 CM (parse 실패는 CM 외부 카운트)
  - WandB: `rel_position/val_accuracy`, `rel_position/parse_fails`, `rel_position/confusion_matrix`, `rel_position/predictions`
- 게이트 + barrier 패턴: counting 의 Fix 5 그대로 (rank-symmetric gate, single `wait_for_everyone()` 분기 밖)
- LoRA / optimizer / scheduler / `TrainStepWrapper` / `save_checkpoint` / `generate_images` 무수정

#### 4. `configs/mmada_rel_position_llada_instruct.yaml`

| 항목 | 값 |
|---|---|
| `pretrained_model_path` | `Gen-Verse/MMaDA-8B-MixCoT` |
| `und_type` | `rel_position` |
| `batch_size_mmu` | 1 |
| `mmu_coeff` | 1.0 / `t2i_coeff`=`lm_coeff`=0.0 |
| `eval_every` | 500 / `save_every` 5000 / `generate_every` 1000 |
| `max_val_rel_position_samples` | 100 |
| `learning_rate` | 2e-5 / `max_train_steps` 50000 / LoRA rank 128 |

#### 5. Wrapper sh 4종

`training/train_mmada_rel_position_{1gpu,8gpu,autogpu,autogpu_debug}.sh` — counting wrapper 4종을 그대로 mirror, `CONFIG`, `MAX_VAL_REL_POSITION_SAMPLES`, echo 라벨, target script 경로만 치환.

- `_8gpu.sh`: DeepSpeed ZeRO-2, `--main_process_port` 명시
- `_autogpu*.sh`: NUM_PROCESSES auto-detect, `auto_multi_gpu.yaml`
- `_autogpu_debug.sh`: `EVAL_EVERY=5 SAVE_EVERY=5 MAX_VAL_REL_POSITION_SAMPLES=5` 디버그 값

#### 6. `evaluation/rel_position/understanding/evaluate_rel_position.py` — 단일 GPU

- `evaluate_pixmo.py` mirror
- `--mode {open, mqa}` (default `open`):
  - `open`: 자유 형식 답변 → regex 추출 (`extract_position`). gen_len=64
  - `mqa`: 질문에 보기 (a/b/c/d) 부착 (`transform_to_mqa`), 첫 글자 letter 추출 (`extract_letter`). gen_len=8
- `--split {test, validation}` (default `test`=500 샘플)
- `--lora_path` 인자 유지 (`PeftModel.from_pretrained` + `merge_and_unload`)
- 출력: `results.json`, `summary.json` (전체 + per-class accuracy), `confusion_matrix_rel_position.png`

#### 7. `evaluation/rel_position/understanding/evaluate_rel_position_multigpu.py` — torchrun 멀티 GPU

- `evaluate_pixmo_multigpu.py` mirror
- 동일한 `--mode` / `--split` / `--lora_path` 인자
- stride 샤딩 (`i % world_size == rank`), 랭크별 `results_rank{n}.jsonl`, rank 0 머지
- Resume 지원

### 실행 방법

```bash
# 훈련 (단일 GPU smoke)
bash training/train_mmada_rel_position_1gpu.sh

# 훈련 (4/8 GPU auto-detect, debug = eval_every 5)
bash training/train_mmada_rel_position_autogpu_debug.sh

# 훈련 (8 GPU DeepSpeed)
bash training/train_mmada_rel_position_8gpu.sh

# 평가 (단일 GPU, open mode, base + LoRA)
python evaluation/rel_position/understanding/evaluate_rel_position.py \
  --model_path Gen-Verse/MMaDA-8B-MixCoT \
  --lora_path mmada-rel-position-llada-instruct/checkpoint-XXXX/unwrapped_model \
  --vq_model_path showlab/magvitv2 \
  --output_dir evaluation_results/rel_position_open --mode open

# 평가 (4 GPU, MQA mode)
torchrun --nproc-per-node=4 --master-port=54321 \
  evaluation/rel_position/understanding/evaluate_rel_position_multigpu.py \
  --model_path Gen-Verse/MMaDA-8B-MixCoT \
  --lora_path <adapter dir> \
  --vq_model_path showlab/magvitv2 \
  --output_dir evaluation_results/rel_position_mqa --mode mqa
```

### 파일 목록 (신규)

| 파일 | 상태 |
|---|---|
| `parquet/my_dataset.py` | 수정 (`RelPositionDataset` 추가) |
| `parquet/__init__.py` | 수정 (export 추가) |
| `training/train_mmada_rel_position.py` | 신규 |
| `configs/mmada_rel_position_llada_instruct.yaml` | 신규 |
| `training/train_mmada_rel_position_{1gpu,8gpu,autogpu,autogpu_debug}.sh` | 신규 4종 |
| `evaluation/rel_position/understanding/evaluate_rel_position.py` | 신규 |
| `evaluation/rel_position/understanding/evaluate_rel_position_multigpu.py` | 신규 |

### RelPosition Fix Log

#### Fix 1 (rel_position): trailing assistant header strip 안정화 (CRITICAL)

위 counting 섹션의 Fix 6 와 동일한 문제 — `RelPositionDataset` 도 `apply_chat_template` + endswith strip 패턴을 사용하므로 같은 silent failure 가 발생. fix 도 동일하게 rfind 기반 robust strip + sanity check (`_trailing_asst_header` / `_assistant_header` 분리) 적용. 본 fix 는 counting Fix 6 와 함께 일괄 patch 됨 (`parquet/my_dataset.py` 의 세 dataset 클래스 — VQADataset, CountingDataset, RelPositionDataset — 동일 패턴 일관 적용). 검증: `prompt_masks == 0` 기준 target_head 가 `'\n\nThe ... top-right ...'` 같은 정답 문장으로 시작.

---

## Counting-Gen 이식 (T2I, 2026-05-09)

### Goal

Lumina-DiMOO `task=counting, mode=gen` 학습/평가 로직을 MMaDA 에 추가 이식. 동일한 `heez/pixmo-point-count-gen-und` 데이터셋을 사용하지만, 기존 counting (und) 가 **이미지→카운트 텍스트** MMU 였던 것과 달리 **descriptions(텍스트)→이미지** T2I 로 학습 목표 반전.

- 데이터셋: `heez/pixmo-point-count-gen-und` (HF Hub) — train split 의 `descriptions` 컬럼이 비어있지 않은 row (gen subset, Lumina parity)
- 베이스 모델: `Gen-Verse/MMaDA-8B-MixCoT`
- 베이스 스크립트: `train_mmada_counting.py` (T2I-only 로 슬롯 교체)
- 메트릭: 자동 카운팅 평가기 미부착 — wandb image table (caption / gt_count / gt_image / gen_image) 시각 로깅만 (Lumina parity)

### 작업 내역

#### 1. `parquet/my_dataset.py` — `CountingGenDataset` 추가

- `CountingDataset` (und) 의 sharding/buffer/repeat 스켈레톤 mirror, chat_template 로직 제거.
- HF 필터 순서 (실측 schema 반영): **descriptions 분할 → numeric count 범위 필터**.
  - 현재 HF train split 의 generation rows 는 `descriptions` 가 non-empty 이지만 `count=None`.
  - gen 모드는 `descriptions` 가 non-empty 인 row 만 유지 (CountingDataset 의 반대).
  - `count` 가 숫자로 존재하는 future schema 에서만 `count_lower_limit <= count <= count_upper_limit` 적용.
- robust 필터 헬퍼 `_has_caption(d)`: `None` / `""` / `"   "` (whitespace) / `[]` / `["..."]` / `"..."` 모두 처리.
- `__iter__` 가 `{'images': squashed_tensor, 'input_ids': caption_string}` yield → `Text2ImageDataset` collate 와 동일 shape, 기존 T2I 슬롯 그대로 소비 가능.
- `image_transform_squash(resolution=512)` 사용 — counting (und/gen) 컨벤션 일관성 (Lumina gen 의 center-crop 과 다르지만 의도적 선택).
- `self.num_samples = len(ds)` 노출 → 학습 스크립트가 직접 읽어 epoch 계산. `num_gen_train_samples` placeholder 불필요.
- 캡션 길이 캡은 dataset 수준 X — `UniversalPrompting.t2i_prompt` 의 자체 truncate (L101-108) 에 위임.

#### 2. `parquet/__init__.py` — `CountingGenDataset` export

#### 3. `training/train_mmada_counting_gen.py` — 신규 학습 스크립트

`train_mmada_counting.py` 베이스, MMU 슬롯 → T2I 슬롯 교체.

- **`T2ITrainStepWrapper`**: 기존 `TrainStepWrapper` (`forward_process` 위임) 가 **사용 불가** — `forward_process` (`models/modeling_mmada.py:213-272`) 가 `batch_size_t2i==0` / `batch_size_lm==0` 가드는 있지만 `batch_size_mmu==0` 가드가 누락 (L246-249, L266-270). T2I-only 호출 시 `p_mask_mmu.to(...)` AttributeError + `masked_indices[-0:]` 가 Python slicing 의미상 전체 슬라이스 (`-0 == 0`) 라 의미적으로도 깨짐.
- 모델 파일에 이미 정의된 `forward_t2i` (L355-378) 도 **그대로 위임 불가** — `attention_bias = torch.ones(...)` 가 `device=` 인자 누락 → CPU 텐서 → CUDA model forward 시 device mismatch. 코드베이스 어디서도 호출되지 않는 dead code.
- 따라서 wrapper 내부에서 **device-safe T2I CE loss 인라인 계산**: `forward_process` L228-244 패턴을 `device=input_ids.device` 명시로 재현. 모델 파일 미수정.

```python
class T2ITrainStepWrapper(torch.nn.Module):
    def forward(self, input_ids, labels, batch_size_t2i, max_seq_length, t2i_masks):
        attention_bias = torch.ones(input_ids.shape[0], 1, ..., device=input_ids.device)
        if batch_size_t2i > 0 and t2i_masks is not None:
            attention_bias_t2i = (t2i_masks[:,:,None] & t2i_masks[:,None,:]).bool().unsqueeze(1)
            attention_bias[:batch_size_t2i] = attention_bias_t2i.to(attention_bias.device)
        logits = self.base(input_ids, attention_bias=attention_bias).logits
        loss_t2i = F.cross_entropy(
            logits[:batch_size_t2i, max_seq_length+1:].contiguous().view(-1, logits.shape[-1]),
            labels[:batch_size_t2i, max_seq_length+1:].contiguous().view(-1),
            ignore_index=-100,
        )
        return loss_t2i
```

- **데이터로더**: T2I-only. LM/MMU 슬롯 dataloader 빌드 자체 제거. `assert batch_size_t2i > 0`, `batch_size_lm == 0`, `batch_size_mmu == 0`.
- **Step-0 sanity**: 이미지 영역만 마스크 카운트 — `labels_t2i[:, msl+2:-1]` slice (text + soi + image + eoi 중 순수 image 만). text label 의 -100 이나 soi/eoi 가 섞이지 않게.
- **`validate_counting_gen()`**: `eval_every` 마다 실행
  - `val_gen` split (max 16 샘플) 사용 — `validation_prompts_file` 기반 `generate_images` 호출 폐기.
  - rank stride sharding (`sample_idx % world_size == rank`).
  - 각 샘플: `t2i_gen` prompt + CFG (`uni_prompting(([caption], image_tokens), 't2i_gen')`) → `t2i_generate(...)` → `vq_model.decode_code` → PIL.
  - **CRITICAL**: `t2i_generate(..., resolution=config.dataset.preprocessing.max_seq_length)` 명시 필수 (Fix 1 참고).
  - PIL → BytesIO PNG bytes 로 변환 후 `dist.all_gather_object` (PIL pickle 안정성 이슈 회피, Lumina parity).
  - rank 0 에서 `wandb.Table(columns=["sample_idx","caption","gt_count","gt_image","gen_image"])` → `wandb.log({"counting_gen/val_images": table})`.
  - 모든 rank 가 함수 끝에서 `model.train()` 복귀 + 호출부 `accelerator.wait_for_everyone()`.

#### 4. `configs/mmada_counting_gen_llada_instruct.yaml` — 훈련 설정

| 항목 | 값 |
|------|-----|
| `pretrained_model_path` | `Gen-Verse/MMaDA-8B-MixCoT` |
| `gen_type` / `und_type` | `"counting_gen"` / `"counting"` (origin 명시) |
| `count_lower_limit / upper_limit` | 0 / 20 |
| `batch_size_t2i` / `batch_size_lm` / `batch_size_mmu` | 1 / 0 / 0 |
| `t2i_coeff` / `lm_coeff` / `mmu_coeff` | 1.0 / 0.0 / 0.0 |
| `guidance_scale` | 4 (Lumina value) |
| `generation_timesteps` | 64 (Lumina value) |
| `cond_dropout_prob` | 0.1 (Stage2 baseline) |
| `max_seq_length` | 512 (T2I text prefix budget; 512-res MMaDA inference convention 과 맞춤) |
| `resolution` | 512 |
| `max_val_counting_gen_samples` | 16 |
| `learning_rate` / `max_train_steps` | 2e-5 / 50000 |
| LoRA rank | 128 |

#### 5. Wrapper sh 4종

`training/train_mmada_counting_gen_{1gpu,8gpu,autogpu,autogpu_debug}.sh` — counting wrapper 4종 mirror, target script + CONFIG 치환.
- 모두 `training.batch_size_lm=0`, `training.batch_size_mmu=0` 명시 (config 실수 방지).
- `_autogpu*.sh` : `training.batch_size_t2i=${TRAIN_BATCH_SIZE}` (counting 의 `batch_size_mmu` override 와 대응), `dataset.preprocessing.max_seq_length=${MAX_SEQ_LENGTH:-512}` 명시.
- `_autogpu_debug.sh` : `MAX_TRAIN_STEPS=10 EVAL_EVERY=5 SAVE_EVERY=10 MAX_VAL_COUNTING_GEN_SAMPLES=4 experiment.validate_before_train=True`.

### 실행 방법

```bash
# 1 GPU smoke
bash training/train_mmada_counting_gen_1gpu.sh

# 4/8 GPU auto, debug (10 step + validate_before_train)
bash training/train_mmada_counting_gen_autogpu_debug.sh

# 8 GPU DeepSpeed
bash training/train_mmada_counting_gen_8gpu.sh

# auto-detect 정식 학습
bash training/train_mmada_counting_gen_autogpu.sh
```

### 파일 목록 (신규)

| 파일 | 상태 |
|---|---|
| `parquet/my_dataset.py` | 수정 (`CountingGenDataset` 추가) |
| `parquet/__init__.py` | 수정 (export 추가) |
| `training/train_mmada_counting_gen.py` | 신규 (865 lines) |
| `configs/mmada_counting_gen_llada_instruct.yaml` | 신규 |
| `training/train_mmada_counting_gen_{1gpu,8gpu,autogpu,autogpu_debug}.sh` | 신규 4종 |

### Counting-Gen Fix Log

본 이식은 plan 단계에서 두 차례 review (Codex 1차/2차) 를 거치며 critical 이슈 6개 + recommended 보완 4개를 사전 차단. 모두 plan 반영 후 구현했으므로 별도 patch 가 아닌 design decision 으로 기록.

#### Fix 1 (counting_gen, CRITICAL): `t2i_generate(resolution=...)` mismatch

`models/modeling_mmada.py:118-211` 의 `t2i_generate` 의 `resolution` 인자는 이름과 달리 실제로는 **CFG text-prefix 길이** (`uncond_prefix = uncond_input_ids[:, :resolution+1]`, L153). 기본값 512 는 demo / stage3-cot config 의 `max_seq_length=512` 와 우연히 맞았던 값. 초기 counting-gen config 처럼 `max_seq_length=256` 으로 호출 시 default 512 면 uncond_prefix 가 text 영역(257)을 넘어 image 영역 일부를 포함 → cond/uncond 의 image 영역 차이 발생 → CFG 의미 손상.

수정: `validate_counting_gen` 의 `t2i_generate` 호출에 `resolution=config.dataset.preprocessing.max_seq_length` 명시. 학습 시 `generate_images` 호출 자체를 제거했으므로 다른 호출 사이트는 없음. 추후 sanity 함수 추가 시에도 동일 인자 전달 필수.

#### Fix 2 (counting_gen, CRITICAL): `forward_process` 의 `batch_size_mmu==0` 미가드 + `forward_t2i` device 버그

(위 작업 내역 #3 의 `T2ITrainStepWrapper` 설명 참고.) wrapper 내부 device-safe 인라인 계산으로 두 버그 모두 회피, 모델 파일 미수정.

#### Fix 3 (counting_gen): `descriptions` 타입 다양성 robust 처리

HF schema 에서 `descriptions` 가 `None` / `""` / `"   "` (whitespace) / `[]` / `["..."]` / `"..."` 모두 가능. `_has_caption()` 헬퍼 + `__iter__` 의 `isinstance(caption, list)` 분기 + `caption.strip()` empty check 로 multi-layer 방어.

#### Fix 4 (counting_gen): `dist.all_gather_object` PIL → PNG bytes 변환

torch 빌드/버전에 따라 PIL 객체 pickle 이 깨질 수 있음. validation 결과를 `BytesIO` PNG bytes 로 변환 후 gather, rank 0 에서 `Image.open(BytesIO(...))` 로 복원해 `wandb.Table` 에 삽입 (Lumina parity).

#### Fix 5 (counting_gen): `validation_prompts_file` 기반 `generate_images` 폐기

사용자 결정사항 "validation 프롬프트 = `val_gen` split 만" 을 엄격히 준수. `validate_counting_gen` 이 `val_gen` captions 으로 wandb 시각 로깅 전담. config 에서도 `validation_prompts_file` 키 제거. 함수 정의는 추후 manual sanity 용도로 남길 수 있으나 main loop 호출 X.

#### Fix 6 (counting_gen): step-0 sanity 의 이미지 영역만 카운트

학습 loss 의 slicing 은 공식 `[:, msl+1:]` 패턴 그대로 유지. `UniversalPrompting` 내부 `max_text_len = config.max_seq_length + 1` 이므로 `msl+1` 은 `<|soi|>` 위치이고, 기존 stage2 T2I 경로와 동일하게 `<|soi|>`, masked image tokens, `<|eoi|>` 가 CE slice 에 포함된다. 단 debug print 의 "masked image tokens" 값은 순수 이미지 토큰만 세도록 `[:, msl+2:-1]` slice 사용 — text label 이나 soi/eoi 가 섞이지 않게.

#### Fix 7 (counting_gen, CRITICAL): train generation subset 0개 문제 해결

실제 실행 (`training/train_mmada_counting_gen_autogpu_debug.sh`, 2026-05-09) 에서 HF cache 생성 후:

```text
CountingGenDataset: after count filter 225914
CountingGenDataset: after descriptions filter 0
```

원인: `train` split 은 앞쪽 225,914개 understanding rows 가 `count=0..20`, `descriptions=None` 이고, 뒤쪽 65,512개 generation rows 가 `descriptions=[caption]`, `count=None`, `question=None`, `answer_count=None` 구조. 기존 구현이 count filter 를 먼저 적용해 generation rows 를 전부 삭제.

수정: `CountingGenDataset` 필터 순서를 `descriptions non-empty` 먼저로 변경하고, numeric `count` 가 있을 때만 count range filter 적용. 현재 schema 에서는 gen subset 65,512개가 유지되어야 함.

#### Fix 8 (counting_gen): `max_seq_length` 256 → 512 조정

`max_seq_length` 는 caption 단독 길이가 아니라 `UniversalPrompting` 의 T2I text prefix budget (`<|t2i|>` + BOS + caption tokens + EOS) 이다. config 256 은 내부 prefix budget 257, 실질 caption budget 약 254 tokens. train gen 65,512개 caption 을 Arrow cache 에서 직접 스캔한 결과:

```text
prompt_prefix_tokens p50=103, p90=131, p95=146, p99=182, p99.9=226, max=521
prefix > 257: 25
prefix > 513: 7
```

256 도 대부분은 커버하지만, 512-res MMaDA inference convention 과 positional distribution 을 맞추고 truncation 을 줄이기 위해 기본 config 와 `_autogpu{,_debug}.sh` 의 `MAX_SEQ_LENGTH` 기본값을 512 로 변경. 예상 sequence 길이: `1283 -> 1539` (`513 text prefix + <|soi|> + 1024 image + <|eoi|>`).
