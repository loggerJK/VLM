# CONTEXT.md

이 파일은 이 저장소에서 진행한 작업 내역을 누적 기록합니다. 작업이 끝날 때마다 새 항목을 위에 추가합니다.

---

## 2026-05-09 ~ 2026-05-10 — `rel_position` task 추가 (Qwen2.5-VL-7B LoRA)

### 목표
Lumina-DiMOO의 `train/train_rel_position.sh`(데이터셋만 counting → rel_position으로 바뀐 변형)을 Qwen3-VL fine-tune 파이프라인으로 이식. 기존 `sft_7b_counting.sh` 세팅을 유지하고 task/output_dir/run_name만 교체하는 실험.

### 커밋/푸시
- 커밋: `0989b94 rel_position Qwen 파인튜닝 태스크 추가`
- 브랜치: `blip3-eval`
- 원격 반영: `origin/blip3-eval`

### 변경 파일
- `qwen-vl-finetune/qwenvl/data/hf_dataset.py` — `HFRelPositionDataset` 추가. `HFCelebDataset`을 거의 그대로 따랐고, 기본 HF 경로는 `heez/relative-position-new`(`split="train"`). 스키마는 `{image, question, answer, position}`이지만 학습은 `image/question/answer`만 사용.
- `qwen-vl-finetune/qwenvl/train/train_qwen_hf.py` — `HFRelPositionDataset` import 추가, `elif task == "rel_position"` 디스패치 분기 추가, error 메시지 갱신.
- `qwen-vl-finetune/qwenvl/train/argument.py` — `task` field help에 `'rel_position'` 추가.
- `qwen-vl-finetune/qwenvl/train/callbacks.py` — `_validate_rel_position` 추가 (`heez/relative-position-new`, `split="validation"` streaming, 100 샘플 제한). regex `(top-left|top-right|bottom-left|bottom-right)`로 GT/pred 파싱:
  - GT는 최소 1개 매치 필요(첫 매치 사용).
  - Prediction은 정확히 1개 매치 필요(0개 또는 2개 이상이면 오답).
  - **파싱 실패 = 오답으로 카운트(skip 금지)** — denominator를 줄이지 않기 위한 의도적 결정. accuracy 부풀림 방지.
  - wandb table 컬럼: `[question, gt, pred, parsed_gt, parsed_pred, correct]`.
- `qwen-vl-finetune/scripts/sft_7b_rel_position.sh` (신규, 실행권한 부여) — `sft_7b_counting.sh`에서 `--task rel_position`, `--output_dir ./output/rel_position`, `--run_name qwen2vl-rel-position-lora`만 변경. 나머지(`lr=2e-5`, `model_max_length=8192`, `effective_batch=128`, `lora_r=128`, `epochs=20` 등) 동일.

### 실행 방법
```bash
cd /mnt/data1/jiwon/Qwen3-VL/qwen-vl-finetune
bash scripts/sft_7b_rel_position.sh
```
`./.env`(conda 활성화 + WandB 설정)는 별도로 만들어야 함 — 기존 counting 스크립트와 동일.

GPU를 지정하려면:
```bash
cd /mnt/data1/jiwon/Qwen3-VL/qwen-vl-finetune
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/sft_7b_rel_position.sh
```

출력 디렉토리:
```bash
/mnt/data1/jiwon/Qwen3-VL/qwen-vl-finetune/output/rel_position
```

### 검증 메모
- `python -m py_compile`로 4개 변경 파일 syntax OK 확인.
- `bash -n scripts/sft_7b_rel_position.sh` shell syntax OK 확인.
- `sft_7b_counting.sh`와 diff 확인: 주석, `--task`, `--output_dir`, `--run_name`만 다르고 hyperparameter는 동일.
- `python -c "from qwenvl.data.hf_dataset import HFRelPositionDataset"` dry-run은 시스템 Python의 `tokenizers.decoders.DecodeStream` AttributeError로 실패. 코드 변경 무관, 환경 문제 — conda env 활성화 후 정상 동작 예상.

### 관련 plan 파일
`/home/cvlab22/claude_jiwon/plans/lively-exploring-frog.md`
