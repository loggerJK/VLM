# CONTEXT.md

## 운영 규칙

- 모든 작업 시작 시 이 파일을 먼저 확인한다.
- 앞으로 진행하는 작업, 의사결정, 파일 수정 내역, 검증 결과를 이 파일의 작업 로그에 기록한다.
- 사용자 변경 사항은 명시 요청 없이 되돌리지 않는다.

## 작업 로그

### 2026-05-10

- 사용자 요청에 따라 `CONTEXT.md` 존재 여부를 확인했다.
- 기존 `CONTEXT.md`가 없어 새로 생성했다.
- 앞으로 모든 작업 및 수정 내역을 이 파일에 기록하기로 했다.
- BAGEL T2I Generation 학습 코드 지원 범위를 확인했다.
  - 공식 parquet 기반 `t2i_pretrain` 학습 경로와 로컬 HF dataset 기반 `counting/ocr/ocr_synthetic` generation 학습 경로를 확인했다.
  - 코드 수정 없이 `TRAIN.md`, `train/pretrain_unified_navit.py`, `data/dataset_info.py`, `data/t2i_dataset.py`, `data/hf_dataset.py`, `data/configs/*_gen.yaml`, `scripts/train_*gen*.sh`를 읽어 정리했다.
  - 지원 데이터셋(생성용) 4종: `T2IIterableDataset`(parquet), `HFCountingGenDataset`(heez/pixmo-point-count-gen-und), `HFOCRGenDataset`(사용자 지정 hf_dataset_path), `HFSyntheticOCRGenDataset`(agentlans/high-quality-english-sentences + ocr_render).
  - 공통 sequence_plan: `text(enable_cfg=1, loss=0)` → `vae_image(loss=1, MSE)`.
  - 사전 정의 YAML: `example.yaml`, `counting_gen.yaml`, `counting_both.yaml`, `ocr_gen.yaml`, `ocr_both.yaml`, `ocr_synthetic_gen.yaml`, `ocr_synthetic_both.yaml`.
  - 학습 스크립트: `scripts/train_{counting,ocr,ocr_synthetic}_gen_2gpu.sh`, `*_gen_lora_1gpu.sh`, `*_both_*.sh`, 일반은 `train.sh` / `train_bagel.sh`.
  - 핵심 옵션: `--visual_gen=True --visual_und=False`(T2I-only), `--max_latent_size 64`(BAGEL 체크포인트 FT 시 필수), VAE 동결, FSDP HYBRID_SHARD, EMA(0.9999), FLEX packing.
- `--task rel_position --mode gen` 지원 추가 (Lumina-DiMOO `train/train_unified.py:1072-1104` 패턴 차용).
  - 결정사항: caption=`answer` 그대로, mode={und,gen}만(both 의도적 제외), inline `val/pos_acc` 메트릭 유지를 위해 gen 스크립트에서도 `--visual_und True --freeze_vit True --eval_everything` 기본.
  - 선행: `.gitignore` UU(merge conflict) 해소 — `env.sh`, `.venv`, `.uv-cache`, `models/`, `*_results.txt` 등 양쪽 항목 union 보존, `.gitignore`만 단독 stage.
  - 코드 변경:
    - `data/hf_dataset.py`: `HFRelPositionGenDataset` 추가 (counting_gen 구조 답습, `heez/relative-position-new` train split, caption=`item['answer']`, sequence_plan=text(cfg=1)→vae_image(loss=1)).
    - `data/dataset_info.py`: `HFRelPositionGenDataset` import + DATASET_REGISTRY/INFO에 `rel_position_gen` 등록.
    - `train/pretrain_unified_navit.py`:
      - `build_task_dataset_meta`의 rel_position 분기를 und/gen 분기로 확장(both는 명시적 ValueError).
      - `_load_validation_dataset` rel_position 분기 게이트를 `mode in (und, gen)`으로 완화.
      - `_setup_val_gen_prompts`에 rel_position 분기 추가(caption=`item['answer']`).
  - 신규 학습 스크립트: `scripts/train_rel_position_gen.sh` (1gpu LoRA, `expected_num_tokens=8192`), `scripts/train_rel_position_gen_2gpu.sh` (2gpu LoRA, `expected_num_tokens=1` — 시퀀스 패킹 비활성화로 샘플당 1개씩 학습하기 위한 의도된 값, und 2gpu와 동일 정책 유지). `HF_HOME` 안전망(`: "${HF_HOME:=/mnt/data1/huggingface}"`) 삽입.
  - 검증:
    - 레지스트리 sanity: `rel_position_gen` REGISTRY/INFO 모두 존재 확인 OK.
    - yield smoke test (bagel conda env): 199,400 samples 로드, caption="The gasmask is at the top-right of the drum." 형식 확인, sequence_plan/이미지 shape (3,512,512)/num_tokens=1036 모두 정상.
  - 계획 파일: `/home/cvlab22/.claude/plans/task-rel-position-mode-gen-generic-nova.md`.
- Claude 구현 리뷰:
  - `expected_num_tokens=1`은 2GPU 스크립트에서 시퀀스 패킹 비활성화 의도 값으로 확인했으며 결함 후보에서 제외.
  - `data/hf_dataset.py`, `data/dataset_info.py`, `train/pretrain_unified_navit.py`, 신규 rel_position gen 스크립트 2개를 함수/스크립트 단위로 검토.
  - 주요 리스크: 신규 스크립트의 절대 `source`/`cd` 경로가 현재 머신에서 존재하지 않음, inline generation validation은 Lumina GenEval spatial-rel metric과 동일하지 않고 이미지 로깅 수준임.
  - 검증: `py_compile` 및 `bash -n`는 통과했지만, 현재 기본 Python env에서는 `cv2` 미설치로 registry import smoke는 재현하지 못함.
- 리뷰 High 후속 조치(2건):
  - High #1 (스크립트 경로): `scripts/train_rel_position_gen.sh`, `scripts/train_rel_position_gen_2gpu.sh`의 머신 의존 절대 경로 제거. `source /data/mm-llm-backbone_890/.../env.sh`(+`cd`/`conda activate`)를 표준 패턴 `source ./.env`로 교체. 2gpu의 `cd` 라인도 제거하여 repo 루트에서 실행되도록 조정. `bash -n` 통과.
  - High #2 (gen 정량 metric 부재): 사용자 결정 = 표현 정정만(최소 침습). 두 스크립트 헤더에 "Validation metrics" 블록 추가하여 inline `val/pos_acc`/`val/pos_parse_fail_rate`은 understanding metric이며 generation 정량 평가는 `evaluation_scripts/rel_position/run_geneval.sh`로 사후 수행함을 명시. `validate_generation`은 wandb 이미지 sanity 로깅만 함을 분명히 기술. 2gpu 스크립트에는 `expected_num_tokens=1`이 시퀀스 패킹 비활성화 의도 값임을 헤더 주석으로 추가.
  - `/tmp/claude-1001/response.md` 확인: 위 두 High 후속 조치 보고와 실제 스크립트 내용이 일치하며, 두 신규 스크립트 모두 `bash -n` 통과.
