# `blip3o/train/train.py` 로직 분석 보고서

## 1. 개요

`train.py`는 BLIP3-o 모델의 학습 파이프라인 전체를 구현하는 핵심 파일이다 (총 1694줄). 주요 역할은 다음과 같다:

- **Argument 파싱**: 모델/데이터/학습 관련 하이퍼파라미터 정의 (HuggingFace dataclass 기반)
- **데이터 로딩**: Counting, OCR Synthetic, WebDataset(Mix) 세 가지 데이터셋 지원
- **전처리**: 대화 포맷팅(Qwen/LLaMA3/Plain), 멀티모달 토큰 삽입, 이미지 프로세싱
- **모델 초기화**: Qwen2.5-VL 또는 LLaMA 백본 로딩, freeze 전략, LoRA 설정
- **학습 실행**: HuggingFace Trainer 기반 (`blip3oTrainer`), DeepSpeed 통합
- **Validation**: 학습 중 이해(Understanding) 및 생성(Generation) 성능 검증 콜백

---

## 2. Argument 정의 (L54~131)

세 개의 `@dataclass` 클래스로 구성된다.

### 2.1 `ModelArguments` (L54~78)

모델 아키텍처 및 체크포인트 관련 인자.

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `model_name_or_path` | `"facebook/opt-125m"` | 기본 LLM 모델 경로 (실제 사용 시 Qwen2.5-VL 경로) |
| `version` | `"v0"` | 대화 템플릿 버전 (`qwen`, `llama3`, `plain` 등) |
| `freeze_backbone` | `True` | LLM 백본 파라미터 동결 여부 |
| `tune_mm_mlp_adapter` | `False` | 멀티모달 프로젝터만 학습할지 여부 |
| `vision_tower` | `None` | 이해(Understanding)용 비전 인코더. `None`이면 Qwen-VL 내장 비전 사용 |
| `gen_vision_tower` | `None` | 생성(Generation)용 비전 인코더 (e.g., `eva-clip-E-14-plus`) |
| `mm_projector_type` | `"linear"` | 이해용 프로젝터 타입 (`linear`, `mlp2x_gelu` 등) |
| `gen_projector_type` | `"linear"` | 생성용 프로젝터 타입 |
| `n_query` | `729` | 생성용 latent query 개수 (CLIP 576, SigLIP 729) |
| `n_und_query` | `729` | 이해용 query 개수 |
| `gen_pooling` | `"all"` | 피처 풀링 전략 (`pool2d_3`, `seq_3`, `early_pool2d_4` 등) |
| `resume_ckpt` | `None` | 수동 체크포인트 resume 경로 |
| `auto_resume` | `False` | output_dir에서 최신 체크포인트 자동 검색 |

### 2.2 `DataArguments` (L80~96)

데이터 로딩 및 전처리 관련 인자.

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `data_path` | `None` | 학습 데이터 경로 |
| `image_folder` | `None` | WebDataset tar 파일이 있는 이미지 폴더 |
| `data_type` | `"mix"` | 데이터 타입: `counting`, `ocr_synthetic`, `mix` |
| `task` | `"counting"` | 태스크 종류: `counting` \| `ocr_synthetic` |
| `mode` | `"gen"` | 학습 모드: `und` \| `gen` \| `both` |
| `hf_dataset_path` | `"heez/pixmo-point-count-gen-und"` | HuggingFace 데이터셋 경로 |
| `ocr_num_samples` | `200000` | OCR 학습 샘플 수 |
| `ocr_image_width` | `512` | OCR 렌더링 이미지 크기 |
| `image_aspect_ratio` | `"square"` | 이미지 비율 처리 방식 (`square`, `pad`) |

### 2.3 `TrainingArguments` (L98~131)

HuggingFace `TrainingArguments`를 상속하여 확장.

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `optim` | `"adamw_torch"` | 옵티마이저 |
| `model_max_length` | `512` | 최대 시퀀스 길이 |
| `bits` | `16` | 양자화 비트 (4/8/16) |
| `lora_enable` | `False` | LoRA 활성화 여부 |
| `lora_r` | `64` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha |
| `lora_target_modules` | `"q_proj,k_proj,v_proj,o_proj"` | LoRA 적용 대상 모듈 (쉼표 구분) |
| `mm_projector_lr` | `None` | 멀티모달 프로젝터 별도 학습률 |
| `validation_interval` | `500` | 이해(Understanding) 검증 주기 (글로벌 스텝) |
| `validation_samples` | `100` | 검증 샘플 수 |
| `log_freq` | `250` | 생성(Generation) 검증/로깅 주기 |

---

## 3. 유틸리티 함수 (L133~286)

### 3.1 DeepSpeed ZeRO-3 파라미터 처리

#### `maybe_zero_3(param, ignore_status, name)` (L133~145)
- DeepSpeed ZeRO-3에서는 파라미터가 여러 GPU에 분산(partitioned)됨
- `ds_id` 속성이 있으면 ZeRO-3 파라미터로 판단
- `GatheredParameters`로 전체 파라미터를 모아서 CPU로 복사 후 반환
- ZeRO-3가 아니면 단순 `.detach().cpu().clone()`

#### `get_peft_state_maybe_zero_3(named_params, bias)` (L149~171)
- LoRA 파라미터만 추출하는 함수 (peft 라이브러리의 `get_peft_model_state_dict` 기반)
- `bias` 옵션에 따라 어떤 bias를 포함할지 결정:
  - `"none"`: LoRA 파라미터만
  - `"all"`: LoRA + 모든 bias
  - `"lora_only"`: LoRA + LoRA 관련 bias만

#### `get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only)` (L174~179)
- LoRA가 **아닌** 학습 가능 파라미터를 추출
- `require_grad_only=True`이면 `requires_grad`인 것만 필터링

#### `get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match)` (L182~185)
- 멀티모달 어댑터(프로젝터) 파라미터를 키 매칭으로 추출

#### `get_vision_tower_state_maybe_zero_3(named_params, keys_to_match)` (L188~191)
- 비전 타워 파라미터를 키 매칭으로 추출

### 3.2 LoRA 유틸리티

#### `find_all_linear_names(model)` (L194~207)
- 모델 내 모든 `nn.Linear` 모듈 이름을 수집
- `mm_projector`, `vision_tower`, `vision_resampler` 관련 모듈은 **제외**
- `lm_head`도 제외 (16비트 학습 시 필요)
- LoRA 자동 적용 시 대상 모듈 목록으로 사용

### 3.3 모델 저장

#### `safe_save_model_for_hf_trainer(trainer, output_dir, vision_tower)` (L210~269)
- 체크포인트 저장의 핵심 함수
- **저장 순서**:
  1. `mm_projector` (이해용 프로젝터) 파라미터를 별도 `.bin`으로 저장
  2. `gen_projector` (생성용 프로젝터) 파라미터를 별도 `.bin`으로 저장
  3. DeepSpeed 사용 시 `trainer.save_model()` 호출
  4. 비-DeepSpeed 시 전체 state_dict를 CPU로 옮겨 저장
- 체크포인트 폴더(`checkpoint-*`)일 때는 `mm_projector/`, `gen_projector/` 하위 폴더에 저장

### 3.4 토크나이저 리사이즈

#### `smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)` (L272~285)
- 특수 토큰 추가 후 임베딩 크기 조정
- 새로 추가된 토큰의 임베딩을 기존 임베딩의 **평균값**으로 초기화

---

## 4. 전처리 파이프라인 (L288~594)

### 4.1 토큰화 기본 함수

#### `_tokenize_fn(strings, tokenizer)` (L288~307)
- 문자열 리스트를 배치 토큰화
- 반환: `input_ids`, `labels`, 각각의 길이

#### `_mask_targets(target, tokenized_lens, speakers)` (L310~318)
- 대화에서 **human** 발화 부분을 `IGNORE_INDEX`로 마스킹
- 모델이 human 발화를 학습하지 않도록 처리

#### `_add_speaker_and_signal(header, source)` (L321~338)
- 각 발화 앞에 `### {speaker}: ` 시그널 추가, 끝에 `\n` 추가
- 기본(legacy) 대화 포맷에 사용

### 4.2 멀티모달 전처리

#### `preprocess_multimodal(sources, data_args)` (L342~358)
- `<image>` 토큰의 위치와 역할에 따라 **inst_type**을 결정:
  - **human 발화에 `<image>`** → `inst_type = "und"` (이해 태스크)
    - `<image>`를 `<|vision_start|>` + `<|image_pad|> × n_und_query` + `<|vision_end|>`로 치환
  - **gpt 발화에 `<image>`** → `inst_type = "gen"` (생성 태스크)
    - `<image>`를 빈 문자열로 치환 (실제 생성 토큰은 DataCollator에서 삽입)
- 반환: 변환된 sources, inst_type

### 4.3 대화 포맷팅 함수

#### `preprocess_qwen(sources, tokenizer, ...)` (L364~414) — **주력 사용**
- Qwen 모델 전용 대화 포맷
- Chat template: `<|im_start|>{role}\n{content}<|im_end|>\n`
- 처리 흐름:
  1. 시스템 메시지를 먼저 인코딩 → 전부 `IGNORE_INDEX`로 마스킹
  2. 각 turn을 `apply_chat_template`으로 인코딩
  3. `user`/`system` 역할 → `IGNORE_INDEX` 마스킹
  4. `assistant` 역할 → 학습 대상 (레이블 유지)

#### `preprocess_llama3(sources, tokenizer, ...)` (L419~501)
- LLaMA-3 모델 전용 대화 포맷
- 추가 처리: BOS 토큰 중복 제거, unmask 토큰 (`<|begin_of_text|>` 등)을 레이블에서 복원

#### `preprocess_plain(sources, tokenizer)` (L505~523)
- 단순 포맷: human 발화 + gpt 발화 + separator를 이어붙임
- human 부분만 `IGNORE_INDEX`로 마스킹

#### `preprocess(sources, tokenizer, has_image)` (L526~570) — **라우터 함수**
- `conversation_lib.default_conversation`의 설정에 따라 적절한 전처리 함수로 분기:
  - `PLAIN` → `preprocess_plain`
  - `llama3` → `preprocess_llama3`
  - `qwen` → `preprocess_qwen`
  - 그 외 → legacy 포맷 (헤더 + 시그널 방식)

### 4.4 이미지 처리

#### `img_process(images, processor, image_aspect_ratio)` (L575~594)
- `image_aspect_ratio == "pad"`: 정사각형으로 패딩 후 프로세서 적용
- 그 외: 프로세서 직접 적용 (`processor.preprocess(images)`)
- 파일 상단의 `transform_und_images`는 이해용 이미지를 448×448로 리사이즈+센터크롭 (L34)

---

## 5. 데이터셋 클래스 (L597~995)

### 5.1 `CountingGenDataset` (L597~651)

**Counting 생성 태스크** 전용 데이터셋.

- **데이터 소스**: HuggingFace `heez/pixmo-point-count-gen-und` (train split)
- **필터링**: `descriptions` 필드가 존재하는 샘플만 사용
- **대화 구성**:
  ```
  human: "Please generate image based on: {caption}"
  gpt:   "<image>"
  ```
- **전처리 흐름**: `preprocess_multimodal` → `preprocess` → `img_process`(gen_image_processor)
- **에러 핸들링**: 예외 발생 시 랜덤 인덱스로 재시도 (while True 루프)

### 5.2 `OCRSyntheticGenDataset` (L653~714)

**OCR 합성 생성 태스크** 전용 데이터셋.

- **데이터 소스**: HuggingFace `agentlans/high-quality-english-sentences` (train split)
- **이미지 생성**: `blip3o.ocr_render.generate_image()`로 텍스트를 이미지로 렌더링
  - 템플릿: `"random"`, 크기: `ocr_image_width × ocr_image_width`
  - 텍스트 앞에 `# ` 마크다운 헤더 추가
- **대화 구성**: CountingGenDataset과 동일 패턴
- **샘플 수 제한**: `data_args.ocr_num_samples`로 최대 샘플 수 조절

### 5.3 `LazySupervisedMixDataset` (L719~902)

**범용 혼합(Mix) 데이터셋** — 원래 BLIP3-o 학습용.

- **데이터 로딩** (L748~768):
  - `image_folder`에서 `*.tar` WebDataset 파일들을 `load_dataset("webdataset", ...)`로 로드
  - 컬럼 리네이밍: `jpg` → `image`, `type` = `"T2I"` 추가
  - 여러 데이터셋을 `concatenate_datasets`로 결합 후 셔플

- **`__getitem__` 로직** (L791~902):
  - **T2I (Text-to-Image)**: caption 기반 생성 대화 구성
    ```
    human: "Please generate image based on the following caption: {txt}"
    gpt:   "<image>"
    ```
  - **I2I (Image-to-Image)**: 이미지 재구성 대화 구성
    ```
    human: "<image>\nPlease reconstruct the given image."
    gpt:   ""
    ```
  - **이미지 처리**:
    - `gen` 태스크: `gen_image_processor`로 처리 → `data_dict["gen_image"]`
    - `und` 태스크: 448×448 리사이즈 후 Qwen 프로세서 적용 → `data_dict["und_image"]` + `data_dict["grid_thw"]`
    - `und` 태스크에서도 `gen_image`를 함께 저장 (이해+생성 동시 학습 가능)
  - **에러 핸들링**: 이미지 로드/처리 실패 시 랜덤 인덱스로 재시도

### 5.4 `make_supervised_data_module(tokenizer, data_args)` (L983~995)

데이터셋 팩토리 함수:
- `data_type == "counting"` → `CountingGenDataset`
- `data_type == "ocr_synthetic"` → `OCRSyntheticGenDataset`
- `data_type == "mix"` → `LazySupervisedMixDataset`
- 공통 `DataCollatorForSupervisedDataset` 반환, `eval_dataset=None`

---

## 6. DataCollator (L905~980)

### `DataCollatorForSupervisedDataset`

배치를 구성하는 핵심 클래스. 단순 패딩 외에 **생성용 이미지 토큰 삽입** 로직이 포함된다.

#### 처리 흐름:

1. **시퀀스 트리밍 + 이미지 토큰 삽입** (L916~927):
   - 각 시퀀스를 `model_max_length - 65`로 자름
   - 끝에 **65개의 이미지 토큰** 추가:
     - 첫 토큰: `151665` (`<|image_pad|>` 시작 마커)
     - 나머지 64개: `IMAGE_TOKEN_IDX` (151667)
   - `i_s_pos`: 이미지 시퀀스 시작 위치 기록 (원래 시퀀스 길이 + 1)

2. **패딩** (L932~937):
   - `input_ids`: `pad_token_id`로 패딩
   - `labels`: `IGNORE_INDEX`로 패딩
   - `attention_mask`: pad가 아닌 위치 = 1

3. **이미지 배치 구성** (L944~978):
   - **`gen_image`**: 생성용 이미지 텐서를 배치 차원으로 concat (shape 일치 시)
   - **`und_image`**: 이해용 이미지 텐서를 배치 차원으로 concat
   - **`grid_thw`**: Qwen-VL의 이미지 그리드 정보 (temporal, height, width)
   - 이미지가 없으면 `None`

4. **반환 딕셔너리**:
   ```python
   {
       "input_ids": Tensor,       # (B, seq_len)
       "labels": Tensor,          # (B, seq_len)
       "attention_mask": Tensor,  # (B, seq_len)
       "gen_image": Tensor|None,  # (B, C, H, W) 생성 대상 이미지
       "und_image": Tensor|None,  # (B, patches, D) 이해 입력 이미지
       "grid_thw": Tensor|None,   # (B, 3) 이미지 그리드 정보
       "ids": List[str],          # 샘플 ID
       "i_s_pos": List[int],      # 이미지 시퀀스 시작 위치
   }
   ```

---

## 7. Validation Callback (L1031~1426)

### `ValidationCallback(TrainerCallback)`

HuggingFace Trainer의 콜백 시스템을 활용하여 학습 중 검증을 수행.

### 7.1 초기화 (L1034~1051)
- 검증 관련 설정 저장: `validation_interval`, `log_freq`, `validation_samples`
- 검증 데이터셋은 **lazy loading** (처음 접근 시 로드)

### 7.2 검증 데이터 로딩

#### `_get_val_und_dataset()` (L1053~1076)
- **counting**: `val_und` split 로드
- **ocr_synthetic**: `agentlans/high-quality-english-sentences`의 마지막 N개 샘플을 검증용으로 사용

#### `_get_val_gen_prompts()` (L1078~1100)
- **counting**: `val_gen` split에서 5개 description 추출 (실패 시 하드코딩 폴백)
- **ocr_synthetic**: 5개 하드코딩 텍스트 프롬프트

### 7.3 콜백 이벤트 핸들러

#### `on_train_begin` (L1115~1125)
- 학습 가능 파라미터 이름을 `optimized_param_names.txt`로 저장 (rank 0만)

#### `on_step_end` (L1127~1143) — **핵심 콜백**
- **체크포인트 저장**: `save_steps` 간격으로 `_save_checkpoint` 호출
- **이해 검증**: `validation_interval` 간격으로 `_validate_understanding` 호출 (mode가 `und` 또는 `both`)
- **생성 검증**: `log_freq` 간격으로 `_validate_generation` 호출 (mode가 `gen` 또는 `both`)

#### `on_epoch_end` (L1145~1149)
- 에포크 종료 시 `epoch{N}` 이름으로 체크포인트 저장

### 7.4 체크포인트 저장 (`_save_checkpoint_to`, L1158~1202)

저장 항목:
1. **LoRA 어댑터**: `save_pretrained()` (LoRA 활성화 시)
2. **생성 컴포넌트**: `gen_components.pt`로 저장
   - `dit`: 디퓨전 트랜스포머 state_dict
   - `down_projector`: 다운 프로젝터 state_dict
   - `latent_queries`: 잠재 쿼리 텐서
3. **토크나이저**: `save_pretrained()`

### 7.5 이해(Understanding) 검증

#### `_validate_counting_understanding` (L1226~1296)
- Qwen2.5-VL의 `generate()`로 counting 질문에 대한 답변 생성
- 정답과 비교: **정확도(Accuracy)**, **평균 절대 편차(MAD)** 산출
- WandB에 `val/accuracy`, `val/mad` 로깅

#### `_validate_ocr_understanding` (L1298~1370)
- OCR 렌더링 이미지에서 텍스트 추출 → `blip3o.ocr_metrics.calculate_metrics`로 평가
- 메트릭: **WER**, **CER**, **METEOR**, **BLEU**, **F1**
- WandB에 OCR 메트릭 전체 로깅

### 7.6 생성(Generation) 검증 (`_validate_generation`, L1372~1426)

- WandB가 없으면 스킵
- `DiffusionPipeline`을 로드하여 5개 프롬프트로 이미지 생성
  - guidance_scale=3.0, num_inference_steps=50
- 생성된 이미지를 `wandb.Image`로 변환하여 `val/generated_images`에 로깅
- 완료 후 파이프라인 삭제 및 `torch.cuda.empty_cache()`

---

## 8. 메인 `train()` 함수 (L1430~1693)

### 8.1 Argument 파싱 및 초기 설정 (L1430~1442)

```python
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
```

- `local_rank` 글로벌 변수 설정
- `compute_dtype` 결정: fp16 → float16, bf16 → bfloat16, 기본 → float32
- `task`가 `counting` 또는 `ocr_synthetic`이면 `data_type`을 해당 값으로 재설정

### 8.2 양자화 설정 (L1443~1463)

- `bits`가 4 또는 8이면 `BitsAndBytesConfig` 구성
- `llm_int8_skip_modules=["mm_projector"]`: 프로젝터는 양자화에서 제외

### 8.3 모델 로딩 (L1465~1483)

**분기 기준**: `vision_tower` 인자의 유무

| 조건 | 로드 모델 | 설명 |
|------|-----------|------|
| `vision_tower is not None` | `blip3oLlamaForCausalLM` | 별도 비전 인코더 + LLaMA 백본 |
| `vision_tower is None` | `blip3oQwenForCausalLM` | Qwen2.5-VL 내장 비전 사용 (주력) |

### 8.4 Freeze 전략 (L1485~1501)

`freeze_backbone=True`일 때:
- `model.get_model()` (내부 모델) 전체 파라미터 동결
- `model.visual` (Qwen-VL 내장 비전 인코더) 동결
- `model.lm_head` 동결
- → 생성 관련 컴포넌트(DiT, latent_queries, down_projector 등)만 학습

`gradient_checkpointing` 활성화 시:
- `enable_input_require_grads()` 호출 또는 forward hook으로 입력 grad 활성화

### 8.5 토크나이저 초기화 (L1503~1530)

1. `AutoProcessor.from_pretrained()`에서 `.tokenizer` 추출 시도
2. `model_max_length` 설정
3. 특수 토큰 추가: `[IMG]`, `[/IMG]`, `<image>`, `<pad>` (필요 시)
4. `conversation_lib.default_conversation`을 `version` 인자에 맞게 설정

### 8.6 비전 모듈 초기화 (L1534~1565)

```python
model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
```

- 생성용 비전 타워 초기화: `gen_vision_tower` → bfloat16/float16, 동결(requires_grad=False)
- 데이터 인자에 프로세서 설정:
  - `gen_image_processor`: 생성용 비전 타워의 프로세서
  - `image_processor`: Qwen2.5-VL-7B-Instruct의 이미지 프로세서
- 모델 config에 이미지 관련 설정 반영

### 8.7 체크포인트 Resume (L1567~1634)

#### Resume 경로 결정:
1. `model_args.resume_ckpt` (수동 지정)
2. `auto_resume=True` → `get_latest_checkpoint(output_dir)` (최신 스텝 기준)

#### LoRA Resume (L1575~1602):
- 기존 체크포인트에 `adapter_config.json`이 있으면 `PeftModel.from_pretrained()`로 로드
- 없으면 새 `LoraConfig` 생성 후 `get_peft_model()` 적용

#### 생성 컴포넌트 Resume (L1604~1634):
- `gen_components.pt` 파일에서 `dit`, `down_projector`, `latent_queries` 복원
- 체크포인트 이름에서 epoch/step 파싱 (로깅용)

### 8.8 학습 실행 (L1636~1693)

```
총 파라미터 / 학습 가능 파라미터 수 출력
    ↓
make_supervised_data_module() → 데이터셋 + 데이터 콜레이터
    ↓
ValidationCallback 생성
    ↓
blip3oTrainer 생성 (model, tokenizer, args, callbacks, data_module)
    ↓
rank 0에서 전체 파라미터 목록 + trainable 여부 테이블 출력
    ↓
HF checkpoint-* 존재 여부 확인
    ↓
trainer.train(resume_from_checkpoint=True/None)
    ↓
trainer.save_state()
    ↓
safe_save_model_for_hf_trainer() → 최종 모델 저장
```

#### `blip3oTrainer`의 역할:
- HuggingFace Trainer를 확장하여 **파라미터 그룹별 학습률** 지원
- `mm_projector_lr`이 설정되면 멀티모달 프로젝터에 별도 학습률 적용
- DeepSpeed ZeRO-1이 기본 (`deepspeed_scripts/zero1.json`)

---

## 9. Loss 계산 상세 분석

Loss는 `train.py`가 아닌 **모델의 `forward()` 메서드**에서 계산된다. 핵심 구현은 `blip3o/model/language_model/blip3o_qwen.py`의 `blip3oQwenForCausalLM.forward()` (L56~174)에 위치한다.

### 9.1 전체 Loss 구조

```
total_loss = img_loss  (디퓨전 loss만 사용, LM loss는 계산하지만 최종 loss에 포함하지 않음)
```

현재 코드에서 **최종 loss(`total_loss`)는 이미지 생성 loss(`img_loss`)만으로 구성**된다. 언어 모델링 loss(`loss`)는 계산되지만 `total_loss`에 합산되지 않는다.

### 9.2 Loss 계산 흐름 (단계별)

#### Step 1: 멀티모달 입력 준비 (`prepare_inputs_labels_for_multimodal`)

`blip3o_arch.py`의 `blip3oMetaForCausalLM.prepare_inputs_labels_for_multimodal()` (L263~368)에서 수행:

```python
# 1. 생성용 비전 타워로 타겟 이미지 인코딩
prompt_image_embeds = gen_vision_tower(gen_images)        # (B, 729, 1792) — CLIP 피처
target_image_embeds = torch.clone(prompt_image_embeds).detach()  # 학습 타겟 (detach)

# 2. latent_queries를 생성 이미지 토큰 위치에 삽입
latent_queries = self.get_model().latent_queries.repeat(B, 1, 1)   # (B, n_query, hidden_size)
text_embeds[gen_img_idx] = latent_queries   # 생성 위치에 학습 가능한 쿼리 삽입

# 3. 이해 이미지가 있으면 해당 위치에 비전 임베딩 삽입
if und_images is not None:
    und_image_embeds = vision_tower(und_images, grid_thw=grid_thw)
    text_embeds[und_img_idx] = und_image_embeds

# 4. 이미지 토큰 위치의 labels를 -100으로 설정 (LM loss에서 제외)
labels[image_idx] = -100
```

**핵심 포인트**: `target_image_embeds`(CLIP 피처)가 `latents`로 반환되어 디퓨전 loss 계산에 사용된다.

#### Step 2: LLM Forward Pass

```python
outputs = self.model(inputs_embeds=text_embeds, ...)  # Qwen2.5-VL 트랜스포머 실행
hidden_states = outputs[0]                             # (B, seq_len, hidden_size)
logits = self.lm_head(hidden_states)                   # (B, seq_len, vocab_size)
```

#### Step 3: 언어 모델링 Loss (CrossEntropy) — 계산만 수행

```python
shift_logits = logits[..., :-1, :].contiguous()       # 시프트: 토큰 n-1로 n 예측
shift_labels = labels[..., 1:].contiguous()
loss_fct = torch.nn.CrossEntropyLoss()                # IGNORE_INDEX(-100)인 위치는 자동 무시
loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
```

> **주의**: 이 `loss`는 현재 코드에서 `total_loss`에 합산되지 않는다. 즉, **LLM의 텍스트 생성 능력은 학습하지 않고 이미지 생성만 학습**하는 구조이다.

#### Step 4: 이미지 Hidden States 추출

```python
img_hidden_states = []
for b in range(hidden_states.shape[0]):
    img_hidden_states.append(hidden_states[b, i_s_pos[b]:i_s_pos[b]+64, :])
img_hidden_states = torch.stack(img_hidden_states, dim=0)  # (B, 64, hidden_size)
img_hidden_states = self.get_model().down_projector(img_hidden_states)  # Identity (현재)
```

- `i_s_pos`: DataCollator에서 기록한 이미지 시퀀스 시작 위치
- 각 배치에서 **64개의 이미지 토큰 위치**의 hidden state를 추출
- `down_projector`는 현재 `IdentityMap`으로 설정되어 있어 변환 없이 통과

#### Step 5: 디퓨전 Loss (Flow Matching MSE) — **실제 학습 loss**

```python
if latents is None:
    # latents가 없으면 self-supervised 더미 loss (0에 수렴)
    img_loss = mse_loss(img_hidden_states, img_hidden_states.detach())
else:
    # === Flow Matching 디퓨전 Loss ===
    bsz = latents.shape[0]
    noise = torch.randn_like(latents)                           # 가우시안 노이즈 샘플링
    u = torch.rand(size=(bsz,), device="cpu")                  # 균등분포 [0,1)
    indices = (u * num_train_timesteps).long()                  # 타임스텝 인덱스
    timesteps = noise_scheduler.timesteps[indices]              # 실제 타임스텝 값
    sigmas = self.get_sigmas(timesteps, ...)                    # 시그마 (노이즈 레벨)

    # 노이즈 보간: noisy_latents = (1-σ)·latents + σ·noise
    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

    # DiT가 노이즈 예측
    noise_pred = self.get_model().dit(
        x=noisy_latents,                                        # 노이즈가 추가된 CLIP 피처
        timestep=timesteps,                                     # 타임스텝 조건
        z_latents=self.mask_drop(img_hidden_states),            # LLM hidden states (조건)
    )

    # 타겟: velocity = noise - latents (Flow Matching 목표)
    target = noise - latents
    img_loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

total_loss = img_loss  # 최종 loss = 디퓨전 loss만
```

### 9.3 Flow Matching 디퓨전 Loss 상세

BLIP3-o의 이미지 생성은 **Flow Matching** 방식을 사용한다. 이는 DDPM과 달리 직선 경로(straight path)를 따르는 확률 흐름 ODE를 학습한다.

| 구성 요소 | 설명 |
|-----------|------|
| **latents** | `gen_vision_tower`(EVA-CLIP)가 타겟 이미지에서 추출한 CLIP 피처. shape: `(B, num_patches, 1792)` |
| **noise** | 표준 가우시안 노이즈 `N(0, I)` |
| **sigmas (σ)** | `noise_scheduler`에서 가져온 노이즈 레벨. `FlowMatchEulerDiscreteScheduler` 기반 |
| **noisy_latents** | 보간: `(1-σ)·x₀ + σ·ε` — 깨끗한 피처와 노이즈의 선형 혼합 |
| **target** | `ε - x₀` (velocity target) — 노이즈에서 원본 데이터 방향의 속도 벡터 |
| **noise_pred** | DiT(NextDiT with Cross-Attention)의 예측값 |
| **loss** | `MSE(noise_pred, target)` — velocity 예측 오차의 평균 제곱 |

#### Flow Matching 수식

```
Forward process:  x_t = (1 - σ_t) · x_0 + σ_t · ε,     ε ~ N(0, I)
Velocity target:  v = ε - x_0
Loss:             L = E_{t,ε} [ || DiT(x_t, t, c) - v ||² ]
```

여기서 `c`는 LLM의 hidden states (텍스트 조건)이다.

### 9.4 Classifier-Free Guidance를 위한 Mask Drop

```python
def mask_drop(self, latents, drop_prob=0.1):
    mask = torch.bernoulli(torch.zeros(B) + drop_prob)
    mask = 1 - mask  # 10% 확률로 0, 90% 확률로 1
    return latents * mask
```

- 학습 시 10% 확률로 조건(LLM hidden states)을 **0으로 드롭**
- 이를 통해 추론 시 Classifier-Free Guidance(CFG) 적용 가능
- 추론 시 `guidance_scale`을 사용하여 조건부/무조건부 예측을 보간

### 9.5 DiT (NextDiT with Cross-Attention) 구조

Loss 계산에서 핵심 역할을 하는 DiT의 구성:

| 설정 | 값 |
|------|-----|
| 모델 | `LuminaNextDiT2DModel` |
| `input_size` | 8 (8×8 공간 격자) |
| `patch_size` | 1 |
| `in_channels` | 1792 (CLIP 피처 차원) |
| `dim` | 1792 (모델 히든 차원) |
| `n_layers` | 24 |
| `n_heads` | 28 |
| `cross_attention_dim` | 3584 (= Qwen2.5-VL hidden_size, LLM 조건) |

DiT는 **CLIP 피처 공간**에서 직접 디노이징을 수행한다 (VAE latent가 아닌 CLIP 피처를 확산).

### 9.6 Loss 데이터 흐름 요약도

```
┌─────────────────────────────────────────────────────────────────────┐
│                        forward() in blip3o_qwen.py                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  gen_image ──→ gen_vision_tower(EVA-CLIP) ──→ latents (타겟 CLIP 피처)│
│                                                  │                   │
│  input_ids ──→ prepare_inputs_labels_for_multimodal()               │
│       │            │                                                 │
│       │            ├── latent_queries → 생성 토큰 위치에 삽입          │
│       │            └── und_image_embeds → 이해 토큰 위치에 삽입        │
│       │                                                              │
│       └──→ LLM (Qwen2.5-VL) ──→ hidden_states                      │
│                                      │                               │
│                    ┌─────────────────┤                               │
│                    │                 │                                │
│                    ▼                 ▼                                │
│            lm_head(logits)     img_hidden_states                    │
│                    │            (i_s_pos 기준 64토큰 추출)            │
│                    │                 │                                │
│                    ▼                 ▼                                │
│          CrossEntropyLoss     down_projector (Identity)              │
│           (계산만, 미사용)          │                                 │
│                                     ▼                                │
│                              mask_drop (10% 조건 드롭)               │
│                                     │                                │
│                                     ▼                                │
│                              ┌──────────────┐                       │
│               noise ───→     │  DiT Forward  │ ←── noisy_latents    │
│                              │  (NextDiT)    │ ←── timesteps        │
│                              └──────┬───────┘                       │
│                                     │                                │
│                                     ▼                                │
│                              noise_pred                              │
│                                     │                                │
│                                     ▼                                │
│                   target = noise - latents                           │
│                                     │                                │
│                                     ▼                                │
│                   img_loss = MSE(noise_pred, target)                │
│                                     │                                │
│                                     ▼                                │
│                          total_loss = img_loss ──→ backward()        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.7 학습 가능 파라미터와 Loss의 관계

`freeze_backbone=True` 설정 시 loss의 gradient가 흐르는 컴포넌트:

| 컴포넌트 | 학습 여부 | Loss에서의 역할 |
|----------|-----------|----------------|
| LLM 백본 (Qwen2.5-VL) | **동결** (LoRA 활성 시 LoRA 파라미터만 학습) | hidden_states 생성 |
| `lm_head` | **동결** | logits 생성 (loss에 미사용) |
| `visual` (Qwen 내장 비전) | **동결** | 이해 이미지 인코딩 |
| `gen_vision_tower` (EVA-CLIP) | **동결** | 타겟 CLIP 피처 생성 |
| `latent_queries` | **학습** | LLM에 주입되는 학습 가능 쿼리 |
| `down_projector` | **학습** (현재 Identity) | hidden states → DiT 입력 변환 |
| `dit` (NextDiT) | **학습** | 디노이징 네트워크 (loss의 핵심) |
| LoRA 파라미터 | **학습** (활성 시) | LLM hidden states 품질 향상 |

---

## 부록: 핵심 상수 및 글로벌 설정

| 항목 | 값 | 위치 |
|------|-----|------|
| `IGNORE_INDEX` | -100 | `blip3o/constants.py` |
| `IMAGE_TOKEN_IDX` | 151667 | `blip3o/constants.py` |
| `UND_IMAGE_TOKEN_IDX` | 151655 | `blip3o/constants.py` |
| `transform_und_images` | Resize(448) + CenterCrop(448) | L34 |
| `ImageFile.LOAD_TRUNCATED_IMAGES` | `True` | L33 |

## 부록: 함수 호출 관계도

```
train()
├── HfArgumentParser.parse_args_into_dataclasses()
├── blip3oQwenForCausalLM.from_pretrained()  또는  blip3oLlamaForCausalLM.from_pretrained()
├── AutoProcessor.from_pretrained()
├── smart_tokenizer_and_embedding_resize()
├── model.get_model().initialize_vision_modules()
├── [LoRA] get_peft_model() 또는 PeftModel.from_pretrained()
├── make_supervised_data_module()
│   ├── CountingGenDataset()        ← data_type == "counting"
│   ├── OCRSyntheticGenDataset()    ← data_type == "ocr_synthetic"
│   └── LazySupervisedMixDataset()  ← data_type == "mix"
│       각각 내부에서:
│       ├── preprocess_multimodal()
│       ├── preprocess() → preprocess_qwen() / preprocess_llama3() / preprocess_plain()
│       └── img_process()
├── ValidationCallback()
│   ├── on_train_begin()    → 학습 가능 파라미터 목록 저장
│   ├── on_step_end()       → 체크포인트 저장 + 이해/생성 검증
│   │   ├── _save_checkpoint_to()
│   │   ├── _validate_understanding()
│   │   │   ├── _validate_counting_understanding()
│   │   │   └── _validate_ocr_understanding()
│   │   └── _validate_generation()
│   └── on_epoch_end()      → 에포크별 체크포인트 저장
├── blip3oTrainer()
│   └── trainer.train()
└── safe_save_model_for_hf_trainer()
    ├── get_mm_adapter_state_maybe_zero_3()
    └── maybe_zero_3()
```
