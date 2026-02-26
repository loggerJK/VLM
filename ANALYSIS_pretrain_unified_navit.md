# Bagel `pretrain_unified_navit.py` Training Logic 분석 보고서

## 1. 개요

**Bagel**은 ByteDance에서 개발한 unified multimodal model로, 하나의 LLM backbone (Qwen2 기반)을 사용하여 **image understanding**(VLM)과 **image generation**(flow-matching 기반)을 동시에 수행한다.

`train/pretrain_unified_navit.py`는 Bagel의 **메인 unified pretraining 스크립트**로, 다음을 통합한다:

- **Visual Understanding**: SigLIP ViT로 이미지 인코딩 → MLP connector → LLM → CE loss (텍스트 예측)
- **Visual Generation**: FLUX VAE로 continuous latent 인코딩 → flow-matching noising → LLM → llm2vae → MSE loss (velocity 예측)

NaViT(Native resolution Vision Transformer) 스타일의 variable-resolution 학습을 지원하며, sequence packing을 통해 다양한 해상도의 이미지와 텍스트를 하나의 packed sequence로 묶어 효율적으로 학습한다.

---

## 2. Arguments 구조

### 2.1 ModelArguments (`pretrain_unified_navit.py:98-172`)

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `model_path` | `hf/BAGEL-7B-MoT` | 사전학습된 Bagel 모델 경로 |
| `llm_path` | `hf/Qwen2.5-0.5B-Instruct/` | Qwen2 LLM 경로 |
| `llm_qk_norm` | `True` | Attention QK LayerNorm |
| `layer_module` | `Qwen2MoTDecoderLayer` | Decoder layer 클래스명 (MoT = Mixture of Transformers) |
| `vae_path` | `flux/vae/ae.safetensors` | FLUX VAE 체크포인트 경로 |
| `vit_path` | `hf/siglip-so400m-14-980-flash-attn2-navit/` | SigLIP ViT 경로 |
| `max_latent_size` | `32` | VAE latent grid 최대 크기 (패치 단위) |
| `latent_patch_size` | `2` | 각 latent patch가 커버하는 공간 크기 |
| `vit_patch_size` | `14` | ViT 패치 크기 (pixels) |
| `vit_max_num_patch_per_side` | `70` | ViT 한 변 최대 패치 수 |
| `connector_act` | `gelu_pytorch_tanh` | MLP connector 활성 함수 |
| `vit_select_layer` | `-2` | ViT hidden layer 선택 (끝에서 -2번째) |
| `text_cond_dropout_prob` | `0.1` | Text conditioning dropout 확률 |
| `vae_cond_dropout_prob` | `0.3` | VAE conditioning dropout 확률 |
| `vit_cond_dropout_prob` | `0.3` | ViT conditioning dropout 확률 |

### 2.2 DataArguments (`pretrain_unified_navit.py:176-208`)

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `dataset_config_file` | `data/configs/example.yaml` | Dataset YAML config 경로 |
| `max_num_tokens_per_sample` | `16384` | 단일 샘플 최대 토큰 수 (초과 시 skip) |
| `max_num_tokens` | `36864` | Packed batch 최대 토큰 수 (hard limit) |
| `expected_num_tokens` | — | `TrainingArguments`에 정의 (기본 32768), batch yield 기준 |
| `prefer_buffer_before` | `16384` | 이 길이 미만일 때 buffer에서 우선 pop |
| `max_buffer_size` | `50` | Overflow buffer 최대 크기 |

### 2.3 TrainingArguments (`pretrain_unified_navit.py:212-405`)

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `visual_gen` | `True` | Image generation 학습 활성화 |
| `visual_und` | `True` | Image understanding 학습 활성화 |
| `lr` | `1e-4` | Peak learning rate |
| `lr_scheduler` | `constant` | LR 스케줄러 (`constant` or `cosine`) |
| `warmup_steps` | `2000` | Linear warmup steps |
| `beta1/beta2` | `0.9/0.95` | AdamW 계수 |
| `eps` | `1e-15` | AdamW epsilon |
| `max_grad_norm` | `1.0` | Gradient clipping (L2 norm) |
| `ema` | `0.9999` | EMA decay rate |
| `mse_weight` | `1.0` | MSE loss (generation) 가중치 |
| `ce_weight` | `1.0` | CE loss (understanding) 가중치 |
| `ce_loss_reweighting` | `False` | CE loss token importance reweighting |
| `timestep_shift` | `1.0` | Flow-matching timestep shift |
| `gradient_accumulation_steps` | `1` | Gradient accumulation |
| `total_steps` | `500000` | 총 학습 step 수 |
| `save_every` | `2000` | Checkpoint 저장 주기 |
| `sharding_strategy` | `HYBRID_SHARD` | FSDP sharding 전략 |
| `num_shard` | `8` | FSDP shard 수 |
| `num_replicate` | `1` | Model replica 수 |
| `freeze_vae` | `True` | VAE 고정 (기본) |
| `freeze_llm` | `False` | LLM 고정 |
| `freeze_vit` | `False` | ViT 고정 |
| `copy_init_moe` | `True` | MoE expert 초기화 복제 |
| `use_flex` | `False` | FlexAttention 사용 여부 |

---

## 3. 모델 초기화

### 3.1 전체 구성 흐름 (`pretrain_unified_navit.py:472-578`)

```
1. Qwen2ForCausalLM (LLM) 로드/생성
   ├── from_pretrained(llm_path) 또는 Qwen2ForCausalLM(config) (finetune_from_hf)
   └── init_moe() — copy_init_moe=True일 때 MoE expert 복제 초기화
2. SiglipVisionModel (ViT) 로드/생성 — visual_und=True일 때
   └── vit_select_layer에 따라 num_hidden_layers 조정
3. FLUX AutoEncoder (VAE) 로드 — visual_gen=True일 때
4. Bagel wrapper 생성 (language_model + vit_model + config)
5. Tokenizer 설정 및 special token 추가
6. Freeze 설정 (VAE 기본 frozen, LLM/ViT 선택적)
7. EMA model = deepcopy(model)
8. Checkpoint 로드 (resume_from)
9. FSDP wrapping (model + ema_model)
10. Activation checkpointing 적용
```

### 3.2 Bagel 모델 아키텍처 (`bagel.py:57-94`)

```python
Bagel(PreTrainedModel)
├── language_model: Qwen2ForCausalLM      # LLM backbone (MoT variants 지원)
├── vit_model: SiglipVisionModel           # 이미지 이해용 ViT (visual_und)
├── connector: MLPconnector(vit_dim → llm_dim)  # ViT→LLM projection
├── vit_pos_embed: PositionEmbedding       # ViT 2D sincos positional embedding
├── time_embedder: TimestepEmbedder        # Flow timestep → hidden_size
├── vae2llm: Linear(patch_latent_dim → hidden_size)  # VAE latent → LLM space
├── llm2vae: Linear(hidden_size → patch_latent_dim)  # LLM output → velocity prediction
└── latent_pos_embed: PositionEmbedding    # Latent 2D sincos positional embedding
```

**핵심 차원 계산:**

```
latent_channel = vae_config.z_channels (=16 for FLUX)
patch_latent_dim = latent_patch_size^2 * latent_channel = 2^2 * 16 = 64
```

`llm2vae`는 zero-initialized (`bagel.py:98-99`), 학습 초기에 velocity prediction이 0에서 시작하도록 함.

### 3.3 MoT (Mixture of Transformers)

`layer_module` 필드로 decoder layer 타입을 지정:
- `Qwen2DecoderLayer`: 표준 decoder
- `Qwen2MoEDecoderLayer`: Mixture of Experts
- `Qwen2MoTDecoderLayer` (기본): Mixture of Transformers — understanding과 generation 토큰에 대해 서로 다른 FFN expert를 사용

MoT 사용 시 forward에서 `packed_und_token_indexes`와 `packed_gen_token_indexes`를 분리하여 전달 (`bagel.py:200-207`).

---

## 4. 데이터 파이프라인

### 4.1 PackedDataset 개요 (`dataset_base.py:45-475`)

`PackedDataset`은 `torch.utils.data.IterableDataset`을 상속하며, 여러 샘플을 하나의 긴 packed sequence로 묶는 **sequence packing** 전략을 사용한다.

**핵심 파라미터:**

| 파라미터 | 역할 |
|---------|------|
| `expected_num_tokens` (32768) | 이 길이에 도달하면 batch를 yield |
| `max_num_tokens` (36864) | 이 길이를 초과하면 buffer에 저장하고 현재 batch yield |
| `max_num_tokens_per_sample` (16384) | 이 길이를 초과하는 개별 샘플은 skip |
| `prefer_buffer_before` (16384) | 현재 batch가 이 길이 미만이면 buffer에서 우선 pop |
| `max_buffer_size` (50) | Overflow buffer 최대 크기 |

### 4.2 Packing 전략 (`dataset_base.py:238-304`)

```
while True:
    1. 현재 batch가 비어있으면 → mandatory 그룹에서 1개씩 반드시 포함
    2. curr < prefer_buffer_before이고 buffer 비어있지 않으면 → buffer에서 pop
    3. 그 외 → grouped_weights에 따라 확률적으로 그룹 선택 후 sampling
    4. 샘플이 max_num_tokens_per_sample 초과 → skip
    5. 현재 batch + 새 샘플이 max_num_tokens 초과 → buffer에 저장 or yield
    6. 아니면 pack_sequence()로 현재 batch에 추가
    7. curr >= expected_num_tokens이면 → yield
```

### 4.3 pack_sequence 상세 (`dataset_base.py:306-475`)

각 샘플의 `sequence_plan`에 따라 텍스트/ViT이미지/VAE이미지를 순서대로 packed sequence에 배치한다.

**세 가지 토큰 타입:**

| Type | 처리 방식 | Position ID | Attention |
|------|----------|-------------|-----------|
| `text` | BOS + text_ids + EOS 형태. `enable_cfg=1`이면 확률적으로 dropout | Causal (sequential RoPE) | `causal` |
| `vit_image` | `<startofimage>` + patchified pixels + `<endofimage>` | 모든 패치가 동일 position (단일 RoPE id) | `full` (bidirectional) |
| `vae_image` | `<startofimage>` + latent tokens + `<endofimage>` | 모든 latent이 동일 position | `noise` (generation) or `full` (conditioning) |

**Attention 모드:**
- `causal`: 표준 causal mask (하삼각)
- `full`: 해당 split 내 모든 토큰이 서로를 attend
- `noise`: full attention이지만 다른 split에서 noise tokens를 attend하지 못함 (generation 타겟의 noisy latent가 다른 샘플로 leak 방지)

**Timestep 샘플링:**
- `loss=1`인 VAE 이미지: `timestep = np.random.randn()` (정규분포에서 샘플, 이후 sigmoid 적용)
- `loss=0`인 VAE 이미지 (conditioning): `timestep = float('-inf')` → sigmoid 후 0이 됨 (clean image 사용)

### 4.4 Batch 필드 설명 (`dataset_base.py:187-236`)

`to_tensor()`가 반환하는 batch dict의 주요 필드:

| 필드 | 타입 | 설명 |
|------|------|------|
| `sequence_length` | int | 전체 packed sequence 길이 |
| `sample_lens` | List[int] | 각 샘플의 길이 (attention mask 구성용) |
| `packed_text_ids` | 1D LongTensor | Text 토큰 ID (BOS/EOS 포함) |
| `packed_text_indexes` | 1D LongTensor | Text 토큰의 packed sequence 내 위치 |
| `packed_position_ids` | 1D LongTensor | RoPE position IDs |
| `nested_attention_masks` | List[2D Tensor] | 샘플별 attention mask (0 = attend, -inf = ignore) |
| `padded_images` | Tensor [N, C, H, W] | VAE 입력 이미지 (zero-padded) |
| `patchified_vae_latent_shapes` | List[(h,w)] | 각 이미지의 patchified latent 크기 |
| `packed_latent_position_ids` | 1D LongTensor | Latent 2D position IDs |
| `packed_vae_token_indexes` | 1D LongTensor | VAE latent 토큰의 packed sequence 내 위치 |
| `packed_vit_tokens` | 2D Tensor | Patchified ViT 입력 토큰 |
| `packed_vit_position_ids` | 1D LongTensor | ViT 2D position IDs |
| `packed_vit_token_indexes` | 1D LongTensor | ViT 토큰의 packed sequence 내 위치 |
| `vit_token_seqlens` | 1D IntTensor | 각 이미지의 ViT 토큰 수 |
| `packed_timesteps` | 1D FloatTensor | Flow-matching timestep (per-token) |
| `mse_loss_indexes` | 1D LongTensor | MSE loss 계산 위치 |
| `packed_label_ids` | 1D LongTensor | CE loss 타겟 토큰 ID |
| `ce_loss_indexes` | 1D LongTensor | CE loss 계산 위치 |
| `ce_loss_weights` | 1D FloatTensor | CE loss per-token 가중치 |

### 4.5 DataLoader 구성 (`pretrain_unified_navit.py:642-650`)

```python
DataLoader(
    train_dataset,
    batch_size=1,           # PackedDataset이 이미 여러 샘플을 pack
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_wrapper(),  # SimpleCustomBatch로 래핑
    drop_last=True,
    prefetch_factor=2,
)
```

`batch_size=1`인 이유: `PackedDataset.__iter__()`가 이미 여러 샘플을 pack한 하나의 "mega-sample"을 yield하므로, DataLoader 레벨에서는 batch_size=1로 설정.

---

## 5. Forward 로직 (`bagel.py:101-229`)

### 5.1 전체 Forward 흐름

```
1. Text embedding: embed_tokens(packed_text_ids) → packed_sequence[text_indexes]에 배치
2. ViT processing (understanding):
   a. SigLIP ViT forward (flash attention + NaViT packed input)
   b. MLPconnector projection (vit_dim → llm_dim)
   c. + vit_pos_embed(2D sincos)
   d. → packed_sequence[vit_token_indexes]에 배치
3. VAE latent processing (generation):
   a. Patchify: padded_latent → (h*w, patch_latent_dim) per image
   b. Flow-matching noising: x_t = (1-t)*x_0 + t*noise
   c. vae2llm(x_t) + timestep_embed + latent_pos_embed
   d. → packed_sequence[vae_token_indexes]에 배치
4. LLM forward: Qwen2 transformer on packed_sequence
5. Loss 계산:
   a. MSE: llm2vae(hidden[mse_indexes]) vs velocity target (noise - x_0)
   b. CE: lm_head(hidden[ce_indexes]) vs label_ids
```

### 5.2 Understanding 분기 (`bagel.py:166-179`)

```python
# ViT forward (NaViT: variable-length packed attention)
packed_vit_token_embed = self.vit_model(
    packed_pixel_values=packed_vit_tokens,
    packed_flattened_position_ids=packed_vit_position_ids,
    cu_seqlens=cu_seqlens,    # cumulative sequence lengths for packed attention
    max_seqlen=max_seqlen,
)
# MLP projection + 2D sincos positional embedding
packed_vit_token_embed = self.connector(packed_vit_token_embed)
vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
packed_vit_token_embed = packed_vit_token_embed + vit_token_pos_emb
# 결과를 packed_sequence에 scatter
packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed
```

ViT 출력은 `cu_seqlens` 기반 packed attention으로, 이미지마다 다른 패치 수를 효율적으로 처리한다.

### 5.3 Generation 분기 (`bagel.py:181-197`)

```python
# 1. Latent patchify: VAE latent → (h*w, p*p*C)
for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
    latent = latent[:, :h*p, :w*p].reshape(C, h, p, w, p)
    latent = einsum("chpwq->hwpqc", latent).reshape(-1, p*p*C)

# 2. Flow-matching noising
packed_timesteps = sigmoid(packed_timesteps)  # randn → [0,1] via sigmoid
packed_timesteps = shift * t / (1 + (shift-1) * t)  # timestep shifting
x_t = (1-t)*x_0 + t*noise   # linear interpolation (flow matching)

# 3. LLM input = vae2llm(x_t) + timestep_embed + pos_embed
packed_latent = self.vae2llm(packed_latent) + self.time_embedder(t) + self.latent_pos_embed(pos)
packed_sequence[packed_vae_token_indexes] = packed_latent
```

**Timestep shift**: `t' = s*t / (1 + (s-1)*t)` where `s = timestep_shift`. `s > 1`일 때 더 많은 시간을 noise 쪽에 할당 (FLUX/Stable Diffusion 3 스타일).

### 5.4 Loss 계산 (`bagel.py:217-229`)

```python
# MSE loss (generation): velocity prediction
packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])
target = noise - packed_latent_clean  # v_t = dx_t/dt = x_1 - x_0 (noise → data 방향)
has_mse = packed_timesteps > 0        # timestep=0인 clean conditioning 이미지는 제외
mse = (packed_mse_preds - target[has_mse]) ** 2

# CE loss (understanding): next token prediction
packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")
```

---

## 6. Loss 계산 및 합산 (`pretrain_unified_navit.py:695-727`)

### 6.1 CE Loss 처리

```python
# per-token CE loss 계산 후, 모든 GPU에서 총 CE 토큰 수를 합산
total_ce_tokens = len(data['ce_loss_indexes'])  # local
dist.all_reduce(total_ce_tokens)                 # global

if ce_loss_reweighting:
    # ce_loss_weights는 len2weight(seq_len)으로 계산됨 (기본: 1/sqrt(seq_len))
    ce = (ce * ce_loss_weights).sum() * world_size / total_ce_loss_weights
else:
    ce = ce.sum() * world_size / total_ce_tokens  # 전역 평균

loss += ce * ce_weight
```

**ce_loss_weights** (`data_utils.py:168-177`):

```python
def len2weight(x, loss_reduction='square'):
    # 'token': 1, 'sample': 1/x, 'square': 1/sqrt(x)
    return 1 / (x ** 0.5)  # 기본값: 긴 시퀀스의 토큰에 낮은 가중치
```

### 6.2 MSE Loss 처리

```python
total_mse_tokens = len(data['mse_loss_indexes'])
dist.all_reduce(total_mse_tokens)

mse = mse.mean(dim=-1).sum() * world_size / total_mse_tokens  # per-token 차원 평균 후 전역 합산/평균
loss += mse * mse_weight
```

### 6.3 최종 Loss

```python
loss = ce * ce_weight + mse * mse_weight
loss = loss / gradient_accumulation_steps
loss.backward()
```

---

## 7. 최적화 (`pretrain_unified_navit.py:581-734`)

### 7.1 Optimizer

```python
optimizer = AdamW(
    fsdp_model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    eps=1e-15,
    weight_decay=0         # weight decay 없음
)
```

### 7.2 LR Scheduler

| 타입 | 설명 |
|------|------|
| `constant` | Linear warmup (2000 steps) → constant LR |
| `cosine` | Linear warmup → cosine decay to min_lr (1e-7) |

### 7.3 Gradient Clipping & Optimizer Step

```python
if (micro_step + 1) % gradient_accumulation_steps == 0:
    total_norm = fsdp_model.clip_grad_norm_(max_grad_norm)  # L2 norm clip at 1.0
    optimizer.step()
    scheduler.step()
    fsdp_ema_update(ema_model, fsdp_model, decay=0.9999)
    optimizer.zero_grad()
```

### 7.4 EMA Update (`fsdp_utils.py:255-269`)

```python
@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    # FSDP-aware EMA: flat_param 단위로 in-place 업데이트
    ema_params = [handle.flat_param.data for handle in ema_handles if requires_grad]
    new_params = [handle.flat_param.data for handle in new_handles if requires_grad]

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
    # → ema = decay * ema + (1-decay) * new
```

FSDP의 `flat_param`에 직접 접근하여 EMA를 수행. `_foreach_mul_`/`_foreach_add_`로 벡터화된 in-place 연산.

---

## 8. FSDP 분산 학습

### 8.1 FSDP 설정 (`fsdp_utils.py:48-83`)

```python
FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy(
        transformer_layer_cls={
            Qwen2DecoderLayer, Qwen2MoEDecoderLayer, Qwen2MoTDecoderLayer,
            SiglipEncoderLayer, SiglipVisionTransformer,
            MLPconnector, TimestepEmbedder, PositionEmbedding,
        }
    ),
    mixed_precision=MixedPrecision(bf16 for param/reduce/buffer),
    sharding_strategy=HYBRID_SHARD,       # 기본: node 간 replicate, node 내 shard
    backward_prefetch=BACKWARD_PRE,
    cpu_offload=CPUOffload(False),
    device_mesh=(num_replicate, num_shard),  # HYBRID_SHARD 시 2D mesh
)
```

**Sharding 전략:**
- `HYBRID_SHARD` (기본): `device_mesh = (num_replicate, num_shard)`. 한 shard 그룹(예: 8 GPU) 내에서 파라미터 분할, 그룹 간에는 복제.
- `FULL_SHARD`: 전체 GPU에 걸쳐 파라미터 분할.

**Auto-wrap 대상**: 각 Transformer layer, SigLIP encoder layer, MLP connector, Timestep embedder, Position embedding이 개별 FSDP unit이 됨.

### 8.2 Activation Checkpointing (`pretrain_unified_navit.py:567-573`)

```python
apply_activation_checkpointing(
    fsdp_model,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    check_fn=grad_checkpoint_check_fn  # Qwen2*DecoderLayer, SiglipEncoderLayer, MLPconnector
)
```

`NO_REENTRANT` 체크포인팅을 Qwen2 decoder layers, SigLIP encoder layers, MLP connector에 적용.

### 8.3 Checkpoint Save/Load

**Save** (`fsdp_utils.py:86-150`):
1. EMA model → `ema.safetensors` (FULL_STATE_DICT, rank0 only)
2. Model → `model.safetensors` (FULL_STATE_DICT, rank0 only)
3. Optimizer → `optimizer.{shard:05d}-of-{total:05d}.pt` (LOCAL_STATE_DICT, per-shard)
4. Scheduler → `scheduler.pt` (rank0 only)
5. Data status → `data_status.pt` (rank0 only, `gather_object`로 모든 rank에서 수집)

**Load** (`fsdp_utils.py:152-233`):
- Model: safetensors 로드, `latent_pos_embed.pos_embed`와 `vit_pos_embed.pos_embed`는 pop (sincos 고정이므로 해상도 변경 대응)
- Optimizer: shard 인덱스에 맞는 파일 로드 (LOCAL_STATE_DICT)
- Train step: checkpoint 디렉터리명에서 추출 (`int(basename) + 1`)
- Data status: rank별로 분배하여 데이터 로딩 재개

---

## 9. MFU 계산 (`pretrain_unified_navit.py:46-65, 748-774`)

### 9.1 FLOP 추정 (Qwen2 기준)

```python
def qwen2_flop_coefficients(config):
    # MLP FLOPs per token
    mlp_N = hidden_size * intermediate_size * 3
    # Attention linear FLOPs per token
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_heads * head_dim)
    # Embedding + LM head FLOPs per token
    emd_and_lm_head_N = vocab_size * hidden_size * 2

    dense_N = (mlp_N + attn_linear_N) * num_layers + emd_and_lm_head_N
    dense_token_factor = 6 * dense_N     # forward + backward = 3x, × 2 for multiply-accumulate
    attn_factor = 12 * head_dim * num_heads * num_layers  # QK^T + softmax*V, seq-length dependent

    return dense_token_factor, attn_factor
```

### 9.2 MFU 계산

```python
# 실제 FLOPs = dense_part + attention_part (sequence-length dependent)
flops_all_token = dense_token_factor * token_window + attn_factor * seqlen_square_window
actual_tflops = flops_all_token / elapsed / 1e12
peak_total_tflops = peak_device_tflops * world_size
mfu = actual_tflops / peak_total_tflops
```

- `token_window`: 로깅 주기 내 총 토큰 수 (모든 GPU 합산)
- `seqlen_square_window`: 각 샘플 길이의 제곱합 (attention FLOPs 추정용)
- `peak_device_tflops`: GPU별 BF16 peak TFLOPs (자동 감지: H100=989, A100=312 등)

---

## 10. Janus/Lumina-DiMOO와의 핵심 차이점

| 항목 | Bagel | Janus / Lumina-DiMOO |
|------|-------|---------------------|
| **Image tokenization** | Continuous latent (FLUX VAE, 16ch) | Discrete tokens (VQ-VAE) |
| **Generation 방식** | Flow-matching (continuous, velocity prediction) | Masked diffusion / MaskGit (discrete, token prediction) |
| **Generation loss** | MSE (v_t = noise - x_0) | Cross-entropy (masked token prediction) |
| **Timestep** | Continuous t ∈ [0,1], sigmoid(randn) | Discrete mask ratio schedule |
| **LLM backbone** | Qwen2 + MoT (Mixture of Transformers) | LLaDA |
| **ViT** | SigLIP (NaViT, variable resolution) | 없음 (VQ-VAE가 image encoder 역할) |
| **Understanding** | ViT → MLP → LLM | VQ tokens → LLM |
| **Sequence packing** | PackedDataset (variable-length, 다양한 attention mode) | Fixed padding / variable-length batch |
| **Attention 모드** | causal / full / noise (per-split) | causal only |
| **CFG dropout** | Text/ViT/VAE 각각 독립적 dropout prob | Single unconditional masking |
| **EMA** | FSDP-aware EMA (flat_param 직접 업데이트) | 없음 |
| **Sharding** | FSDP HYBRID_SHARD (2D device mesh) | FSDP FULL_SHARD or DDP |
| **Position embedding** | 2D sincos (latent: max 32×32, ViT: max 70×70) | 1D (with newline tokens for 2D structure) |
| **VAE** | FLUX AutoEncoder (continuous, encode+decode) | VQ-VAE (discrete codebook) |
| **Decoder** | VAE decode (continuous latent → image) | VQ-VAE decode (discrete tokens → image) |

### 10.1 Flow-Matching vs Masked Diffusion

**Bagel (Flow-matching)**:
```
x_t = (1-t) * x_0 + t * noise        # noising
v_t = noise - x_0                      # target velocity (data → noise 방향)
loss = MSE(predicted_v, v_t)           # velocity prediction loss
inference: x_{t-dt} = x_t - v_t * dt  # Euler integration (noise → data)
```

**Lumina-DiMOO (Masked diffusion)**:
```
x_masked = mask_tokens(x_0, mask_ratio)  # 토큰 마스킹
loss = CE(predicted_tokens, x_0)           # 마스크된 토큰 예측
inference: iterative unmask (MaskGit)      # 고확신 토큰부터 순차 복원
```

### 10.2 MoT (Mixture of Transformers) 특징

Bagel의 `Qwen2MoTDecoderLayer`는 understanding 토큰과 generation 토큰에 대해 **서로 다른 FFN pathway**를 사용한다. 이를 통해 modality 간 간섭을 줄이면서 하나의 attention layer를 공유한다. Forward 시 `packed_und_token_indexes`와 `packed_gen_token_indexes`로 토큰을 분리하여 각각의 expert FFN에 라우팅한다.

---

## 11. 가변 해상도(Variable Resolution) 지원 메커니즘 상세 분석

Bagel은 Understanding과 Generation에서 **서로 다른 방식**으로 가변 해상도를 처리한다. 두 경로 모두 `ImageTransform` + `MaxLongEdgeMinShortEdgeResize`로 이미지를 리사이즈하지만, 이후 처리 파이프라인이 근본적으로 다르다.

### 11.1 Understanding: NaViT 방식 (Padding-Free)

#### 11.1.1 이미지 리사이즈

Understanding 경로는 `ImageTransform(980, 224, 14)`로 설정된다 (`app.py:72`).

```python
# app.py:72
vit_transform = ImageTransform(980, 224, 14)
```

`MaxLongEdgeMinShortEdgeResize` (`transforms.py:15-87`)의 리사이즈 로직:

```python
# transforms.py:60-87
def forward(self, img, img_num=1):
    scale = min(self.max_size / max(width, height), 1.0)   # 긴 변을 max_size(980) 이하로
    scale = max(scale, self.min_size / min(width, height))  # 짧은 변을 min_size(224) 이상으로
    new_width, new_height = self._apply_scale(width, height, scale)
    # _make_divisible: stride(14)의 배수로 반올림
    # → 결과 크기는 항상 14의 배수 (예: 224×336, 420×560, 784×980 등)
```

**핵심**: 출력 크기가 `stride=14`의 배수로 보장되므로 ViT patch_size=14로 나누어떨어진다. 그러나 이미지마다 크기가 다르므로 **패치 수도 이미지마다 다르다**.

- 예: 224×336 → 16×24 = 384 패치, 784×980 → 56×70 = 3920 패치

#### 11.1.2 Patchify: Conv2d → Linear 변환

`SiglipVisionEmbeddings` (`siglip_navit.py:145-195`)에서 ViT의 patch embedding은 원래 `nn.Conv2d`이지만, NaViT에서는 `nn.Linear`로 변환된다:

```python
# siglip_navit.py:153-159
self.patch_embedding = nn.Conv2d(
    in_channels=config.num_channels,
    out_channels=self.embed_dim,
    kernel_size=self.patch_size,  # 14
    stride=self.patch_size,
    padding="valid",
)

# siglip_navit.py:167-182 — Conv2d → Linear 변환
def convert_conv2d_to_linear(self, config, meta=False):
    linear_patch_embedding = nn.Linear(
        config.num_channels * self.patch_size ** 2,  # 3 * 14^2 = 588
        self.embed_dim, bias=True
    )
    W = self.patch_embedding.weight.permute(0, 2, 3, 1).reshape(
        self.embed_dim, config.num_channels * self.patch_size ** 2
    )
    linear_patch_embedding.weight.data = W
    linear_patch_embedding.bias.data = self.patch_embedding.bias.data
    del self.patch_embedding
    self.patch_embedding = linear_patch_embedding
```

이 변환이 필요한 이유: Conv2d는 고정 크기 이미지 (H×W tensor)를 입력으로 받지만, NaViT는 **이미 patchify된 1D sequence** (packed tokens)를 입력으로 받기 때문. 데이터 파이프라인에서 `patchify()` (`data_utils.py:43-50`)가 먼저 수행된다:

```python
# data_utils.py:43-50
def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)  # (num_patches, 588) — Linear 입력과 동일 차원
    return image
```

#### 11.1.3 Packed Attention: cu_seqlens + flash_attn_varlen_func

데이터 파이프라인 (`dataset_base.py:353-395`)에서 여러 이미지의 patchified 토큰을 하나의 1D sequence로 concatenate하고, 각 이미지의 토큰 수를 `vit_token_seqlens`로 기록한다:

```python
# dataset_base.py:366-373
vit_tokens = patchify(image_tensor, self.data_config.vit_patch_size)
num_img_tokens = vit_tokens.shape[0]  # 이미지마다 다름!
sequence_status['packed_vit_tokens'].append(vit_tokens)
sequence_status['vit_token_seqlens'].append(num_img_tokens)
```

Forward 시 (`bagel.py:166-179`), 이 정보로 `cu_seqlens`(cumulative sequence lengths)를 구성하여 `flash_attn_varlen_func`에 전달한다:

```python
# bagel.py:167-175
cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
cu_seqlens = cu_seqlens.to(torch.int32)
max_seqlen = torch.max(vit_token_seqlens).item()
packed_vit_token_embed = self.vit_model(
    packed_pixel_values=packed_vit_tokens,      # [total_patches, 588]
    packed_flattened_position_ids=packed_vit_position_ids,
    cu_seqlens=cu_seqlens,    # [0, n1, n1+n2, n1+n2+n3, ...]
    max_seqlen=max_seqlen,
)
```

ViT 내부 attention (`siglip_navit.py:232-241`):

```python
# siglip_navit.py:232-241
attn_output = flash_attn_varlen_func(
    query_states.to(torch.bfloat16),
    key_states.to(torch.bfloat16),
    value_states.to(torch.bfloat16),
    cu_seqlens_q=cu_seqlens,   # 이미지 경계 정보
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    causal=False,              # 이미지 내 bidirectional attention
)
```

#### 11.1.4 전체 흐름 다이어그램

```
이미지 A (800×600)     이미지 B (300×400)
       │                      │
  ImageTransform          ImageTransform
  (980, 224, 14)          (980, 224, 14)
       │                      │
  798×602 → 리사이즈     → 308×406 → 리사이즈
  → 798÷14=57패치/행       → 308÷14=22패치/행
  → 602÷14=43패치/열       → 406÷14=29패치/열
       │                      │
   patchify()             patchify()
  [2451, 588]             [638, 588]
       │                      │
       └──── concatenate ─────┘
                  │
        [3089, 588]  (packed)
        cu_seqlens = [0, 2451, 3089]
                  │
        Linear (588 → hidden_size)
                  │
        flash_attn_varlen_func
        (각 이미지 내에서만 attention)
                  │
        [3089, hidden_size]  (packed output)
                  │
        MLP connector + pos_embed
                  │
        packed_sequence[vit_indexes] = output
```

**핵심 장점**: ViT 내부에서 **padding이 전혀 없다**. 모든 연산이 실제 패치에 대해서만 수행되므로 연산 낭비가 없다.

---

### 11.2 Generation: Zero-padding + Crop 방식

#### 11.2.1 이미지 리사이즈

Generation 경로는 `ImageTransform(1024, 512, 16)`으로 설정된다 (`app.py:71`):

```python
# app.py:71
vae_transform = ImageTransform(1024, 512, 16)
```

- `max_size=1024`: 긴 변 최대 1024
- `min_size=512`: 짧은 변 최소 512
- `stride=16`: VAE downsample factor의 배수

#### 11.2.2 Batch 내 Zero-padding (`dataset_base.py:205-211`)

학습 시, batch 내 여러 이미지의 크기가 다르므로 **batch 내 최대 크기로 zero-padding**한다:

```python
# dataset_base.py:205-211
image_tensors = sequence_status.pop('vae_image_tensors')
image_sizes = [item.shape for item in image_tensors]
max_image_size = [max(item) for item in list(zip(*image_sizes))]
padded_images = torch.zeros(size=(len(image_tensors), *max_image_size))
for i, image_tensor in enumerate(image_tensors):
    padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor
```

예를 들어, batch에 512×1024와 768×768 이미지가 있으면:
- `padded_images` shape: `[2, 3, 768, 1024]`
- 첫 번째 이미지: `[:, :512, :1024]`에만 값, 나머지 zero
- 두 번째 이미지: `[:, :768, :768]`에만 값, 나머지 zero

#### 11.2.3 VAE Encoding: Padding 영역도 처리 (연산 낭비)

VAE (FLUX AutoEncoder)는 **CNN 기반**이므로 padded_images를 통째로 encode한다. CNN은 입력 텐서의 모든 공간 위치에 대해 convolution을 수행하므로, zero-padded 영역에 대한 연산도 수행된다.

```python
# bagel.py (forward_cache_update_vae):512
padded_latent = vae_model.encode(padded_images)
# padded_latent shape: [N, 16, H//8, W//8] — padding 영역의 latent도 포함
```

#### 11.2.4 Crop: 실제 크기만 추출 (`bagel.py:184-187`)

VAE encoding 후, forward에서 **padding된 영역의 latent를 제거**하고 실제 이미지 크기만 crop한다:

```python
# bagel.py:182-188
p = self.latent_patch_size  # 2
packed_latent = []
for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
    latent = latent[:, :h * p, :w * p]  # ← 여기서 crop! 실제 크기만 남김
    latent = latent.reshape(self.latent_channel, h, p, w, p)
    latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
    packed_latent.append(latent)
packed_latent_clean = torch.cat(packed_latent, dim=0)
```

`patchified_vae_latent_shapes`는 데이터 파이프라인에서 계산된 각 이미지의 실제 latent grid 크기:

```python
# dataset_base.py:419-422
H, W = image_tensor.shape[1:]
h = H // self.data_config.vae_image_downsample  # vae_image_downsample=16
w = W // self.data_config.vae_image_downsample
sequence_status['vae_latent_shapes'].append((h, w))
```

여기서 `vae_image_downsample=16`은 `VAE downsample(8) × latent_patch_size(2)`로, VAE의 8x 공간 축소와 이후 2x2 patchify를 합친 것.

#### 11.2.5 LLM으로의 Scatter

Crop된 latent 토큰들은 `packed_vae_token_indexes`를 통해 LLM packed sequence의 정확한 위치에 scatter된다:

```python
# bagel.py:197
packed_sequence[packed_vae_token_indexes] = packed_latent
```

이 단계 이후에는 Understanding과 동일하게 **packing 기반**으로 효율적으로 처리된다.

#### 11.2.6 전체 흐름 다이어그램

```
이미지 A (512×1024)    이미지 B (768×768)
       │                      │
  ImageTransform          ImageTransform
  (1024, 512, 16)         (1024, 512, 16)
       │                      │
  [3, 512, 1024]          [3, 768, 768]
       │                      │
       └──── zero-pad to max ─┘
                  │
        padded_images: [2, 3, 768, 1024]
        (이미지 A의 512~768행은 모두 0)
                  │
           VAE encode (CNN)
        (padding 영역도 연산)
                  │
        padded_latent: [2, 16, 96, 128]
                  │
        ┌─── crop per image ──┐
        │                     │
  A: [:, :32, :64]      B: [:, :48, :48]
  → patchify(2×2)       → patchify(2×2)
  [16×32, 64]           [24×24, 64]
  = [512, 64]           = [576, 64]
        │                     │
        └──── concatenate ────┘
                  │
         [1088, 64] (packed)
                  │
         vae2llm + timestep + pos_embed
                  │
         packed_sequence[vae_indexes] = result
```

---

### 11.3 비교 표

| 항목 | Understanding (NaViT) | Generation (VAE + Crop) |
|------|----------------------|------------------------|
| **ImageTransform 설정** | `(980, 224, 14)` | `(1024, 512, 16)` |
| **Visual Encoder** | SigLIP ViT (Transformer) | FLUX VAE (CNN) |
| **Patchify 시점** | 데이터 파이프라인에서 (ViT 전) | VAE encode 후 forward에서 |
| **Padding** | 없음 (각 이미지 독립 patchify) | Batch 내 최대 크기로 zero-pad |
| **Encoder 입력** | Packed 1D sequence (모든 패치 concatenate) | Padded 4D tensor (batch) |
| **가변 길이 처리** | `cu_seqlens` + `flash_attn_varlen_func` | Zero-pad → encode → crop |
| **연산 낭비** | 없음 | VAE가 padding 영역도 encode |
| **Attention 메커니즘** | `flash_attn_varlen_func` (각 이미지 내 bidirectional) | N/A (CNN이므로 attention 없음) |
| **Position Encoding** | 2D flattened position IDs (extrapolate/interpolate) | 2D sincos position embedding |
| **LLM으로의 전달** | MLP connector → scatter to packed_sequence | vae2llm → scatter to packed_sequence |
| **LLM 단계 효율성** | Packing으로 효율적 | Packing으로 효율적 (동일) |

**요약**: Understanding 경로는 ViT가 Transformer이므로 NaViT 스타일의 **완전 padding-free** 처리가 가능하다. 반면 Generation 경로는 VAE가 CNN이므로 **batch 레벨 padding이 불가피**하지만, VAE encoding 직후 crop하여 이후 LLM 단계에서는 packing으로 효율적으로 처리한다.

---

### 11.4 Inference에서의 차이

#### 11.4.1 Understanding (Inference)

Inference 시에도 학습과 동일한 파이프라인을 사용한다 (`inferencer.py:81-90`):

```python
# inferencer.py:81-90
if vit:
    generation_input, kv_lens, ropes = self.model.prepare_vit_images(
        curr_kvlens=kv_lens,
        curr_rope=ropes,
        images=[image],
        transforms=self.vit_transform,    # ImageTransform(980, 224, 14)
        new_token_ids=self.new_token_ids,
    )
    past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)
```

`prepare_vit_images`에서 `vit_transform`으로 리사이즈 → patchify → cu_seqlens 구성까지 동일하게 수행. Inference는 batch_size=1이 일반적이므로 padding 문제가 원래 없다.

#### 11.4.2 Generation — Text-to-Image (Inference)

T2I inference 시에는 사용자가 **고정 크기를 직접 지정**한다 (`app.py:169-178`):

```python
# app.py:169-178
if image_ratio == "1:1":
    image_shapes = (1024, 1024)
elif image_ratio == "4:3":
    image_shapes = (768, 1024)
elif image_ratio == "3:4":
    image_shapes = (1024, 768)
elif image_ratio == "16:9":
    image_shapes = (576, 1024)
elif image_ratio == "9:16":
    image_shapes = (1024, 576)
```

이 경우 입력 이미지가 없으므로 VAE encoding이 필요 없고, 순수 noise에서 시작하여 flow-matching으로 생성한다. **Zero-padding 문제가 발생하지 않는다.**

#### 11.4.3 Generation — Image Editing (Inference)

Image editing 시에는 입력 이미지의 크기가 그대로 출력 크기를 결정한다 (`inferencer.py:249-252`):

```python
# inferencer.py:249-252
input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)
image_shapes = input_term.size[::-1]  # PIL (W, H) → (H, W)로 변환
```

입력 이미지를 `vae_transform.resize_transform`으로 리사이즈하면 stride=16의 배수 크기가 되고, 이 크기가 생성될 이미지의 해상도로 사용된다. Inference에서는 batch_size=1이므로 padding이 필요 없다.

#### 11.4.4 Inference 요약

| 시나리오 | 가변 해상도 | Zero-padding |
|---------|-----------|-------------|
| **Understanding** | `vit_transform`으로 리사이즈 (이미지마다 다른 패치 수) | 불필요 (NaViT) |
| **T2I** | 사용자가 고정 크기 지정 (1024x1024 등) | 불필요 (noise에서 시작) |
| **Image Edit** | 입력 이미지 크기가 출력 크기 결정 | 불필요 (batch_size=1) |

**결론**: Inference 시에는 batch_size=1이 일반적이므로 Generation의 zero-padding 문제가 사실상 발생하지 않는다. Zero-padding에 의한 연산 낭비는 **학습 시에만** 존재하는 문제이며, 이는 CNN 기반 VAE의 구조적 한계에 기인한다.
