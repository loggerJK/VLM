# BLIP3o Inference Analysis Report

## 1. 개요

BLIP3o는 **autoregressive 언어 모델(Qwen2.5-VL)과 diffusion 기반 이미지 생성**을 하나의 통합 프레임워크에서 결합한 멀티모달 모델이다. 핵심 아이디어는 다음과 같다:

- **CLIP feature 공간에서의 Diffusion**: 기존의 text-to-image 모델들이 VAE latent 또는 pixel 공간에서 diffusion을 수행하는 것과 달리, BLIP3o는 **의미적으로 풍부한 EVA-CLIP 이미지 feature 공간**에서 diffusion을 수행한다.
- **2-Stage 생성 파이프라인**:
  - **Stage 1**: LLM이 텍스트를 처리하고, Learnable Latent Query가 LLM hidden state에 attend → DiT(Diffusion Transformer)가 Flow Matching으로 CLIP feature를 denoising
  - **Stage 2**: Diffusion Decoder(UNet + VAE)가 CLIP feature를 pixel 이미지로 변환

```
Text Prompt → [Qwen2.5-VL LLM] → Hidden States → [DiT Flow Matching] → CLIP Features
                                                                              ↓
                                            Output Image ← [VAE] ← [UNet] ← ┘
```

---

## 2. 모델 아키텍처 (`architecture.py` 기반)

`architecture.py`에서 덤프된 전체 모델 구조는 `blip3oQwenForInferenceLM`이며, 크게 4개의 주요 컴포넌트로 구성된다:

### 2.1 전체 구조도

```
blip3oQwenForInferenceLM
├── visual (Qwen2_5_VisionTransformerPretrainedModel)    ← 이미지 이해용 ViT
│   ├── patch_embed: Conv3d(3, 1280, kernel=(2,14,14))
│   ├── rotary_pos_emb
│   ├── blocks: 32 × Qwen2_5_VLVisionBlock (dim=1280)
│   └── merger: MLP(5120 → 3584)                        ← ViT → LLM 차원 변환
│
├── model (blip3oQwenModel)
│   ├── embed_tokens: Embedding(151668, 3584)            ← 토큰 임베딩
│   ├── layers: 28 × Qwen2_5_VLDecoderLayer             ← Qwen2.5 LLM
│   │   ├── self_attn (GQA: q=3584, k=v=512)
│   │   ├── mlp (3584 → 18944 → 3584, SiLU)
│   │   └── RMSNorm
│   ├── norm: RMSNorm(3584)
│   ├── rotary_emb
│   │
│   ├── gen_vision_tower (EvaClipVisionTower)            ← 생성용 EVA-CLIP (학습 시에만 사용)
│   │
│   ├── latent_queries: Parameter(1, n_query, 3584)      ← 학습 가능한 latent query
│   │
│   └── dit (NextDiTCrossAttn)                           ← Diffusion Transformer
│       └── model (LuminaNextDiT2DModel)
│           ├── caption_projection: MLP(3584 → 1792)
│           ├── patch_embedder: Linear(1792 → 1792)
│           ├── time_caption_embed: Timestep + Caption embedding
│           ├── layers: 24 × LuminaNextDiTBlock
│           │   ├── attn1 (Self-Attention, dim=1792, 28 heads)
│           │   ├── attn2 (Cross-Attention, dim=1792, 28 heads)
│           │   ├── feed_forward (1792 → 4864 → 1792)
│           │   └── LuminaRMSNormZero (adaptive norm)
│           └── norm_out (LuminaLayerNormContinuous)
│
└── lm_head: Linear(3584 → 151668)                      ← 텍스트 생성용 head
```

### 2.2 주요 차원 요약

| 컴포넌트 | 차원 | 비고 |
|-----------|------|------|
| Qwen2.5 ViT | 1280 → 3584 (merger 후) | 32 blocks, patch 14×14 |
| Qwen2.5 LLM | 3584 | 28 layers, GQA (q=3584, kv=512) |
| EVA-CLIP E-14-plus | 1792 | 64 layers, image 448×448, patch 14 |
| DiT (LuminaNextDiT2D) | 1792 | 24 layers, 28 heads |
| Latent Queries | n_query × 3584 | n_query=64 (default) |
| Vocabulary | 151,668 tokens | Qwen2.5 tokenizer |

---

## 3. 인퍼런스 파이프라인 단계별 분석

### 전체 흐름 다이어그램

```
[inference.py]                          [pipeline_llava_gen.py]
     │                                          │
     ▼                                          │
add_template()                                  │
     │                                          │
     ▼                                          │
pipe(prompt, guidance_scale=3.0)  ──────────►  __call__()
                                                │
                                                ▼
                                     _prepare_and_encode_inputs()
                                                │
  ┌─────────────────────────────────────────────┤
  │  Stage 1: BLIP3o 모델 내부                  │
  │  ┌──────────────────────────┐               │
  │  │  generate_image()        │               │
  │  │    ├── Tokenize + Embed  │               │
  │  │    ├── Latent Query 결합 │               │
  │  │    ├── LLM Forward (28L) │               │
  │  │    └── sample_images()   │               │
  │  │        (DiT FlowMatch    │               │
  │  │         30 steps, CFG)   │               │
  │  └──────────┬───────────────┘               │
  │             ▼                                │
  │   CLIP Features [1, 64, 1792]               │
  └─────────────┬───────────────────────────────┘
                │
  ┌─────────────┤
  │  Stage 2: UNet + VAE 파이프라인              │
  │  ┌──────────────────────────┐               │
  │  │  SDXL-style UNet         │               │
  │  │    ├── time_ids 설정     │               │
  │  │    ├── text_embeds 풀링  │               │
  │  │    └── Denoising Loop    │               │
  │  │        (EulerDiscrete    │               │
  │  │         50 steps, CFG)   │               │
  │  │             │            │               │
  │  │             ▼            │               │
  │  │  VAE Decode              │               │
  │  │    latents → pixels      │               │
  │  └──────────┬───────────────┘               │
  │             ▼                                │
  │        PIL Image (1024×1024)                 │
  └──────────────────────────────────────────────┘
```

---

### Step 1: 프롬프트 포맷팅 (Qwen Conversation Template)

**소스**: `inference.py:89-94`, `blip3o/conversation.py:442-451`

```python
def add_template(prompt):
    conv = conv_templates['qwen'].copy()
    conv.append_message(conv.roles[0], prompt[0])   # user 메시지
    conv.append_message(conv.roles[1], None)         # assistant (비어있음)
    prompt = conv.get_prompt()
    return [prompt]
```

Qwen ChatML 템플릿 (`conv_qwen`)이 적용된다:

```python
conv_qwen = Conversation(
    system="<|im_start|>system\nYou are a helpful assistant.",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
)
```

**입력 예시**: `"Please generate image based on the following caption: A photo of cute cat"`

**포맷팅 결과**:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Please generate image based on the following caption: A photo of cute cat<|im_end|>
<|im_start|>assistant
```

이 포맷은 ChatML 표준을 따르며, assistant 턴이 비어있어 모델이 이미지 생성을 수행하게 된다.

---

### Step 2: 토크나이징 + 텍스트 임베딩

**소스**: `blip3o/model/language_model/blip3o_qwen_inference.py:62-71`

```python
N_QUERY = self.get_n_query()                          # 64
inputs = tokenizer(text, padding="longest", return_tensors="pt")
input_ids = inputs.input_ids.to(device)               # [1, seq_len]

# 생성 시작 토큰 추가 (151665 = <|endoftext|> 또는 generation trigger)
input_ids = torch.cat([input_ids, torch.tensor([[151665]]).to(device)], dim=1)
                                                       # [1, seq_len + 1]

# 토큰 임베딩 변환
text_embeds = self.get_model().embed_tokens(input_ids) # [1, seq_len + 1, 3584]
```

**핵심 토큰 상수** (`blip3o/constants.py`):

| 상수 | 값 | 용도 |
|------|-----|------|
| `IMAGE_TOKEN_IDX` | 151667 | 생성 이미지 토큰 위치 (학습 시) |
| `UND_IMAGE_TOKEN_IDX` | 151655 | 이해 이미지 토큰 위치 (`<|image_pad|>`) |
| 151665 | — | 생성 트리거 토큰 (인퍼런스 시 append) |

---

### Step 3: Latent Query 결합

**소스**: `blip3o/model/language_model/blip3o_qwen_inference.py:72-83`

```python
# Learnable latent queries를 batch 크기에 맞게 복제
latent_queries = self.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)
                                                       # [1, 64, 3584]

# (이미지 이해 모드인 경우: pixel_values → ViT → und_image_embeds 삽입)
# 텍스트 임베딩 뒤에 latent queries를 concat
text_embeds = torch.cat([text_embeds, latent_queries], dim=1)
                                                       # [1, seq_len + 1 + 64, 3584]

# Attention mask도 동일하게 확장
attention_mask = torch.cat([attention_mask, torch.ones_like(latent_queries[:, :, 0])], dim=1)
                                                       # [1, seq_len + 1 + 64]
```

**Latent Query의 역할**: 학습 가능한 파라미터 `nn.Parameter(torch.randn(1, n_query, hidden_size))`로 초기화되며, LLM의 causal attention을 통해 앞에 오는 텍스트 토큰들의 의미를 집약하는 "요약 토큰" 역할을 한다. DiT에 전달할 conditioning 정보를 만들어내는 **bridge** 역할이다.

**시퀀스 구조**:
```
[system tokens] [user prompt tokens] [151665] [latent_q_1] [latent_q_2] ... [latent_q_64]
←────────────── text_embeds ────────────────→ ←──── latent_queries (64개) ────→
```

---

### Step 4: LLM Forward Pass → Hidden States 추출

**소스**: `blip3o/model/language_model/blip3o_qwen_inference.py:86-97`

```python
outputs = self.model(
    inputs_embeds=text_embeds,          # [1, seq_len+65, 3584]
    attention_mask=attention_mask,       # [1, seq_len+65]
    output_hidden_states=True,
    return_dict=True,
)

# 마지막 레이어의 hidden states에서 latent query 위치만 추출
hidden_states = outputs.hidden_states[-1][:, -N_QUERY:, :]
                                                       # [1, 64, 3584]
img_hidden_states = hidden_states
```

이 단계에서 Qwen2.5-VL의 28개 decoder layer를 모두 통과한다. **Causal attention** 덕분에 각 latent query는 자신 앞에 있는 모든 텍스트 토큰과 이전 latent query의 정보에 attend할 수 있다. 결과적으로 `img_hidden_states`는 텍스트 프롬프트의 의미를 응축한 64개의 conditioning 벡터가 된다.

**메모리 참고**: input_ids가 아닌 `inputs_embeds`로 전달되므로 embedding lookup은 한 번만 수행되고, LLM은 전체 시퀀스에 대해 single forward pass를 수행한다.

---

### Step 5: DiT Denoising Loop (Flow Matching + CFG)

**소스**: `blip3o/model/language_model/blip3o_qwen_inference.py:99-157`

이 단계가 BLIP3o의 핵심으로, **CLIP feature 공간에서 diffusion**을 수행한다.

#### 5.1 Classifier-Free Guidance (CFG) 설정

```python
# Unconditional (null) conditioning: 0 벡터
img_hidden_states_null = torch.zeros_like(img_hidden_states)     # [1, 64, 3584]

# Conditional + Unconditional 결합 (batch 차원으로)
img_hidden_states_input = torch.cat([img_hidden_states_null, img_hidden_states], 0)
                                                                  # [2, 64, 3584]
```

#### 5.2 Latent 초기화

```python
latent_size = self.get_model().dit.config.input_size      # 8
latent_channels = self.get_model().dit.config.in_channels  # 1792

latents = randn_tensor(
    shape=(batch_size, latent_channels, latent_size, latent_size),
    ...
)                                                          # [1, 1792, 8, 8]
```

초기 latents는 **순수 Gaussian noise**이며, 8×8 spatial grid에 1792 channel을 가진다. 1792는 EVA-CLIP E-14-plus의 hidden dimension과 일치한다.

#### 5.3 Scheduler 설정 (Flow Matching)

```python
sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
# sigmas = [1.0, 0.9667, 0.9333, ..., 0.0333]  (30 steps)
scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)
```

**FlowMatchEulerDiscreteScheduler**: Optimal Transport conditional flow matching을 사용하며, sigma가 1.0(pure noise)에서 시작하여 ~0(clean signal)으로 선형 감소한다.

#### 5.4 Denoising Loop (30 steps)

```python
for t in scheduler.timesteps:
    # CFG를 위해 latents를 2배로 복제
    latent_model_input = latents.repeat(2, 1, 1, 1)      # [2, 1792, 8, 8]

    # DiT forward: noisy latents + timestep + LLM hidden states
    noise_pred = self.get_model().dit(
        x=latent_model_input,                              # [2, 1792, 8, 8]
        timestep=t.unsqueeze(0).expand(2),                 # [2]
        z_latents=img_hidden_states_input,                 # [2, 64, 3584]
    )                                                      # [2, 1792, 8, 8]

    # CFG 적용
    noise_pred_uncond, noise_pred = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
                                                           # [1, 1792, 8, 8]

    # Flow Matching step: x_t → x_{t-1}
    latents = scheduler.step(noise_pred, t, latents).prev_sample
                                                           # [1, 1792, 8, 8]
```

**CFG 수식**: `ε_guided = ε_uncond + s × (ε_cond − ε_uncond)` (기본 `s=3.0`)

#### 5.5 DiT 내부 동작 상세

**소스**: `blip3o/model/nextdit_crossattn.py:86-95`, `blip3o/model/lumina_nextdit2d.py:289-366`

```python
# NextDiTCrossAttn.forward()
def forward(self, x, timestep, z_latents):
    model_pred = self.model(
        hidden_states=x,                    # noisy CLIP latents [2, 1792, 8, 8]
        timestep=timestep,                  # diffusion timestep [2]
        encoder_hidden_states=z_latents,    # LLM hidden states [2, 64, 3584]
        encoder_mask=ones(2, 64),           # 모든 토큰 attend
        image_rotary_emb=self.freqs_cis,    # 2D RoPE
    ).sample
    return model_pred
```

**LuminaNextDiT2DModel.forward() 내부**:

1. **Patch Embedding**: `[2, 1792, 8, 8]` → `[2, 64, 1792]` (8×8 = 64 patches, patch_size=1이므로 reshape만)
2. **Caption Projection**: LLM hidden states `[2, 64, 3584]` → `[2, 64, 1792]` (MLP: 3584 → 1792)
3. **Timestep + Caption Embedding**: timestep과 caption의 mean을 결합하여 `temb` 생성 (dim=1024)
4. **24개 LuminaNextDiTBlock**: 각 block에서:
   - **Self-Attention**: patch들 간의 attention (2D RoPE 적용)
   - **Cross-Attention**: patch hidden states가 LLM hidden states(projected)에 attend
   - **Gated Fusion**: `gate.tanh()` 로 self-attn과 cross-attn 출력 혼합
   - **FFN**: SwiGLU 변형 (linear_1 × silu(linear_3) → linear_2)
   - **Adaptive LayerNorm**: `LuminaRMSNormZero`가 temb로 scale/gate 파라미터 생성
5. **Output Norm + Unpatchify**: `[2, 64, 1792]` → `[2, 1792, 8, 8]`

#### 5.6 DiT Block 상세 (LuminaNextDiTBlock)

```
                    ┌─────────────────────────────────────┐
                    │         LuminaNextDiTBlock           │
                    │                                     │
    hidden_states ──┤  LuminaRMSNormZero(temb)            │
                    │     → gate_msa, scale_mlp, gate_mlp │
                    │                                     │
                    │  Self-Attention (attn1)              │
                    │     Q, K, V ← norm_hidden_states    │
                    │     + 2D RoPE on Q, K               │
                    │     → self_attn_output               │
                    │                                     │
                    │  Cross-Attention (attn2)             │
                    │     Q ← norm_hidden_states           │
                    │     K, V ← RMSNorm(encoder_hidden)  │
                    │     + 2D RoPE on Q only              │
                    │     → cross_attn_output              │
                    │                                     │
                    │  Gated Mixing:                       │
                    │     mixed = self_attn + gate.tanh()  │
                    │              × cross_attn            │
                    │     out_proj(mixed)                  │
                    │                                     │
                    │  Residual + gate_msa.tanh() × Norm  │
                    │                                     │
                    │  FFN: RMSNorm × (1 + scale_mlp)     │
                    │       → SwiGLU FFN                   │
                    │                                     │
                    │  Residual + gate_mlp.tanh() × Norm  │
                    └─────────────────────────────────────┘
```

---

### Step 6: CLIP Features → Pixel Image (Diffusion Decoder)

**소스**: `pipeline_llava_gen.py`, `blip3o/model/language_model/blip3o_qwen_inference.py:94-97`, `inference.py:53-66`

이 구조는 **unCLIP (DALL-E 2)** 와 유사하게, CLIP 임베딩을 중간 표현으로 사용하여 pixel 이미지를 생성하는 2단계 접근 방식이다.

#### 6.1 DiT 출력 reshape

**소스**: `blip3o/model/language_model/blip3o_qwen_inference.py:95-97`

```python
# DiT에서 나온 denoised CLIP features
output_img = latents                                    # [1, 1792, 8, 8]
output_img = output_img.view(1, 1792, -1)               # [1, 1792, 64]
output_img = output_img.permute(0, 2, 1).contiguous()   # [1, 64, 1792]
return output_img
```

8×8 spatial grid가 64개의 1792차원 CLIP feature token 시퀀스로 변환된다. 이것이 `generate_image()`의 반환값이며, `_prepare_and_encode_inputs()`를 통해 UNet 파이프라인으로 전달된다.

#### 6.2 `EmuVisualGenerationPipeline` 클래스 구조

**소스**: `pipeline_llava_gen.py:49-81`

```python
class EmuVisualGenerationPipeline(DiffusionPipeline):
    def __init__(self, tokenizer, multimodal_encoder, scheduler, unet, vae,
                 feature_extractor, safety_checker, eva_size=448, ...):
        # 등록 컴포넌트:
        #   multimodal_encoder = BLIP3o 모델 (blip3oQwenForInferenceLM)
        #   unet = UNet2DConditionModel (SDXL 기반)
        #   vae = AutoencoderKL
        #   scheduler = EulerDiscreteScheduler
        #   tokenizer = Qwen2.5 tokenizer

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)  # = 8
        self.transform = TF.Compose([
            TF.Resize((448, 448), interpolation=BICUBIC),
            TF.ToTensor(),
            TF.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
        ])
        self.negative_prompt = {}  # CFG용 negative prompt 캐시
```

| 컴포넌트 | 타입 | 역할 |
|-----------|------|------|
| `multimodal_encoder` | `blip3oQwenForInferenceLM` | Stage 1 전체 (LLM + DiT) |
| `unet` | `UNet2DConditionModel` | SDXL 스타일 pixel-space denoising |
| `vae` | `AutoencoderKL` | Latent → pixel 디코딩 |
| `scheduler` | `EulerDiscreteScheduler` | UNet용 noise scheduler |
| `tokenizer` | `AutoTokenizer` | Qwen2.5 토크나이저 |
| `feature_extractor` | `CLIPImageProcessor` | Safety checker용 (미사용) |

**로딩** (`inference.py:53-62`):
```python
pipe = DiffusionPipeline.from_pretrained(
    diffusion_path,                          # "{model_path}/diffusion-decoder"
    custom_pipeline="pipeline_llava_gen",     # 커스텀 파이프라인
    torch_dtype=torch.bfloat16,
    multimodal_encoder=multi_model,          # BLIP3o 모델 (LLM + DiT)
    tokenizer=tokenizer,
    safety_checker=None,
)
pipe.vae.to('cuda:0')
pipe.unet.to('cuda:0')
```

#### 6.3 `_prepare_and_encode_inputs()` — 3가지 입력 모드

**소스**: `pipeline_llava_gen.py:182-252`

이 함수는 입력 타입(텍스트, 이미지, 또는 둘 다)에 따라 분기하여 CLIP features를 생성한다.

**Mode A: Text Only** (일반 text-to-image 생성)
```python
# 텍스트만 있는 경우
prompt = self.multimodal_encoder.generate_image(
    text=[text_prompt], tokenizer=self.tokenizer
)                                                       # [1, 64, 1792]

# CFG용 negative prompt: 공백 문자 " "로 생성
if do_classifier_free_guidance:
    negative = self.multimodal_encoder.generate_image(
        text=[" "], tokenizer=self.tokenizer
    )                                                   # [1, 64, 1792]
    prompt = torch.cat([prompt, negative], dim=0)       # [2, 64, 1792]
```

**Mode B: Image Only** (CLIP feature 직접 인코딩)
```python
# 이미지만 있는 경우: EVA-CLIP으로 직접 인코딩
prompt = self.multimodal_encoder.model.encode_image(image=image_prompt)
                                                        # [1, N, 1792]

# CFG용: zero 이미지로 인코딩
if do_classifier_free_guidance:
    negative_image = torch.zeros_like(image_prompt)
    negative = self.multimodal_encoder.model.encode_image(image=negative_image)
    prompt = torch.cat([prompt, negative], dim=0)       # [2, N, 1792]
```

**Mode C: Text + Image** (이미지 reconstruction)
```python
# 이미지를 Qwen2.5-VL image_processor로 전처리
resized_images = x.resize((448, 448))
image_inputs = image_processor(resized_images, return_tensors="pt")
# image_inputs.pixel_values, image_inputs.image_grid_thw

# 텍스트에서 <image> → <|vision_start|><|image_pad|>*256<|vision_end|> 치환
text_prompt = text_prompt.replace(
    "<image>",
    "<|vision_start|>" + "<|image_pad|>" * 256 + "<|vision_end|>"
)

# LLM이 이미지와 텍스트를 함께 처리
prompt = self.multimodal_encoder.generate_image(
    text=[text_prompt],
    pixel_values=image_prompt.cuda(),
    image_grid_thw=image_grid_thw.cuda(),
    tokenizer=self.tokenizer
)                                                       # [1, 64, 1792]
```

**Negative prompt 캐싱**: `self.negative_prompt` 딕셔너리에 한 번 생성한 negative prompt를 캐싱하여 반복 호출 시 재계산을 방지한다. Text/Text+Image 모드에서는 `" "`(공백), Image 모드에서는 `zeros`가 사용된다.

#### 6.4 SDXL 스타일 UNet Conditioning

**소스**: `pipeline_llava_gen.py:121-127`

UNet은 SDXL 아키텍처를 따르며, 3가지 conditioning 입력을 받는다:

```python
# 1. time_ids: SDXL 해상도 조건
time_ids = torch.LongTensor(
    [original_h, original_w, crop_top, crop_left, target_h, target_w]
)
# 기본값: [1024, 1024, 0, 0, 1024, 1024]

# 2. text_embeds: CLIP features의 pooled embedding (SDXL의 pooled text embedding 역할)
text_embeds = torch.mean(prompt_embeds, dim=1)          # [2, 1792]
# 64개 CLIP feature token의 평균 → 1개의 pooled vector

# 3. encoder_hidden_states: CLIP features 전체 시퀀스 (cross-attention용)
# = prompt_embeds                                       # [2, 64, 1792]

unet_added_conditions = {
    "time_ids": torch.cat([time_ids, time_ids], dim=0), # [2, 6] (CFG 포함)
    "text_embeds": text_embeds,                          # [2, 1792]
}
```

**SDXL과의 대응 관계**:

| SDXL 원본 | BLIP3o 파이프라인 | 차원 |
|-----------|------------------|------|
| CLIP-L text embeddings (cross-attn) | DiT가 생성한 CLIP features | `[2, 64, 1792]` |
| CLIP-G pooled text embeddings | CLIP features의 mean pooling | `[2, 1792]` |
| time_ids (해상도 조건) | 동일 (1024×1024 고정) | `[2, 6]` |

#### 6.5 UNet Denoising Loop (50 steps)

**소스**: `pipeline_llava_gen.py:129-162`

```python
# Scheduler 설정
self.scheduler.set_timesteps(num_inference_steps=50, device=device)

# 초기 latents: pixel-space Gaussian noise
shape = (batch_size, self.unet.config.in_channels,
         height // self.vae_scale_factor,
         width // self.vae_scale_factor)
# = (1, 4, 1024//8, 1024//8) = (1, 4, 128, 128)
latents = torch.randn(shape, device=device, dtype=dtype)
latents = latents * self.scheduler.init_noise_sigma

# 50-step denoising
for t in timesteps:
    # CFG를 위해 latents 2배
    latent_model_input = torch.cat([latents] * 2)       # [2, 4, 128, 128]
    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

    # UNet forward
    noise_pred = self.unet(
        latent_model_input,                              # [2, 4, 128, 128]
        t,                                               # scalar timestep
        encoder_hidden_states=prompt_embeds,             # [2, 64, 1792]
        added_cond_kwargs=unet_added_conditions,         # time_ids + text_embeds
    ).sample                                             # [2, 4, 128, 128]

    # CFG 적용 (주의: cond가 먼저, uncond가 나중 — DiT와 순서 반대)
    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                                                         # [1, 4, 128, 128]

    # Euler step: x_t → x_{t-1}
    latents = self.scheduler.step(noise_pred, t, latents).prev_sample
```

**DiT와의 주요 차이점**:

| 속성 | Stage 1 (DiT) | Stage 2 (UNet) |
|------|--------------|----------------|
| Scheduler | `FlowMatchEulerDiscreteScheduler` | `EulerDiscreteScheduler` |
| Steps | 30 | 50 |
| Latent 공간 | CLIP feature (1792-dim) | Pixel-space VAE latent (4-dim) |
| Latent shape | `[1, 1792, 8, 8]` | `[1, 4, 128, 128]` |
| Conditioning | LLM hidden states `[B, 64, 3584]` | CLIP features `[B, 64, 1792]` |
| CFG null | zeros | `" "` 프롬프트로 생성한 CLIP features |
| CFG 순서 | `[uncond, cond]` → chunk → `uncond + s*(cond-uncond)` | `[cond, uncond]` → chunk → `uncond + s*(cond-uncond)` |

#### 6.6 VAE Decoding

**소스**: `pipeline_llava_gen.py:254-259`

```python
def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
    latents = 1 / self.vae.config.scaling_factor * latents  # scaling 역변환
    image = self.vae.decode(latents).sample                  # [1, 3, 1024, 1024]
    image = (image / 2 + 0.5).clamp(0, 1)                   # [-1,1] → [0,1]
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()  # NCHW → NHWC
    return image
```

최종 변환: `numpy_to_pil()` → `(images * 255).round().astype("uint8")` → `Image.fromarray()` → `PIL.Image`

**Shape 흐름**:
```
UNet output     [1, 4, 128, 128]    VAE latent
    ↓ / scaling_factor
VAE decode      [1, 3, 1024, 1024]  RGB pixels (float, [-1,1])
    ↓ (x/2 + 0.5)
normalize       [1, 3, 1024, 1024]  RGB pixels (float, [0,1])
    ↓ permute + numpy
numpy           [1, 1024, 1024, 3]  HWC format
    ↓ * 255
PIL Image       (1024, 1024)        uint8 RGB
```

---

### 2중 CFG (Classifier-Free Guidance) 구조

BLIP3o의 인퍼런스에서는 **두 단계에 걸쳐 독립적으로 CFG가 적용**된다:

```
┌─ Stage 1: DiT CFG ──────────────────────────────────────────────┐
│  소스: blip3o_qwen_inference.py:115-116, 148-150                │
│                                                                  │
│  Conditional:   LLM hidden states [1, 64, 3584]                  │
│  Unconditional: zeros             [1, 64, 3584]                  │
│  결합: [uncond, cond] → [2, 64, 3584]                           │
│                                                                  │
│  CFG: ε_uncond + 3.0 × (ε_cond − ε_uncond)                     │
│  결과: denoised CLIP features [1, 1792, 8, 8]                   │
└──────────────────────────────────────────────────────────────────┘
                            ↓
                   CLIP features [1, 64, 1792]
                            ↓
┌─ Stage 2: UNet CFG ─────────────────────────────────────────────┐
│  소스: pipeline_llava_gen.py:224-236, 156-159                    │
│                                                                  │
│  Conditional:   generate_image(text_prompt) 결과 [1, 64, 1792]  │
│  Unconditional: generate_image(" ") 결과         [1, 64, 1792]  │
│  결합: [cond, uncond] → [2, 64, 1792]                           │
│                                                                  │
│  CFG: ε_uncond + 3.0 × (ε_cond − ε_uncond)                     │
│  결과: denoised pixel latents [1, 4, 128, 128]                  │
└──────────────────────────────────────────────────────────────────┘
```

**핵심 차이점**:
- **DiT CFG**: null conditioning이 **zero 벡터**. LLM hidden states 공간에서 작동.
- **UNet CFG**: null conditioning이 **공백 문자 `" "`를 Stage 1 전체(LLM → DiT)를 통과시킨 CLIP features**. CLIP feature 공간에서 작동.
- 두 CFG의 `guidance_scale`은 동일하게 3.0이지만, 독립적으로 조절 가능한 별개의 하이퍼파라미터이다.
- **CFG 결합 순서 차이**: DiT는 `[uncond, cond]` 순서로 결합하고, UNet은 `[cond, uncond]` 순서로 결합한다. 각각 `chunk(2)` 후 적절히 처리된다.

---

### Image Reconstruction 모드

**소스**: `inference.py:132-144`, `pipeline_llava_gen.py:237-251`

텍스트와 이미지를 함께 입력하여 이미지를 reconstruction하는 모드가 지원된다:

```python
# inference.py — Image Reconstruction 사용법
inputs = add_template(["<image>\nPlease reconstruct the given image."])
inputs.append(Image.open("your_image.jpg"))  # PIL Image 추가
gen_img = pipe(inputs, guidance_scale=3.0)
```

**처리 흐름**:

1. `pipe.__call__()` 에서 `inputs` 리스트를 순회:
   - `str` → `has_text = True`, `text_prompt`에 추가
   - `PIL.Image` → `has_image = True`, Qwen2.5-VL image_processor로 전처리 (448×448 resize)
2. 텍스트 내 `<image>` 토큰을 `<|vision_start|><|image_pad|>×256<|vision_end|>`로 치환
3. `_prepare_and_encode_inputs()` → Mode C (Text + Image) 분기:
   - `generate_image(text, pixel_values, image_grid_thw, tokenizer)` 호출
   - `generate_image()` 내부에서 `pixel_values` → Qwen2.5-VL ViT로 인코딩 → `<|image_pad|>` 위치에 삽입
   - LLM이 이미지 토큰 + 텍스트를 함께 처리 → latent queries → DiT → CLIP features
4. 이후 Stage 2 (UNet + VAE)는 text-to-image와 동일

**입력 시퀀스 구조** (Image Reconstruction):
```
[system tokens] [<|vision_start|>] [img_embed×256] [<|vision_end|>] [\n] [Please...] [151665] [latent_q×64]
←──── text embeddings (이미지 임베딩 삽입 포함) ──────────────────────────→ ←── queries ──→
```

---

## 4. 핵심 컴포넌트 상세

### 4.1 DiT (NextDiTCrossAttn / LuminaNextDiT2DModel)

**소스**: `blip3o/model/nextdit_crossattn.py`, `blip3o/model/lumina_nextdit2d.py`

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `input_size` | 8 | Spatial grid 크기 (8×8) |
| `patch_size` | 1 | Patch 크기 (패칭 없음) |
| `in_channels` | 1792 | 입력 채널 수 = EVA-CLIP feature dim |
| `dim` (hidden_size) | 1792 | 내부 hidden dimension |
| `n_layers` | 24 | Transformer block 수 |
| `n_heads` | 28 | Attention head 수 |
| `n_kv_heads` | 28 | KV head 수 (GQA 없음, full MHA) |
| `latent_embedding_size` | 3584 | Cross-attention input dim (= LLM hidden) |
| `learn_sigma` | False | Sigma 학습 안 함 |
| `qk_norm` | True | QK normalization 적용 |

**2D RoPE**: `get_2d_rotary_pos_embed_lumina(dim=64, h=384, w=384)`로 사전 계산된 positional embedding을 self-attention에 적용한다. `dim // n_heads = 1792 // 28 = 64`이 head dimension이다.

**구조적 특징**:
- Self-attention과 Cross-attention이 **한 block 안에 모두 존재** (Joint block이 아닌 Sequential)
- Cross-attention의 key/value에는 RoPE가 적용되지 않음 (LLM hidden states는 spatial structure가 아님)
- Learnable **gate parameter** (`nn.Parameter(torch.zeros([n_heads]))`)로 cross-attention 기여도를 제어
- **Adaptive LayerNorm** (LuminaRMSNormZero): timestep embedding으로 scale/gate 파라미터를 동적 생성

### 4.2 Latent Queries

**소스**: `blip3o/model/blip3o_arch.py:33`

```python
self.latent_queries = nn.Parameter(torch.randn(1, config.n_query, config.hidden_size))
# Shape: [1, 64, 3584]
```

- **초기화**: 표준 정규분포 `torch.randn`으로 random 초기화
- **학습 방식**: 전체 모델 학습 과정에서 gradient로 업데이트됨
- **역할**: LLM 시퀀스의 끝에 concat되어 causal attention을 통해 텍스트 정보를 응축
  - 첫 번째 latent query: 모든 텍스트 토큰 + generation trigger 토큰에 attend
  - 마지막 latent query: 텍스트 + 이전 63개 latent query 모두에 attend
- **output**: LLM의 마지막 hidden layer에서 추출된 `[1, 64, 3584]` 벡터가 DiT의 cross-attention conditioning이 됨

### 4.3 EVA-CLIP Vision Tower (Generation용)

**소스**: `blip3o/model/multimodal_encoder/eva_clip/eva_clip_encoder.py`

**설정** (`eva-clip-E-14-plus.json`):

| 파라미터 | 값 |
|----------|-----|
| `image_size` | 448 |
| `patch_size` | 14 |
| `width` (hidden dim) | 1792 |
| `layers` | 64 |
| `head_width` | 112 |
| `mlp_ratio` | 8.571 |
| Patch 수 | (448/14)² = 1024 |

**인퍼런스 시 역할**: EVA-CLIP tower는 **학습 시에만** 사용되어 target 이미지를 CLIP feature로 인코딩한다. DiT는 이 CLIP feature를 reconstruction target으로 학습한다. **인퍼런스 시에는 EVA-CLIP tower를 사용하지 않는다** — DiT가 직접 CLIP feature 공간에서 noise를 denoise하여 새로운 feature를 생성한다.

**학습 시 pooling**: 1024 patches (32×32) → `early_pool2d_4` → 64 patches (8×8), 이는 DiT의 `input_size=8`과 정확히 일치한다.

### 4.4 FlowMatchEulerDiscreteScheduler

**소스**: `blip3o/model/language_model/blip3o_qwen_inference.py:59`, `blip3o/model/multimodal_encoder/builder.py:61`

- **Origin**: `Alpha-VLLM/Lumina-Next-SFT-diffusers` 에서 로드
- **방식**: Optimal Transport Conditional Flow Matching
- **Sigma schedule**: 선형 `np.linspace(1.0, 1/30, 30)` → `[1.0, 0.967, ..., 0.033]`
- **Step 수**: 기본 30 steps
- **수식**: Flow Matching은 noise에서 data로의 직선 경로를 학습:
  - Forward: `x_t = (1-t) × x_0 + t × ε` (t ∈ [0, 1])
  - 모델은 velocity field `v_θ(x_t, t)`를 예측
  - Euler step: `x_{t-Δt} = x_t - Δt × v_θ(x_t, t)`

### 4.5 Qwen2.5-VL Vision Transformer (이해용)

**소스**: `architecture.py:2-31`

| 파라미터 | 값 |
|----------|-----|
| Patch embedding | Conv3d(3, 1280, kernel=(2,14,14)) |
| Blocks | 32 × Qwen2_5_VLVisionBlock (dim=1280) |
| Attention | SDPA, qkv: 1280→3840 |
| MLP | 1280 → 3420 → 1280 (SiLU) |
| Merger | MLP(5120 → 3584) |

**Conv3d patch embed**: temporal dimension이 2인 3D convolution으로, video input도 처리 가능하도록 설계되었다. 이미지의 경우 temporal=1로 동작.

**Merger**: 2×2 spatial merge를 수행하여 `4 × 1280 = 5120` → `3584`로 변환. 이를 통해 ViT 출력을 LLM hidden dimension에 맞춘다.

인퍼런스 시 텍스트 전용 생성에서는 이 ViT가 사용되지 않지만, **이미지 이해**(VQA 등)나 **이미지 reconstruction**(image-to-image) 모드에서는 입력 이미지를 인코딩하는 데 사용된다.

---

## 5. 텐서 Shape 흐름

### 5.1 Text-to-Image 생성 (전체 shape 추적)

가정: `prompt = "A photo of cute cat"`, `n_query = 64`, `batch_size = 1`
포맷팅 후 토큰 수를 `T`라 하자 (약 25-35 tokens).

```
Stage                              Tensor                  Shape              비고
─────────────────────────────────────────────────────────────────────────────────────
[1] Prompt formatting
    raw prompt                     string                  —                  텍스트
    formatted prompt               string                  —                  ChatML 형식

[2] Tokenizing
    input_ids                      LongTensor              [1, T]             Qwen tokenizer
    + gen trigger token            LongTensor              [1, T+1]           151665 append
    text_embeds                    FloatTensor             [1, T+1, 3584]     embed_tokens

[3] Latent Query 결합
    latent_queries                 Parameter               [1, 64, 3584]      학습 파라미터
    combined_embeds                FloatTensor             [1, T+65, 3584]    concat
    attention_mask                 BoolTensor              [1, T+65]          all ones

[4] LLM Forward (Qwen2.5-VL, 28 layers)
    all_hidden_states[-1]          FloatTensor             [1, T+65, 3584]    last layer
    img_hidden_states              FloatTensor             [1, 64, 3584]      last 64 positions

[5] DiT Denoising (Flow Matching, 30 steps)
    ┌── CFG setup ──
    │  img_hidden_states_null      FloatTensor             [1, 64, 3584]      zeros
    │  img_hidden_states_input     FloatTensor             [2, 64, 3584]      cat(null, cond)
    │
    ├── Initial noise ──
    │  latents                     FloatTensor             [1, 1792, 8, 8]    randn
    │
    ├── Per step (×30) ──
    │  latent_model_input          FloatTensor             [2, 1792, 8, 8]    repeat for CFG
    │  │
    │  ├── DiT Internal ──
    │  │  patch_embed              FloatTensor             [2, 64, 1792]      reshape
    │  │  caption_proj             FloatTensor             [2, 64, 1792]      MLP(3584→1792)
    │  │  temb                     FloatTensor             [2, 1024]          timestep+caption
    │  │  ×24 blocks:
    │  │    self_attn              FloatTensor             [2, 64, 28, 64]    28 heads × 64 dim
    │  │    cross_attn             FloatTensor             [2, 64, 28, 64]    attend to caption
    │  │    ffn                    FloatTensor             [2, 64, 1792]      SwiGLU
    │  │  norm_out                 FloatTensor             [2, 64, 1792]      final norm
    │  │  unpatchify               FloatTensor             [2, 1792, 8, 8]    reshape back
    │  │
    │  noise_pred                  FloatTensor             [2, 1792, 8, 8]    model output
    │  noise_pred (after CFG)      FloatTensor             [1, 1792, 8, 8]    guided
    │  latents (updated)           FloatTensor             [1, 1792, 8, 8]    scheduler step
    └──

    denoised latents               FloatTensor             [1, 1792, 8, 8]    clean CLIP features

[5→6] Reshape
    output_img                     FloatTensor             [1, 1792, 64]      view
    output_img                     FloatTensor             [1, 64, 1792]      permute

[6] Diffusion Decoder (EmuVisualGenerationPipeline)
    ┌── CFG setup (UNet) ──
    │  prompt_embeds (cond)         FloatTensor             [1, 64, 1792]      from Stage 1
    │  prompt_embeds (uncond)       FloatTensor             [1, 64, 1792]      from " " prompt
    │  prompt_embeds (combined)     FloatTensor             [2, 64, 1792]      cat(cond, uncond)
    │
    ├── UNet Conditioning ──
    │  time_ids                     LongTensor              [2, 6]             [1024,1024,0,0,1024,1024]
    │  text_embeds                  FloatTensor             [2, 1792]          mean(prompt_embeds, dim=1)
    │
    ├── Initial noise ──
    │  latents                      FloatTensor             [1, 4, 128, 128]   randn × init_noise_sigma
    │
    ├── Per step (×50, EulerDiscrete) ──
    │  latent_model_input           FloatTensor             [2, 4, 128, 128]   cat for CFG
    │  noise_pred                   FloatTensor             [2, 4, 128, 128]   UNet output
    │  noise_pred (after CFG)       FloatTensor             [1, 4, 128, 128]   guided
    │  latents (updated)            FloatTensor             [1, 4, 128, 128]   scheduler step
    └──

    denoised pixel latents          FloatTensor             [1, 4, 128, 128]   VAE latent space

[7] VAE Decode
    / scaling_factor               FloatTensor             [1, 4, 128, 128]   역정규화
    vae.decode()                   FloatTensor             [1, 3, 1024, 1024] RGB pixels [-1,1]
    (x/2 + 0.5).clamp(0,1)        FloatTensor             [1, 3, 1024, 1024] RGB pixels [0,1]
    numpy + *255                   ndarray                 [1, 1024, 1024, 3] uint8 HWC
    final output                   PIL.Image               (1024, 1024)       output image
```

### 5.2 DiT 내부 Shape 변환 상세

```
Input:  hidden_states = [2, 1792, 8, 8]  (NCHW)

┌─ patch_embedder ─────────────────────────────────────────┐
│  Linear(1792 → 1792)                                     │
│  [2, 1792, 8, 8] → reshape → [2, 64, 1792]              │
│  (patch_size=1이므로 spatial reshape만 수행)                │
└──────────────────────────────────────────────────────────┘

┌─ caption_projection ─────────────────────────────────────┐
│  encoder_hidden_states: [2, 64, 3584]                    │
│  PixArtAlphaTextProjection:                              │
│    Linear(3584 → 1792) → GELU → Linear(1792 → 1792)     │
│  → [2, 64, 1792]                                         │
└──────────────────────────────────────────────────────────┘

┌─ time_caption_embed ─────────────────────────────────────┐
│  timestep: [2]                                           │
│  TimestepEmbedding: Timesteps → Linear(256→1024)         │
│                     → SiLU → Linear(1024→1024)           │
│  CaptionEmbedding: mean([2,64,1792]) → [2,1792]         │
│                    → LayerNorm → Linear(1792→1024)       │
│  temb = timestep_emb + caption_emb → [2, 1024]          │
└──────────────────────────────────────────────────────────┘

┌─ LuminaNextDiTBlock (×24) ───────────────────────────────┐
│                                                          │
│  norm1 (LuminaRMSNormZero):                              │
│    temb [2,1024] → Linear(1024→7168)                     │
│    → split into: gate_msa, scale_mlp, gate_mlp + norm_h  │
│                   [2,1792] each                          │
│                                                          │
│  Self-Attn (attn1):                                      │
│    Q,K,V ← norm_hidden [2,64,1792]                       │
│    heads=28, head_dim=64                                 │
│    Q,K: [2,28,64,64] + 2D RoPE                          │
│    Attn: softmax(QK^T/√64) × V → [2,28,64,64]          │
│    → reshape → [2,64,28,64]                              │
│                                                          │
│  Cross-Attn (attn2):                                     │
│    Q ← norm_hidden [2,64,1792]                           │
│    K,V ← RMSNorm(encoder_hidden) [2,64,1792]            │
│    Q: [2,28,64,64] + 2D RoPE                            │
│    K: [2,28,64,64] (no RoPE)                             │
│    Attn: softmax(QK^T/√64) × V → [2,28,64,64]          │
│    × gate.tanh() → gated cross-attention                 │
│                                                          │
│  Mix: self_attn + cross_attn → flatten → out_proj        │
│  Residual: h = h + gate_msa.tanh() × Norm(mixed)        │
│                                                          │
│  FFN: SwiGLU                                             │
│    RMSNorm(h) × (1 + scale_mlp) → [2,64,1792]           │
│    linear_1(x) × silu(linear_3(x)) → [2,64,4864]        │
│    linear_2 → [2,64,1792]                                │
│  Residual: h = h + gate_mlp.tanh() × RMSNorm(ffn_out)   │
│                                                          │
│  Output: [2, 64, 1792]                                   │
└──────────────────────────────────────────────────────────┘

┌─ norm_out (LuminaLayerNormContinuous) ───────────────────┐
│  temb [2,1024] → SiLU → Linear(1024→1792) → scale,shift │
│  LayerNorm(h) × (1 + scale) + shift                     │
│  → Linear(1792→1792) → [2, 64, 1792]                    │
└──────────────────────────────────────────────────────────┘

┌─ Unpatchify ─────────────────────────────────────────────┐
│  [2, 64, 1792] → view [2, 8, 8, 1, 1, 1792]            │
│  → permute [2, 1792, 8, 1, 8, 1]                        │
│  → flatten → [2, 1792, 8, 8]                             │
└──────────────────────────────────────────────────────────┘
```

---

## 6. 주요 하이퍼파라미터

### 6.1 인퍼런스 하이퍼파라미터

#### Stage 1 (DiT — CLIP feature 생성)

| 파라미터 | 기본값 | 위치 | 설명 |
|----------|--------|------|------|
| `guidance_scale` | 3.0 | `blip3o_qwen_inference.py:103` | DiT CFG 강도 |
| `num_inference_steps` | 30 | `blip3o_qwen_inference.py:105` | DiT denoising step 수 |
| `num_images_per_prompt` | 1 | `blip3o_qwen_inference.py:106` | 프롬프트당 생성 이미지 수 |
| `n_query` | 64 | `config.n_query` | Latent query 개수 (= 8×8 spatial grid) |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | `blip3o_qwen_inference.py:59` | Flow Matching ODE solver |
| Sigma schedule | `linspace(1.0, 1/30, 30)` | `blip3o_qwen_inference.py:130` | 선형 sigma 감소 |

#### Stage 2 (UNet — pixel image 생성)

| 파라미터 | 기본값 | 위치 | 설명 |
|----------|--------|------|------|
| `guidance_scale` | 3.0 | `pipeline_llava_gen.py:98` | UNet CFG 강도 |
| `num_inference_steps` | 50 | `pipeline_llava_gen.py:97` | UNet denoising step 수 |
| `height` / `width` | 1024 / 1024 | `pipeline_llava_gen.py:95-96` | 출력 이미지 해상도 |
| Scheduler | `EulerDiscreteScheduler` | `pipeline_llava_gen.py:55` | Karras sigma schedule |
| `original_size` | `[1024, 1024]` | `pipeline_llava_gen.py:100` | SDXL time_ids 조건 |
| `crop_info` | `[0, 0]` | `pipeline_llava_gen.py:99` | SDXL crop offset 조건 |

#### 공통

| 파라미터 | 기본값 | 위치 | 설명 |
|----------|--------|------|------|
| `seed` | 42 | `inference.py:112` | 재현성을 위한 random seed |

### 6.2 모델 구성 하이퍼파라미터

| 파라미터 | 8B 모델 값 | 설명 |
|----------|-----------|------|
| LLM hidden size | 3584 | Qwen2.5-VL-7B hidden dimension |
| LLM layers | 28 | Decoder layer 수 |
| LLM heads | 28 (q), 4 (kv) | GQA: k,v는 512차원 (4 heads × 128) |
| LLM vocab size | 151,668 | 토큰 수 |
| DiT hidden size | 1792 | = EVA-CLIP feature dim |
| DiT layers | 24 | Transformer block 수 |
| DiT heads | 28 | Full multi-head attention |
| DiT latent size | 8×8 | Spatial resolution |
| DiT channels | 1792 | = EVA-CLIP feature dim |
| EVA-CLIP image size | 448×448 | 학습 시 target 이미지 크기 |
| EVA-CLIP patches | 32×32 = 1024 | Pool → 8×8 = 64 |
| gen_pooling | `early_pool2d_4` | 학습 시 EVA-CLIP feature pooling |

### 6.3 Precision & Device

| 설정 | 값 | 비고 |
|------|-----|------|
| 모델 로딩 dtype | `torch.float16` | `builder.py:37` |
| Pipeline dtype | `torch.bfloat16` | `inference.py:56` |
| Device | `cuda:0` (기본) | 모델과 파이프라인 동일 GPU |

---

## 7. 코드 참조 인덱스

| 파일 | 핵심 함수/클래스 | 역할 |
|------|------------------|------|
| `inference.py` | `add_template()`, main loop | 인퍼런스 진입점 |
| `pipeline_llava_gen.py` | `EmuVisualGenerationPipeline`, `_prepare_and_encode_inputs()` | Diffusion Decoder 파이프라인 (Stage 2: UNet + VAE) |
| `blip3o/model/builder.py` | `load_pretrained_model()` | 모델 로딩 |
| `blip3o/model/language_model/blip3o_qwen_inference.py` | `generate_image()`, `sample_images()` | Stage 1 핵심 생성 로직 (LLM + DiT) |
| `blip3o/model/blip3o_arch.py` | `blip3oMetaModel`, `blip3oMetaForCausalLM` | 모델 초기화, 공통 메서드 |
| `blip3o/model/nextdit_crossattn.py` | `NextDiTCrossAttn` | DiT wrapper |
| `blip3o/model/lumina_nextdit2d.py` | `LuminaNextDiT2DModel`, `LuminaNextDiTBlock` | DiT 본체 |
| `blip3o/model/multimodal_encoder/builder.py` | `build_dit()`, `build_gen_vision_tower()` | 컴포넌트 팩토리 |
| `blip3o/model/multimodal_encoder/eva_clip/eva_clip_encoder.py` | `EvaClipVisionTower` | EVA-CLIP 인코더 |
| `blip3o/model/multimodal_projector/builder.py` | `build_vision_projector()` | MLP/Linear projector |
| `blip3o/conversation.py` | `conv_qwen`, `Conversation` | ChatML 템플릿 |
| `blip3o/constants.py` | `IMAGE_TOKEN_IDX`, `UND_IMAGE_TOKEN_IDX` | 특수 토큰 상수 |
