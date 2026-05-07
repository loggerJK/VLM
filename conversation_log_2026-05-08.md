# Conversation Log

Date: 2026-05-08

## User

`/data/mm-llm-backbone_890/personal/heidi/ufo/abl/VLM/t2i_inference.sh`는 `/data/mm-llm-backbone_890/personal/heidi/ufo/abl/VLM/count_subset.jsonl`로 t2i generation 할때 모든 레이어의 attention map 을 캐싱해서 저장하는 스크립트임. 모델이름은 lumina이고, context 파악하길 바람.

내가 하고싶은것은 동일한 분석을 `/data/mm-llm-backbone_890/personal/heidi/ufo/abl/VLM_ori` 의 모델, 즉 janus에서도 하는것임. janus의 generation 예시 스크립트는 `/data/mm-llm-backbone_890/personal/heidi/ufo/abl/VLM_ori/evaluation_scripts/counting/generation/evaluate_counting_generation_multigpu_more_samples.sh` 이것임. (베이스라인과 lora adapted 모델에 대해 전부 진행해야 함).

먼저 모델을 파악하길 바람.
참고로, Lumina는 diffusion model인 반면 janus는 autoregressive모델임. 그래서 이미지 생성기작이 다른것을 감안하여야 함.

내가 janus에서 추출하고싶은 attention map은 이미지 생성 시 마지막 이미지 토큰이 생성될때, full attention map에서 query=image, key=text인 영역들을 저장하는 것이며, 각 레이어마다 head 축으로 쌓아서 저장해주길바람. 저장되는 한 단위는 `(head, seq_image, seq_prompt)` 여야 할것임. 추후 각 text token의 image column 기준으로 reshape해서 2d spatial attention map 을 시각화해야하며, 이때 각 어텐션 맵이 어떤 텍스트 토큰에 해당하는것인지 알아야 하기 때문에, tokenize 된 text가 decode되면 어떤 token이 되는지도 함께 저장해주길 바람. (저장산물은 lumina의 것과 동일, 단 timestep별 저장은 이 모델에 해당하지 않음)

먼저 모델을 파악하고, 내가 요청한 것을 나에게 확인해. 그리고 선택해야 하는 옵션이 있다면 나에게 물어봐.

## Assistant

`Lumina` 쪽은 `VLM/t2i_inference.sh`가 `inference/inference_t2i_seeds.py`를 호출하고, 여기서 prompt별 디렉토리를 만든 뒤 attention 저장 메타데이터를 구성해서 모델 내부 `set_attention_capture`로 넘긴다는 점을 확인함. 실제 저장 payload는 레이어별 `.pt`이고, 핵심 필드는 `attention_map`, `prompt_text`, `layer`, `query/key token indices`, `query_token_ids/pieces/decoded`, `grid_size`이며 diffusion 특성상 timestep별 저장이 있음.

`Janus` 쪽은 `VLM_ori/evaluation_scripts/counting/generation/evaluate_counting_generation_multigpu_more_samples.sh`가 `evaluate_counting_generation_multigpu.py`를 호출하고, 그 안의 `generate()`가:

- `prompt = sft_template + <begin_of_image>`
- 내부 `LlamaForCausalLM` backbone에 `inputs_embeds`를 넣음
- `576`개 image token을 autoregressive하게 한 개씩 샘플링
- 매 step에서 `gen_head(hidden_states[:, -1, :])`로 다음 image token 예측
- CFG 때문에 cond/uncond 2개 시퀀스를 같이 돌림
- 마지막에 `gen_vision_model.decode_code(...)`로 이미지 복원

이라는 구조임을 설명함.

또한 KV cache를 쓰는 autoregressive 생성의 특성상 마지막 스텝 forward에서는 query가 마지막 토큰 1개뿐이므로, 요청한 `(head, seq_image, seq_prompt)` shape를 얻으려면 생성 완료 후 완성된 전체 시퀀스에 대해 `use_cache=False` full forward를 다시 한 번 수행해 각 레이어 attention에서 `query=image`, `key=text` 영역을 잘라 저장하는 방식이 가장 자연스럽다고 설명함.

확인 질문으로 아래 3가지를 제시함.

1. 생성 완료 후 `full forward` 1회를 추가하는 방식으로 가도 되는지
2. `text token` 범위를 시스템 프롬프트를 제외한 실제 사용자 텍스트로 제한할지
3. Janus용 새 inference 스크립트를 만들지, 기존 eval 경로를 유지할지

## User

1. 그렇게 하면 됨. 2. text token은 시스템프롬프트 제외 텍스트 토큰으로 제한 3. 새 inference 스크립트를 작성하면되고, eval은 할 필요없을듯? 그냥 인퍼런스만 ㄱㄱ

## Assistant

요청을 반영해 Janus용 새 inference 스크립트와 실행용 쉘 스크립트를 구현함.

추가된 파일:

- `VLM_ori/inference_t2i_attention.py`
- `VLM_ori/t2i_inference_attention.sh`

구현 내용:

- `count_subset.jsonl` 형식 입력 지원
- prompt별 출력 디렉토리 생성
- `samples/`, `attentions/<sample_id>/`, `metadata.jsonl`, `user_prompt_tokens.json`, `grid.png` 저장
- baseline / counting LoRA / generation LoRA 모두 실행 가능하도록 쉘 스크립트 구성
- Janus 생성 후 full forward 1회를 통해 레이어별 `query=image`, `key=text` attention 저장
- 저장 tensor shape는 `(head, seq_image, seq_prompt)`
- text token 메타데이터로 `text_token_ids`, `text_token_pieces`, `text_token_decoded`, `text_token_indices` 저장
- 추가로 `prompt_text`, `sample_idx`, `sample_seed`, `grid_size` 저장

검증 내용:

- `python -m py_compile VLM_ori/inference_t2i_attention.py` 통과
- 실행 스크립트에 실행 권한 부여
- 실제 모델 런타임 테스트는 아직 수행하지 않음
- `git` 상태 확인은 repository safe.directory 설정 문제로 실패

## User

지금 대화 기록 VLM_ori 아래에 남겨줄래?
