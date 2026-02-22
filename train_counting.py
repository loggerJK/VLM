# from multiprocess.managers import dispatch
# from cupy._manipulation.split import split
import os
import torch
import re
import argparse
import requests
from io import BytesIO
from datasets import load_dataset, load_from_disk
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoConfig
from transformers import TrainerCallback
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image
import numpy as np
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm
import wandb
from transformers.modeling_outputs import CausalLMOutputWithPast
from tqdm import tqdm


def extract_number(text):
    """모델 출력에서 **숫자** 패턴 추출"""
    match = re.search(r"\*\*(\d+)\*\*", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return -1

def encode_image_to_vq_tokens(vq_model, image, img_size=384):
    """PIL Image -> 576 VQ token IDs (LongTensor)"""
    image = image.resize((img_size, img_size))
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor * 2 - 1  # [0,1] -> [-1,1]
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype

    with torch.no_grad():
        _, _, (_, _, indices) = vq_model.encode(image_tensor.to(device=device, dtype=dtype))

    return indices.view(-1)  # LongTensor [576]


class EnhancedMultiModalModel(MultiModalityCausalLM):
    """
    향상된 멀티모달 인과 언어 모델로, 사용자 정의 순전파(forward) 로직을 지원합니다.
    학습을 위해 MultiModalityCausalLM에 forward 메서드를 추가한 클래스입니다.
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        image_token_masks: torch.BoolTensor = None,
        labels: torch.LongTensor = None,
        gen_token_mask: torch.BoolTensor = None,
        **kwargs,
    ):
        """
        이미지와 텍스트의 처리를 지원하는 forward 메서드.
        gen_token_mask가 있으면 generation forward, 없으면 understanding forward.
        """
        if gen_token_mask is not None and gen_token_mask.any():
            # === GENERATION forward ===
            return self._forward_generation(input_ids, attention_mask, labels, gen_token_mask)

        if pixel_values is not None and image_token_masks is not None:
            # 멀티모달 입력 처리 (텍스트 임베딩 일부를 이미지 임베딩으로 교체)
            inputs_embeds = self._process_multimodal_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_masks=image_token_masks,
            )
        else:
            # 텍스트 전용 입력 처리
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 이미 inputs_embeds를 계산했으므로 input_ids는 제거 (중복 방지)
        kwargs.pop("inputs_embeds", None)

        # 실제 언어 모델 호출
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return outputs

    def _forward_generation(self, input_ids, attention_mask, labels, gen_token_mask):
        """Image generation forward: text -> LLM -> gen_head -> VQ logits"""
        # 1) Base text embedding
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2) Image token 위치는 gen_embed + gen_aligner로 교체
        #    (teacher forcing: GT VQ token IDs를 입력으로 사용)
        image_token_ids = input_ids[gen_token_mask]
        image_embeds = self.prepare_gen_img_embeds(image_token_ids)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[gen_token_mask] = image_embeds.to(inputs_embeds.dtype)

        # 3) LLM forward (hidden states만 필요 — lm_head 스킵)
        #    PeftModel 구조: language_model.model → LlamaForCausalLM → .model → LlamaModel
        #    Non-LoRA 구조: language_model.model → LlamaModel
        lm = self.language_model
        # PeftModel이면 한 단계 더 진입
        if hasattr(lm, 'model') and hasattr(lm.model, 'model'):
            lm_model = lm.model.model  # LlamaModel (BaseModelOutput with last_hidden_state)
        elif hasattr(lm, 'model'):
            lm_model = lm.model        # LlamaModel
        else:
            lm_model = lm.base_model
        outputs = lm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden_dim]

        # 4) gen_head로 image logits 생성
        gen_logits = self.gen_head(hidden_states)  # [batch, seq, 16384]

        # 5) Shifted cross-entropy loss (next-token prediction)
        shift_logits = gen_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        return CausalLMOutputWithPast(loss=loss, logits=gen_logits)

    def _process_multimodal_inputs(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_token_masks: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        시각적 특징을 추출하고 정렬하여 텍스트 임베딩 시퀀스 내의 적절한 위치에 삽입합니다.
        """
        bs, n = pixel_values.shape[0:2] # (bs, n, 3, h, w)
        images = pixel_values.view(bs * n, *pixel_values.shape[2:]) # (bs * n, 3, h, w)
        
        # 비전 인코더 및 얼라이너 통과
        image_features = self.vision_model(images) 
        aligned_features = self.aligner(image_features) # (bs * n, n_image_tokens, D)

        # 배치 차원 재구성
        aligned_features = aligned_features.view(bs, n, *aligned_features.shape[1:]) # (bs, n, n_image_tokens, D)
        aligned_features = aligned_features.flatten(1, 2) # (bs, n * n_image_tokens, D)

        # 기본 텍스트 임베딩 생성
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 마스크를 사용하여 이미지 토큰 자리에 정렬된 이미지 특징 삽입
        for i in range(bs):
            num_image_tokens = image_token_masks[i].sum().item()
            num_aligned_tokens = aligned_features[i].shape[0]
            if num_image_tokens != num_aligned_tokens:
                raise ValueError(
                    f"Mismatch at sample {i}: "
                    f"image_token_masks has {num_image_tokens} tokens, "
                    f"but aligned_features has {num_aligned_tokens} tokens!"
                )
            text_embeds[i][image_token_masks[i]] = aligned_features[i]

        return text_embeds

class StreamingDatasetWrapper(torch.utils.data.IterableDataset):
    """
    URL만 있는 데이터셋을 받아 실시간으로 이미지를 다운로드하여 제공하는 Wrapper.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_samples = len(dataset)
        tmp_sample = self.dataset[0]
        if 'image' in tmp_sample.keys():
            print("[INFO] Dataset already contains 'image' column. Using existing images.")
        else:
            print("[INFO] Dataset does not contain 'image' column. Images will be downloaded on-the-fly.")

    def fetch_image(self, url):
        if not url: return None
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                return img
        except Exception:
            pass
        return None

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            indices = range(self.num_samples)
        else:
            per_worker = int(np.ceil(self.num_samples / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, self.num_samples)
            indices = range(start, end)

        for idx in indices:
            item = self.dataset[idx]
            
            # 만약 'image' column이 없다면
            if 'image' in item.keys():
                image = item['image']
            else :
                image = self.fetch_image(item.get('image_url'))
                
                while image is None:
                    random_idx = np.random.randint(0, self.num_samples)
                    item = self.dataset[random_idx]
                    image = self.fetch_image(item.get('image_url'))
                
            image = image.convert("RGB")
            # image.save(f"./debug.png")  # 디버깅용 저장
            
            output_item = item.copy()
            output_item['image'] = image
            yield output_item

def collate_fn(batch, processor, task="counting"):
    prepare_list = []

    q_col = 'question' if task == 'pointing' else 'question_count'
    a_col = 'answer' if task == 'pointing' else 'answer_count'

    for item in batch:
        image = item.get('image')
        if image is None: continue

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{item[q_col]}",
                "images": [image]
            },
            {
                "role": "<|Assistant|>",
                "content": item[a_col]
            }
        ]
        
        try:
            prepare = processor(
                conversations=conversation,
                images=[image],
                force_batchify=False
            )
            prepare_list.append(prepare)
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    if not prepare_list:
        return {}

    inputs = processor.batchify(prepare_list)
    inputs_dict = dict(inputs)
    
    input_ids = inputs_dict['input_ids']
    labels = input_ids.clone() # (bs, seq_len)
    
    # Assistant 역할 토큰 ID를 가져옵니다.
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|Assistant|>")

    for i in range(input_ids.shape[0]):
        # Assistant 토큰의 위치를 찾습니다.
        assistant_indices = (input_ids[i] == assistant_token_id).nonzero(as_tuple=True)[0]
        
        if assistant_indices.numel() > 0:
            # 첫 번째 등장 위치를 어시스턴트 턴의 시작으로 간주합니다.
            split_point = assistant_indices[0]
            
            # 어시스턴트 토큰을 포함하여 그 이전의 모든 레이블을 -100으로 마스킹합니다.
            # 모델이 어시스턴트 토큰 이후부터 답변을 생성하도록 유도합니다.
            labels[i, :split_point + 1] = -100

    # 이미지 토큰도 손실 계산에서 제외합니다.
    if "images_seq_mask" in inputs_dict:
        image_mask = inputs_dict["images_seq_mask"]
        # 이미지 마스크가 True인 위치의 레이블을 -100으로 설정합니다.
        labels[image_mask] = -100
        inputs_dict["image_token_masks"] = image_mask
    
    inputs_dict["labels"] = labels
    

    # 텐서가 아닌 필드 제거
    keys_to_remove = [k for k, v in inputs_dict.items() if not isinstance(v, torch.Tensor)]
    for k in keys_to_remove:
        del inputs_dict[k]

    return inputs_dict

def collate_fn_generation(batch, processor, vq_model, img_size=384):
    """
    Text-to-image 학습용 collate.
    각 샘플: {text, image} -> 시퀀스 구성:
    [text_prompt_tokens] [assistant] [boi] [vq_tok_0..575] [eoi]
    """
    boi_id = processor.image_start_id    # <begin_of_image>
    eoi_id = processor.image_end_id      # <end_of_image>

    input_ids_list, labels_list, gen_token_mask_list = [], [], []

    for item in batch:
        image = item.get('image')
        # TODO: 'descriptions' 가져와야 하는 것 아닌지?
        text = item.get('text') or item.get('caption') or item.get('prompt')
        if image is None or text is None:
            continue

        # 1) VQ encode
        try:
            vq_tokens = encode_image_to_vq_tokens(vq_model, image, img_size=img_size)  # [576]
        except Exception as e:
            print(f"Error encoding image: {e}")
            continue

        # 2) Text prompt 토큰화
        conversation = [
            {"role": "<|User|>", "content": text},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=processor.sft_format,
            system_prompt="",
        )
        text_ids = processor.tokenizer.encode(sft_format)  # List[int]

        # 3) 시퀀스 구성: [text] [boi] [vq_tok_0..575] [eoi]
        full_ids = text_ids + [boi_id] + vq_tokens.tolist() + [eoi_id]
        full_ids = torch.LongTensor(full_ids)

        # 4) Labels: text+boi 부분은 -100, VQ image tokens만 loss 계산 (eoi 제외)
        labels = torch.full_like(full_ids, -100)
        img_start = len(text_ids) + 1  # boi 다음부터
        num_vq = len(vq_tokens)
        labels[img_start:img_start + num_vq] = full_ids[img_start:img_start + num_vq]

        # 5) Gen token mask: image token 위치 표시 (boi, eoi 제외, VQ tokens만)
        num_vq_tokens = len(vq_tokens)
        gen_mask = torch.zeros(len(full_ids), dtype=torch.bool)
        gen_mask[img_start:img_start + num_vq_tokens] = True

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        gen_token_mask_list.append(gen_mask)

    if not input_ids_list:
        return {}

    # Left-padding: autoregressive 모델에서 마지막 토큰 위치가 중요하므로 실제 콘텐츠를 오른쪽 정렬.
    # F.pad(x, (left, right), value) 로 왼쪽에 max_len - len(x) 만큼 패딩.
    #
    # 예시) 배치 내 길이 7, 길이 5 시퀀스 (max_len=7):
    #   원본 시퀀스:    [text, text, boi, vq0, vq1, vq2, eoi]   (len=7)
    #                  [text, boi, vq0, vq1, eoi]                (len=5)
    #
    #   input_ids:      [text, text, boi, vq0, vq1, vq2, eoi]
    #                   [PAD,  PAD, text, boi, vq0, vq1, eoi]
    #   labels:         [-100, -100, -100, vq0, vq1, vq2, -100]
    #                   [-100, -100, -100, -100, vq0, vq1, -100]
    #   attention_mask: [1,    1,    1,    1,   1,   1,   1   ]
    #                   [0,    0,    1,    1,   1,   1,   1   ]
    #   gen_token_mask: [F,    F,    F,    T,   T,   T,   F   ]
    #                   [F,    F,    F,    F,   T,   T,   F   ]
    max_len = max(len(ids) for ids in input_ids_list)
    pad_id = processor.pad_id

    batched = {
        # 토큰 ID: 패딩 위치는 pad_id로 채움
        'input_ids': torch.stack([F.pad(ids, (max_len - len(ids), 0), value=pad_id) for ids in input_ids_list]),
        # Loss 대상: -100은 CrossEntropyLoss에서 무시됨, VQ 이미지 토큰 위치만 실제 값
        'labels': torch.stack([F.pad(lb, (max_len - len(lb), 0), value=-100) for lb in labels_list]),
        # 실제 토큰=1, 패딩=0: 모델이 패딩 위치를 attend하지 않도록 마스킹
        'attention_mask': torch.stack([F.pad(torch.ones(len(ids), dtype=torch.long), (max_len - len(ids), 0), value=0) for ids in input_ids_list]),
        # VQ 이미지 토큰 위치=True: _forward_generation에서 해당 위치의 embedding을 이미지 embedding으로 교체
        'gen_token_mask': torch.stack([F.pad(m, (max_len - len(m), 0), value=False) for m in gen_token_mask_list]),
    }
    return batched

@torch.no_grad()
def generate_image_from_prompt(
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    prompt_text: str,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    """
    공식 Janus T2I 로직 기반 이미지 생성 (generation_inference.py 참조).
    CFG (classifier-free guidance) + temperature sampling + KV cache 사용.
    LoRA/PeftModel 호환.

    Returns:
        PIL.Image
    """
    device = next(model.parameters()).device

    # 1) Prompt → token IDs
    conversation = [
        {"role": "<|User|>", "content": prompt_text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + processor.image_start_tag
    input_ids = processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    # 2) CFG용 dual batch: [conditional, unconditional]
    #    unconditional = 중간 토큰을 pad로 교체 (공식 구현과 동일)
    tokens = torch.zeros((2, len(input_ids)), dtype=torch.int, device=device)
    tokens[0, :] = input_ids  # conditional
    tokens[1, :] = input_ids  # unconditional
    tokens[1, 1:-1] = processor.pad_id

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)

    # 3) LLM backbone 접근 (LoRA/PeftModel 호환)
    lm = model.language_model
    if hasattr(lm, 'model') and hasattr(lm.model, 'model'):
        lm_model = lm.model.model  # PeftModel → LlamaForCausalLM → LlamaModel
    elif hasattr(lm, 'model'):
        lm_model = lm.model
    else:
        lm_model = lm.base_model

    # 4) Autoregressive generation with CFG + KV cache
    generated_tokens = torch.zeros((1, image_token_num_per_image), dtype=torch.int, device=device)
    past_key_values = None

    for i in tqdm(range(image_token_num_per_image), total=image_token_num_per_image, desc="Generating image tokens"):
        outputs = lm_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state

        logits = model.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0:1, :]
        logit_uncond = logits[1:2, :]

        # CFG: logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        logits_cfg = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits_cfg / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]

        generated_tokens[:, i] = next_token.squeeze(-1)

        # Next step input: same token for both cond & uncond
        next_token_pair = next_token.repeat(2, 1).squeeze(-1)  # [2]
        img_embeds = model.prepare_gen_img_embeds(next_token_pair)
        inputs_embeds = img_embeds.unsqueeze(1)  # [2, 1, D]

    # 5) Decode VQ tokens → image
    spatial_size = img_size // patch_size
    dec = model.gen_vision_model.decode_code(
        generated_tokens, shape=[1, 8, spatial_size, spatial_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    return Image.fromarray(dec[0].astype(np.uint8))


def collate_fn_both(batch, processor, vq_model, task="counting", img_size=384):
    """
    Multi-task collate: understanding과 generation 샘플을 자동 분기.
    generation 샘플: 'text'/'caption'/'prompt' 필드 존재 + 'question_count'/'question' 없음
    understanding 샘플: 'question_count'/'question' 필드 존재
    """
    und_batch = []
    gen_batch = []

    for item in batch:
        # Generation 샘플 우선 판별 (같은 데이터셋에서 분리 시 gen에도 und 키가 존재할 수 있음)
        has_gen_keys = any(k in item and item[k] is not None for k in ['text', 'caption', 'prompt'])
        has_und_keys = any(k in item and item[k] is not None for k in ['question_count', 'question'])
        if has_gen_keys and not has_und_keys:
            gen_batch.append(item)
        elif has_und_keys and not has_gen_keys:
            und_batch.append(item)
        elif has_gen_keys:
            # 양쪽 키 모두 있으면 descriptions 유무로 판별
            if item.get('descriptions') is not None and item['descriptions'] != '' and item['descriptions'] != []:
                gen_batch.append(item)
            else:
                und_batch.append(item)
        # 키가 전혀 없으면 스킵

    if und_batch and gen_batch:
        raise ValueError(
            f"Mixed batch detected: {len(und_batch)} und + {len(gen_batch)} gen samples. "
            f"Use batch_size=1 for 'both' task, or separate datasets."
        )

    # Understanding 배치 처리
    if und_batch:
        return collate_fn(und_batch, processor, task=task)
    # Generation 배치 처리
    if gen_batch:
        return collate_fn_generation(gen_batch, processor, vq_model, img_size=img_size)

    return {}

class ValidationCallback(TrainerCallback):
    def __init__(self, processor, eval_dataset, args, log_freq=500):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.log_freq = log_freq
        self.args = args
        self.trainer = None
        self.val_table = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        optimized_param_names = [n for n, p in self.trainer.model.named_parameters() if p.requires_grad]
        print(f"Starting training..., Number of Optimized parameters: {len(optimized_param_names)}")
        with open(os.path.join(args.output_dir, "optimized_param_names.txt"), "w") as f:
            for name in optimized_param_names:
                f.write(f"{name}\n")
        if wandb.run is not None:
            wandb.config.update({"optimized_param_names": optimized_param_names})

    def _setup_validation_prompts(self):
        """val_gen split에서 generation validation 프롬프트 추출 (BAGEL/Lumina 동일 방식)"""
        import random
        rng = random.Random(42)

        val_ds = load_dataset("heez/pixmo-point-count-gen-und", split="val_gen")
        num_prompts = 5
        indices = rng.sample(range(len(val_ds)), min(num_prompts, len(val_ds)))

        self.validation_prompts = []
        for idx in indices:
            item = val_ds[idx]
            caption = item.get('descriptions', "Generate an image.")
            if isinstance(caption, list):
                caption = caption[0]
            self.validation_prompts.append(caption)

        print(f"[Validation] Selected {len(self.validation_prompts)} generation prompts from val_gen:")
        for i, p in enumerate(self.validation_prompts):
            print(f"  {i+1}. {p}")

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.args.save_steps == 0 and state.is_world_process_zero:
        # if state.global_step % self.args.save_steps == 0 and state.is_world_process_zero:
            checkpoint_dir = os.path.join(self.args.output_dir, f"epoch{int(state.epoch)}_step-{state.global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            if self.args.tuning_mode == "transformer_lora":
                if self.trainer.is_world_process_zero:
                    print(f"\n[Step {state.global_step}] Saving checkpoint to {checkpoint_dir}")
                    peft_model = self.trainer.model.module.language_model if hasattr(self.trainer.model, 'module') else self.trainer.model.language_model
                    peft_model.save_pretrained(checkpoint_dir)
                    self.processor.save_pretrained(checkpoint_dir)
            else:
                self.trainer.save_model(checkpoint_dir)
                if self.trainer.is_world_process_zero:
                    self.processor.save_pretrained(checkpoint_dir)

        if state.global_step > 0 and state.global_step % self.log_freq == 0 and state.is_world_process_zero:
        # if state.global_step % self.log_freq == 0 and state.is_world_process_zero:
            if self.args.task in ["counting", "pointing", "both"]:
                self.validate(model, state)
            if self.args.task in ["generation", "both"]:
                self.validate_generation(model, state)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero:
            epoch = int(state.epoch)
            checkpoint_dir = os.path.join(self.args.output_dir, f"epoch{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            if self.args.tuning_mode == "transformer_lora":
                print(f"\n[Epoch {epoch}] Saving LoRA checkpoint to {checkpoint_dir}")
                peft_model = self.trainer.model.module.language_model if hasattr(self.trainer.model, 'module') else self.trainer.model.language_model
                peft_model.save_pretrained(checkpoint_dir)
                self.processor.save_pretrained(checkpoint_dir)
            else:
                print(f"\n[Epoch {epoch}] Saving checkpoint to {checkpoint_dir}")
                self.trainer.save_model(checkpoint_dir)
                self.processor.save_pretrained(checkpoint_dir)



    def validate(self, model, state):
        print(f"\n[Step {state.global_step}] Running Validation on CountBenchQA...")
        model.eval()
        correct = 0
        total = 0
        eval_limit = 100 # len(self.eval_dataset)
        
        val_dataset_stream = load_dataset("heez/pixmo-point-count-gen-und", split="val_und", streaming=True)
        eval_dataset = iter(val_dataset_stream)  
        
        if hasattr(model, "module"):
            unwrap_model = model.module
        else:
            unwrap_model = model
            
        device = next(unwrap_model.parameters()).device

        import wandb
        
        # 테이블 초기화 (최초 1회)
        if self.val_table is None and wandb.run is not None:
            self.val_table = wandb.Table(columns=["Step", "Accuracy", "Details"], log_mode='MUTABLE')

        details_buffer = []

        with torch.no_grad():
            count = 0
            # len() 호출 없이 순차적으로 접근
            progress  = tqdm(range(eval_limit), desc="Validation", unit="sample")
            for item in eval_dataset:
                if count >= eval_limit:
                    break
                
                
                image = item.get('image')
                question = item.get('question')
                if image is None: continue
                gt_count = item.get('count')
                if gt_count is None: continue
                
                label = item.get('label', '<object>')

                question_prompt = f"{question}? Response Example : There are **<number>** {label} in the image."
                
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{question_prompt}",
                        "images": [image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                
                pil_images = [image]
                prepare_inputs = self.processor(
                    conversations=conversation, images=pil_images, force_batchify=True
                ).to(device)
                
                inputs_embeds = unwrap_model.prepare_inputs_embeds(**prepare_inputs)
                
                outputs = unwrap_model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    max_new_tokens=20,
                    do_sample=False,
                    use_cache=True
                )
                
                answer = self.processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                print("=========================================")
                print(f"answer : {answer}, gt: {gt_count}")
                pred_count = extract_number(answer)
                
                # 명시적 타입 변환
                gt_count_val = int(gt_count)
                pred_count_val = int(pred_count)
                is_correct = bool(pred_count_val == gt_count_val)
                
                if is_correct:
                    correct += 1
                total += 1
                count += 1
                
                # 결과 문자열 포맷팅
                res_str = f"[{count}] GT: {gt_count_val} | Pred: {pred_count_val} | Correct: {is_correct} | Ans: {answer}"
                details_buffer.append(res_str)
                
                progress.update(1)
            progress.close()

        
        accuracy = correct / total if total > 0 else 0
        print(f"Validation Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        if wandb.run is not None and self.val_table is not None:
            # 모든 상세 내용을 하나의 문자열로 합침
            all_details = "\n".join(details_buffer)
            
            # 하나의 Row 추가 (Step, Accuracy, Details)
            self.val_table.add_data(state.global_step, accuracy, all_details)
            
            wandb.log({
                "val/accuracy": accuracy, 
                "global_step": state.global_step,
                "val/predictions": self.val_table
            })
        
        model.train()

    def validate_generation(self, model, state):
        """텍스트 프롬프트로 이미지 생성하여 wandb에 로깅 (공식 Janus T2I 로직 사용)"""
        print(f"\n[Step {state.global_step}] Running Generation Validation...")
        model.eval()

        unwrap_model = model.module if hasattr(model, "module") else model

        if not hasattr(self, 'validation_prompts'):
            self._setup_validation_prompts()
        prompts = self.validation_prompts

        gen_images = []
        for prompt_text in prompts:
            try:
                pil_img = generate_image_from_prompt(
                    model=unwrap_model,
                    processor=self.processor,
                    prompt_text=prompt_text,
                    temperature=1.0,
                    cfg_weight=5.0,
                    img_size=self.args.gen_img_size,
                )
                gen_images.append(wandb.Image(pil_img, caption=prompt_text[:80]))
            except Exception as e:
                print(f"  Generation validation error: {e}")
                continue

        if wandb.run is not None and gen_images:
            wandb.log({
                "val/generated_images": gen_images,
                "global_step": state.global_step,
            })
            print(f"  Logged {len(gen_images)} generated images to wandb")

        model.train()

def main():
    parser = argparse.ArgumentParser(description="Train Janus model for counting task")
    parser.add_argument("--tuning_mode", type=str, default="lora", help="Tuning mode: 'full' or 'lora'")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B", help="Path to pretrained model")
    parser.add_argument("--data_path", type=str, default="./data/pixmo_counting_filtered", help="Path to prepared dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/janus-counting", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", type=int, default=1, help="Enable gradient checkpointing (1=True, 0=False)")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of dataloader workers")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every n steps")
    parser.add_argument("--log_freq", type=int, default=500, help="Log validation every n steps")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (if tuning_mode is 'lora')")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (if tuning_mode is 'lora')")
    parser.add_argument("--task", type=str, default="counting",
                        choices=["counting", "pointing", "generation", "both"],
                        help="Task: counting/pointing (understanding), generation (T2I), both (multi-task)")
    parser.add_argument("--gen_data_path", type=str, default=None,
                        help="HuggingFace dataset name or local path for generation data (text+image pairs)")
    parser.add_argument("--gen_img_size", type=int, default=384, help="Image size for VQ encoding")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for unlimited)")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume from (loads LoRA + gen_components)")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Use bitsandbytes 8-bit AdamW optimizer to save memory")
    parser.add_argument("--run_name", type=str, default=None,
                        help="WandB run name (defaults to output_dir if not set)")
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = "janus-counting"

    # Validate args
    if args.task in ["generation", "both"] and args.gen_data_path is None:
        raise ValueError("--gen_data_path is required for generation/both task")

    # Helper: load a HF or local dataset
    def _load_raw_dataset(path, split="train"):
        if not os.path.exists(path):
            return load_dataset(path, split=split, num_proc=64)
        else:
            return load_from_disk(path)

    # Helper: check if dataset has 'descriptions' column for filtering
    def _has_descriptions_column(ds):
        return 'descriptions' in ds.column_names

    # Helper: filter generation samples (descriptions is not None/empty)
    def _filter_gen(ds):
        print(f"Filtering generation samples from dataset with {len(ds)} samples")
        return ds.filter(
            lambda descriptions: descriptions is not None and descriptions != '' and descriptions != [],
            input_columns=['descriptions'], num_proc=64
        )

    # Helper: filter understanding samples (descriptions is None/empty)
    def _filter_und(ds):
        print(f"Filtering understanding samples from dataset with {len(ds)} samples")
        return ds.filter(
            lambda descriptions: descriptions is None or descriptions == '' or descriptions == [],
            input_columns=['descriptions'], num_proc=64
        )

    # Helper: map descriptions[0] -> text field for collate_fn_generation compatibility
    def _map_descriptions_to_text(ds):
        print(f"Mapping 'descriptions' to 'text' for generation dataset with {len(ds)} samples")
        def _to_text(descriptions):
            return {
                "text": descriptions[0] if isinstance(descriptions, list) else descriptions
            }

        return ds.map(
            _to_text,
            input_columns=["descriptions"],
            num_proc=64,
        )

    # Helper: wrap dataset (use StreamingDatasetWrapper only if images need downloading)
    def _maybe_wrap(ds):
        sample = ds[0]
        if 'image' in sample and sample['image'] is not None:
            return ds  # map-style dataset with images already loaded
        return StreamingDatasetWrapper(ds)

    # Load datasets based on task
    if args.task in ["counting", "pointing"]:
        raw_dataset = _load_raw_dataset(args.data_path)
        if _has_descriptions_column(raw_dataset):
            raw_dataset = _filter_und(raw_dataset)
            print(f"[INFO] Filtered understanding samples: {len(raw_dataset)}")
        train_dataset = _maybe_wrap(raw_dataset)
    elif args.task == "generation":
        gen_raw_dataset = _load_raw_dataset(args.gen_data_path)
        if _has_descriptions_column(gen_raw_dataset):
            gen_raw_dataset = _filter_gen(gen_raw_dataset)
            gen_raw_dataset = _map_descriptions_to_text(gen_raw_dataset)
            print(f"[INFO] Filtered generation samples: {len(gen_raw_dataset)}")
        train_dataset = _maybe_wrap(gen_raw_dataset)
    elif args.task == "both":
        from torch.utils.data import ConcatDataset
        # Understanding dataset
        und_raw_dataset = _load_raw_dataset(args.data_path)
        if _has_descriptions_column(und_raw_dataset):
            und_raw_dataset = _filter_und(und_raw_dataset)
            print(f"[INFO] Filtered understanding samples: {len(und_raw_dataset)}")
        # Generation dataset
        gen_raw_dataset = _load_raw_dataset(args.gen_data_path)
        if _has_descriptions_column(gen_raw_dataset):
            gen_raw_dataset = _filter_gen(gen_raw_dataset)
            gen_raw_dataset = _map_descriptions_to_text(gen_raw_dataset)
            print(f"[INFO] Filtered generation samples: {len(gen_raw_dataset)}")
        und_dataset = _maybe_wrap(und_raw_dataset)
        gen_dataset = _maybe_wrap(gen_raw_dataset)
        train_dataset = ConcatDataset([und_dataset, gen_dataset])
    
    print(f"Loading model from {args.model_path}...")
    processor = VLChatProcessor.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    # EnhancedMultiModalModel 클래스 사용 (forward 구현 포함)
    model = EnhancedMultiModalModel.from_pretrained(
        args.model_path,
        language_config=language_config,
        trust_remote_code=True,
        torch_dtype=torch_dtype
    )
    
    # Determine resume path: CLI arg takes priority, then env var
    resume_path = args.resume_checkpoint or os.getenv("RESUME_CHECKPOINT_PATH")

    if args.tuning_mode == "transformer_lora":
        for param in model.parameters():
            param.requires_grad = False

        if resume_path is None:
            print(f"[INFO] Transformer LoRA (tuning with r={args.lora_r}, alpha={args.lora_alpha})...")
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            model.language_model = get_peft_model(model.language_model, lora_config)
            model.language_model.print_trainable_parameters()
        else:
            print(f"[INFO] Resuming Transformer LoRA (with r={args.lora_r}, alpha={args.lora_alpha}) from checkpoint: {resume_path}")
            model.language_model = PeftModel.from_pretrained(
                model.language_model,
                resume_path,
                torch_dtype=torch_dtype,
            )
            for name, param in model.language_model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
            print(f"[INFO] Loaded LoRA adapter from checkpoint: {resume_path}")
            model.language_model.print_trainable_parameters()
    elif args.tuning_mode == "transformer":
        print("[INFO] Transformer fine-tuning...")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.language_model.parameters():
            param.requires_grad = True
    elif args.tuning_mode == "transformer_ONLY":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.language_model.model.parameters():
            param.requires_grad = True
        
    elif args.tuning_mode == "full":
        print("[INFO] Full model fine-tuning...")
        # 모든 파라미터가 기본적으로 requires_grad=True이므로 별도 설정 불필요
        for param in model.parameters():
            param.requires_grad = False
        for param in model.language_model.parameters():
            param.requires_grad = True
        for param in model.vision_model.parameters():
            param.requires_grad = True
        for param in model.aligner.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported tuning mode: {args.tuning_mode}")

    if args.gradient_checkpointing:
        print("Enabling Gradient Checkpointing...")
        model.gradient_checkpointing_enable()
        
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        # optim="adamw_bnb_8bit"  if args.use_8bit_adam else "adamw_torch",
        optim="adamw_bnb_8bit", # if args.use_8bit_adam else "adamw_torch",
        # optim="adamw_torch",
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        run_name=args.run_name,
        report_to="wandb",
        remove_unused_columns=False,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        ddp_find_unused_parameters=True if args.task in ["generation", "both"] else (False if args.gradient_checkpointing else None),
        dataloader_num_workers=0 if args.task in ["generation", "both"] else args.num_workers,
        # split_batches=True,
        # dispatch_batches=False
        accelerator_config = {
            # "split_batches": True,
            "dispatch_batches": False
        }
    )
    
    # Get VQ model reference for generation collate (frozen, on CPU initially)
    vq_model_ref = model.gen_vision_model if args.task in ["generation", "both"] else None

    if args.task in ["counting", "pointing"]:
        def data_collator(batch):
            return collate_fn(batch, processor, task=args.task)
    elif args.task == "generation":
        def data_collator(batch):
            return collate_fn_generation(batch, processor, vq_model_ref, img_size=args.gen_img_size)
    elif args.task == "both":
        def data_collator(batch):
            return collate_fn_both(batch, processor, vq_model_ref, task=args.task, img_size=args.gen_img_size)

    val_callback = ValidationCallback(processor, None, args, log_freq=args.log_freq)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[val_callback]
    )
    val_callback.trainer = trainer
    
    resume_kwargs = {}
    if os.getenv("WANDB_RESUME", "never") in ['allow', 'must']:
        resume_kwargs = {
            'wandb_resume': os.getenv("WANDB_RESUME"),
            'wandb_resume_id': os.getenv("WANDB_RESUME_ID"),
            'resume_global_step': int(os.getenv("RESUME_GLOBAL_STEP", "0")),
            'resume_epoch': int(os.getenv("RESUME_EPOCH", "0")),
        }

    print("Starting training...")
    trainer.train(**resume_kwargs)
    
    final_output_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Saving model to {final_output_dir}")
    
    if args.tuning_mode == "transformer_lora":
        # Save only the LoRA adapter
        model.language_model.save_pretrained(final_output_dir)
    else:
        # Save the full model
        trainer.save_model(final_output_dir)

    processor.save_pretrained(final_output_dir)

if __name__ == "__main__":
    main()
