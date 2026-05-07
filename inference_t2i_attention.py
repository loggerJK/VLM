import argparse
import json
import os
import random
import re
from datetime import timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from einops import rearrange
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from tqdm import tqdm
from transformers import AutoModelForCausalLM, set_seed

from janus.models import MultiModalityCausalLM, VLChatProcessor


torch.set_grad_enabled(False)


def sanitize_prompt_dirname(prompt_text: str, max_length: int = 120) -> str:
    sanitized = re.sub(r"\s+", "_", prompt_text.strip())
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", sanitized)
    sanitized = sanitized.strip("._")
    if not sanitized:
        sanitized = "prompt"
    return sanitized[:max_length]


def find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> Optional[int]:
    if not subsequence or len(subsequence) > len(sequence):
        return None
    for start_idx in range(len(sequence) - len(subsequence) + 1):
        if list(sequence[start_idx : start_idx + len(subsequence)]) == list(subsequence):
            return start_idx
    return None


def decode_single_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode([token_id])


def build_user_prompt_token_metadata(
    tokenizer,
    full_prompt: str,
    prompt_text: str,
    full_token_ids: List[int],
) -> Dict:
    char_start = full_prompt.index(prompt_text)
    char_end = char_start + len(prompt_text)

    token_indices: List[int] = []
    try:
        encoded = tokenizer(
            full_prompt,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        encoded_ids = encoded["input_ids"]
        token_offset = 0
        if len(encoded_ids) != len(full_token_ids):
            token_offset = find_subsequence(full_token_ids, encoded_ids)
            if token_offset is None:
                raise ValueError("Failed to align offset-mapped tokens with the full prompt tokenization.")

        for token_idx, (offset_start, offset_end) in enumerate(encoded["offset_mapping"]):
            if offset_end <= offset_start:
                continue
            if offset_end <= char_start or offset_start >= char_end:
                continue
            token_indices.append(token_offset + token_idx)
    except Exception:
        token_indices = []

    if not token_indices:
        prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        start_idx = find_subsequence(full_token_ids, prompt_token_ids)
        if start_idx is None:
            raise ValueError("Failed to locate user prompt token span inside the Janus prompt.")
        token_indices = list(range(start_idx, start_idx + len(prompt_token_ids)))

    token_ids = [full_token_ids[idx] for idx in token_indices]
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
    else:
        token_pieces = [decode_single_token(tokenizer, token_id) for token_id in token_ids]

    return {
        "query_indices": token_indices,
        "token_ids": token_ids,
        "token_pieces": token_pieces,
        "token_decoded": [decode_single_token(tokenizer, token_id) for token_id in token_ids],
        "char_span": [char_start, char_end],
        "prompt_text": prompt_text,
    }


def set_all_seeds(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


def resolve_inner_model(mmgpt: MultiModalityCausalLM):
    try:
        from peft import PeftModel

        if isinstance(mmgpt.language_model, PeftModel):
            return mmgpt.language_model.model.model
    except ImportError:
        pass
    return mmgpt.language_model.model


def force_eager_attention(module):
    config = getattr(module, "config", None)
    if config is not None and hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"


def module_device(module) -> torch.device:
    return next(module.parameters()).device


@torch.inference_mode()
def generate_one_image(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1.0,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
):
    inner_model = resolve_inner_model(mmgpt)
    device = module_device(inner_model)

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)

    tokens = torch.zeros((2, len(input_ids)), dtype=torch.long, device=device)
    tokens[0, :] = input_ids_tensor
    tokens[1, :] = input_ids_tensor
    tokens[1, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((1, image_token_num_per_image), dtype=torch.long, device=device)

    outputs = None
    for step_idx in range(image_token_num_per_image):
        outputs = inner_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if step_idx != 0 else None,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = mmgpt.gen_head(hidden_states[:, -1, :])

        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        guided_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(guided_logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[0, step_idx] = next_token.squeeze(dim=-1)

        duplicated = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)],
            dim=1,
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(duplicated)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    return input_ids, generated_tokens


@torch.inference_mode()
def save_attention_maps(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    input_ids: List[int],
    generated_tokens: torch.LongTensor,
    prompt_text: str,
    prompt: str,
    sample_idx: int,
    sample_seed: int,
    attention_dir: str,
    img_size: int,
    patch_size: int,
):
    inner_model = resolve_inner_model(mmgpt)
    force_eager_attention(inner_model)

    device = module_device(inner_model)
    prompt_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    prompt_embeds = mmgpt.language_model.get_input_embeddings()(prompt_ids_tensor)
    image_embeds = mmgpt.prepare_gen_img_embeds(generated_tokens.view(-1)).unsqueeze(0)
    full_inputs_embeds = torch.cat([prompt_embeds, image_embeds], dim=1)

    outputs = inner_model(
        inputs_embeds=full_inputs_embeds,
        use_cache=False,
        output_attentions=True,
        return_dict=True,
    )
    attentions = outputs.attentions
    if attentions is None:
        raise RuntimeError("Janus/Llama forward did not return attention weights.")

    user_prompt_meta = build_user_prompt_token_metadata(
        vl_chat_processor.tokenizer,
        prompt,
        prompt_text,
        input_ids,
    )
    text_indices = torch.tensor(user_prompt_meta["query_indices"], dtype=torch.long, device=device)
    image_start = len(input_ids)
    image_indices = torch.arange(
        image_start,
        image_start + generated_tokens.shape[1],
        dtype=torch.long,
        device=device,
    )

    os.makedirs(attention_dir, exist_ok=True)

    grid_size = [img_size // patch_size, img_size // patch_size]
    for layer_idx, layer_attn in enumerate(attentions):
        # layer_attn: [batch, heads, q_len, k_len]
        selected = layer_attn[0].index_select(1, image_indices).index_select(2, text_indices)
        save_path = os.path.join(attention_dir, f"layer_{layer_idx:02d}_i2t.pt")
        torch.save(
            {
                "attention_map": selected.to(dtype=torch.float32).cpu(),
                "layer": int(layer_idx),
                "prompt_text": prompt_text,
                "sample_idx": int(sample_idx),
                "sample_seed": int(sample_seed),
                "direction": "i2t",
                "query_type": "image",
                "key_type": "text",
                "text_token_indices": list(user_prompt_meta["query_indices"]),
                "image_token_indices": list(range(image_start, image_start + generated_tokens.shape[1])),
                "text_token_ids": list(user_prompt_meta["token_ids"]),
                "text_token_pieces": list(user_prompt_meta["token_pieces"]),
                "text_token_decoded": list(user_prompt_meta["token_decoded"]),
                "query_token_ids": list(user_prompt_meta["token_ids"]),
                "query_token_pieces": list(user_prompt_meta["token_pieces"]),
                "query_token_decoded": list(user_prompt_meta["token_decoded"]),
                "grid_size": grid_size,
            },
            save_path,
        )


def decode_image(
    mmgpt: MultiModalityCausalLM,
    generated_tokens: torch.LongTensor,
    img_size: int,
    patch_size: int,
) -> Image.Image:
    decoded = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[1, 8, img_size // patch_size, img_size // patch_size],
    )
    decoded = decoded.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    decoded = np.clip((decoded + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = decoded[0]
    return Image.fromarray(visual_img)


def parse_args():
    parser = argparse.ArgumentParser(description="Janus text-to-image inference with attention saving")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B")
    parser.add_argument("--lora_ckpt_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output_attention")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--cfg_weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_attention_maps", action="store_true")
    parser.add_argument("--no-save-attention-maps", dest="save_attention_maps", action="store_false")
    parser.set_defaults(save_attention_maps=True)
    return parser.parse_args()


def load_prompt_data(args) -> List[Dict]:
    prompt_data_list: List[Dict] = []
    if args.prompt:
        prompt_data_list.append(
            {
                "prompt": args.prompt,
                "metadata": {
                    "tag": "text-to-image",
                    "prompt": args.prompt,
                    "include": [],
                    "exclude": [],
                },
            }
        )

    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                prompt_str = data.get("prompt") or data.get("user_prompt")
                if prompt_str:
                    prompt_data_list.append({"prompt": prompt_str, "metadata": data})

    if not prompt_data_list:
        raise ValueError("At least one prompt must be provided via --prompt or --prompt_file.")
    return prompt_data_list


def main():
    args = parse_args()

    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(days=1))
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    prompt_data_list = load_prompt_data(args)
    my_batch = list(enumerate(prompt_data_list))[rank::world_size]

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = model.to(torch.bfloat16).to(device).eval()

    if args.lora_ckpt_path is not None:
        from peft import PeftModel

        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            args.lora_ckpt_path,
            torch_device="cpu",
        )
        model = model.to(torch.bfloat16).to(device).eval()

    force_eager_attention(model.language_model)
    force_eager_attention(resolve_inner_model(model))

    for index, prompt_data in tqdm(my_batch, total=len(my_batch), desc=f"Rank {rank}"):
        prompt_text = prompt_data["prompt"]
        metadata = prompt_data["metadata"]

        prompt_dir = os.path.join(args.output_dir, sanitize_prompt_dirname(prompt_text))
        samples_dir = os.path.join(prompt_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        with open(os.path.join(prompt_dir, "metadata.jsonl"), "w") as mf:
            json.dump(metadata, mf, ensure_ascii=False, indent=2)

        conversation = [
            {"role": "<|User|>", "content": prompt_text},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        user_prompt_meta = build_user_prompt_token_metadata(
            vl_chat_processor.tokenizer,
            prompt,
            prompt_text,
            input_ids,
        )
        with open(os.path.join(prompt_dir, "user_prompt_tokens.json"), "w") as tf:
            json.dump(
                {
                    "prompt_text": prompt_text,
                    "query_indices": user_prompt_meta["query_indices"],
                    "token_ids": user_prompt_meta["token_ids"],
                    "token_pieces": user_prompt_meta["token_pieces"],
                    "token_decoded": user_prompt_meta["token_decoded"],
                    "char_span": user_prompt_meta["char_span"],
                    "grid_size": [args.img_size // args.patch_size, args.img_size // args.patch_size],
                },
                tf,
                ensure_ascii=False,
                indent=2,
            )

        sample_tensors = []
        for sample_idx in range(args.n_samples):
            sample_seed = args.seed + index * args.n_samples + sample_idx
            set_all_seeds(sample_seed)

            cond_input_ids, generated_tokens = generate_one_image(
                model,
                vl_chat_processor,
                prompt,
                temperature=args.temperature,
                cfg_weight=args.cfg_weight,
                image_token_num_per_image=(args.img_size // args.patch_size) ** 2,
            )

            if args.save_attention_maps:
                sample_attention_dir = os.path.join(prompt_dir, "attentions", f"{sample_idx:04d}")
                save_attention_maps(
                    model,
                    vl_chat_processor,
                    cond_input_ids,
                    generated_tokens,
                    prompt_text,
                    prompt,
                    sample_idx,
                    sample_seed,
                    sample_attention_dir,
                    args.img_size,
                    args.patch_size,
                )

            sample_image = decode_image(model, generated_tokens, args.img_size, args.patch_size)
            sample_path = os.path.join(samples_dir, f"{sample_idx:04d}.png")
            sample_image.save(sample_path)
            sample_tensors.append(ToTensor()(sample_image))

        if sample_tensors:
            grid = make_grid(torch.stack(sample_tensors, dim=0), nrow=len(sample_tensors))
            grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
            Image.fromarray(grid.astype(np.uint8)).save(os.path.join(prompt_dir, "grid.png"))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
