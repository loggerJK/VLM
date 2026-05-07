"""
FID evaluation inference script for Janus.

Generates images from caption annotations with support for:
- single-process or `torchrun` multi-GPU inference
- optional LoRA adapter loading
- deterministic per-sample seeds
- resume via output file existence checks
"""

import argparse
import json
import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, set_seed

from janus.models import MultiModalityCausalLM, VLChatProcessor


torch.set_grad_enabled(False)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


def init_distributed():
    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(days=1))
        device = torch.device("cuda", local_rank)
        print(f"[Rank {rank}] Distributed initialized. World Size: {world_size}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Running in single process mode on {device}.")

    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description="FID Evaluation Inference (Janus)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/Janus-Pro-7B",
        help="Hugging Face model path or local checkpoint directory",
    )
    parser.add_argument(
        "--prompt_files",
        type=str,
        default="captions_val2014.json",
        help="Caption JSON file (absolute path or relative to this script)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_fid",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=1000,
        help="Maximum number of captions to process",
    )
    parser.add_argument(
        "--num_samples_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for each prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=65513,
        help="Base random seed",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="Generated image size",
    )
    parser.add_argument(
        "--image_token_num_per_image",
        type=int,
        default=576,
        help="Number of image tokens per generated image",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="Patch size used by the Janus image tokenizer",
    )
    parser.add_argument(
        "--lora_ckpt_path",
        type=str,
        default=None,
        help="Path to LoRA adapter checkpoint directory",
    )
    return parser.parse_args()


def resolve_prompt_file(prompt_file_path: str) -> str:
    if os.path.isabs(prompt_file_path):
        return prompt_file_path
    return os.path.join(os.path.dirname(__file__), prompt_file_path)


def load_prompts(prompt_file_path: str, max_prompts: int):
    with open(prompt_file_path, "r") as f:
        loaded = json.load(f)

    if isinstance(loaded, dict) and "annotations" in loaded:
        raw_items = loaded["annotations"]
    elif isinstance(loaded, list):
        raw_items = loaded
    else:
        raise ValueError(
            "Unsupported prompt file format. Expected a COCO-style JSON with "
            "`annotations` or a list of caption entries."
        )

    prompts = []
    for item in raw_items:
        if len(prompts) >= max_prompts:
            break

        if isinstance(item, dict):
            caption = item.get("caption")
        else:
            caption = item

        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Invalid caption entry: {item}")

        prompts.append(caption)

    return prompts


def build_prompt(vl_chat_processor: VLChatProcessor, caption: str) -> str:
    conversation = [
        {"role": "<|User|>", "content": caption},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft_format + vl_chat_processor.image_start_tag


def resolve_generation_backbone(mmgpt: MultiModalityCausalLM):
    inner_model = mmgpt.language_model.model

    # Some adapter wrapping paths expose the base transformer under `.model`.
    if hasattr(inner_model, "model") and hasattr(inner_model.model, "embed_tokens"):
        if not hasattr(inner_model, "embed_tokens"):
            inner_model = inner_model.model

    return inner_model


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cuda":
        generator_device = f"cuda:{device.index if device.index is not None else 0}"
    else:
        generator_device = "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return generator


def sample_next_tokens(probs: torch.Tensor, generators) -> torch.Tensor:
    if generators is None:
        return torch.multinomial(probs, num_samples=1)

    sampled = [
        torch.multinomial(probs[idx], num_samples=1, generator=generator)
        for idx, generator in enumerate(generators)
    ]
    return torch.stack(sampled, dim=0)


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1.0,
    parallel_size: int = 1,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
    generators=None,
):
    inner_model = resolve_generation_backbone(mmgpt)
    device = next(inner_model.parameters()).device

    input_ids = torch.tensor(
        vl_chat_processor.tokenizer.encode(prompt),
        dtype=torch.long,
        device=device,
    )

    tokens = torch.zeros((parallel_size * 2, input_ids.shape[0]), dtype=torch.long, device=device)
    tokens[:] = input_ids.unsqueeze(0)
    if input_ids.shape[0] > 2:
        tokens[1::2, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image), dtype=torch.long, device=device
    )

    outputs = None
    for image_token_idx in range(image_token_num_per_image):
        outputs = inner_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if outputs is not None else None,
        )
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = sample_next_tokens(probs, generators)
        generated_tokens[:, image_token_idx] = next_token.squeeze(dim=-1)

        duplicated_next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(duplicated_next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    decoded = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    decoded = decoded.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)

    return [Image.fromarray(decoded[idx]) for idx in range(parallel_size)]


def get_save_path(output_dir: str, prompt_index: int, sample_offset: int, num_samples_per_prompt: int) -> str:
    if num_samples_per_prompt == 1:
        filename = f"{prompt_index:05d}.png"
    else:
        filename = f"{prompt_index:05d}_{sample_offset:02d}.png"
    return os.path.join(output_dir, filename)


def main():
    args = parse_args()
    if args.num_samples_per_prompt != 1:
        raise ValueError("Janus FID inference only supports --num_samples_per_prompt=1")

    rank, world_size, _, device = init_distributed()
    set_all_seeds(args.seed)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    barrier()

    prompt_file_path = resolve_prompt_file(args.prompt_files)
    prompts = load_prompts(prompt_file_path, args.max_prompts)

    if rank == 0:
        print(f"Reading captions from {prompt_file_path}")
        print(f"Total prompts: {len(prompts)}")

    my_prompt_indices = list(range(rank, len(prompts), world_size))
    todo = []
    already_done = 0
    for prompt_index in my_prompt_indices:
        missing_offsets = []
        for sample_offset in range(args.num_samples_per_prompt):
            save_path = get_save_path(
                args.output_dir,
                prompt_index,
                sample_offset,
                args.num_samples_per_prompt,
            )
            if os.path.exists(save_path):
                already_done += 1
            else:
                missing_offsets.append(sample_offset)

        if missing_offsets:
            todo.append((prompt_index, missing_offsets))

    if rank == 0:
        print(f"Prompts for rank 0: {len(my_prompt_indices)}")
        print(f"[Resume] Rank 0: {already_done} samples already done, {len(todo)} prompts remaining")

    if not todo:
        if rank == 0:
            print("[Resume] All expected images already exist. Nothing to generate.")
        barrier()
        cleanup_distributed()
        return

    if rank == 0:
        print(f"[Rank {rank}] Loading Janus model from {args.model_path}")
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = model.to(torch.bfloat16).to(device).eval()

    if args.lora_ckpt_path is not None:
        model.language_model.load_adapter(args.lora_ckpt_path)
        model.language_model.eval().to(device)
        if rank == 0:
            print(f"Loaded LoRA weights from {args.lora_ckpt_path}")

    barrier()

    disable_tqdm = rank != 0
    for prompt_index, missing_offsets in tqdm(
        todo,
        total=len(todo),
        desc=f"Rank {rank}",
        dynamic_ncols=True,
        disable=disable_tqdm,
    ):
        prompt = build_prompt(vl_chat_processor, prompts[prompt_index])
        sample_seeds = [args.seed + prompt_index for _ in missing_offsets]
        generators = [make_generator(device, seed) for seed in sample_seeds]

        if sample_seeds:
            set_all_seeds(sample_seeds[0])

        images = generate(
            model,
            vl_chat_processor,
            prompt,
            temperature=args.temperature,
            parallel_size=len(missing_offsets),
            cfg_weight=args.cfg_weight,
            image_token_num_per_image=args.image_token_num_per_image,
            img_size=args.img_size,
            patch_size=args.patch_size,
            generators=generators,
        )

        for image, sample_offset, sample_seed in zip(images, missing_offsets, sample_seeds):
            save_path = get_save_path(
                args.output_dir,
                prompt_index,
                sample_offset,
                args.num_samples_per_prompt,
            )
            image.save(save_path)
            if rank == 0 and args.num_samples_per_prompt > 1:
                print(f"Saved {save_path} (seed={sample_seed})")

    barrier()

    if rank == 0:
        print(f"Done. Images saved to {args.output_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
