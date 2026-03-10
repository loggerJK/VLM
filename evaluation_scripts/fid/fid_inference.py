# -*- coding: utf-8 -*-
"""
FID evaluation inference script — generates one image per COCO caption.
Supports distributed inference via torchrun.
"""
import os
import json
import argparse
import time
import torch
import random
import torch.distributed as dist
from transformers import AutoTokenizer
from PIL import Image
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line
from generators.image_generation_generator import generate_image
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates

torch.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser(description="FID evaluation — text-to-image inference")
    parser.add_argument("--checkpoint", type=str, default='Alpha-VLLM/Lumina-DiMOO', help="Model checkpoint path")
    parser.add_argument("--prompt_files", type=str, default="captions_val2014.json", help="COCO captions JSON file (relative to script dir or absolute)")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of timesteps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=65513, help="Random seed")
    parser.add_argument("--vae_ckpt", type=str, default="Alpha-VLLM/Lumina-DiMOO", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="results_fid", help="Output directory")
    parser.add_argument("--use-cache", action='store_true', help="Enable caching for faster inference")
    parser.add_argument("--cache_ratio", type=float, default=0.9, help="Ratio of reused tokens")
    parser.add_argument("--warmup_ratio", type=float, default=0.3, help="Warmup ratio for caching")
    parser.add_argument("--refresh_interval", type=int, default=5, help="Refresh all cache every N steps")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per prompt (1 for FID)")
    parser.add_argument("--max_prompts", type=int, default=1000, help="Maximum number of prompts to process")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path")

    args = parser.parse_args()

    if args.num_samples != 1:
        raise ValueError("Lumina FID inference only supports --num_samples=1")

    # ------------------------------------------------------------------ #
    #                          DDP Setup                                  #
    # ------------------------------------------------------------------ #
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if rank == 0:
            print(f"Initialized DDP with world_size={world_size}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running in single process mode on {device}")

    # ------------------------------------------------------------------ #
    #                        Special tokens                               #
    # ------------------------------------------------------------------ #
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]
    EOA = SPECIAL_TOKENS["answer_end"]
    BOI = SPECIAL_TOKENS["boi"]
    EOI = SPECIAL_TOKENS["eoi"]

    if args.seed != 0:
        setup_seed(args.seed)

    # Create output directory (rank 0 only, then barrier)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #                        Load model                                   #
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(args.vae_ckpt, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map=None,
    ).to(device)

    if args.lora_ckpt_path:
        if rank == 0:
            print(f"[INFO] Loading LoRA from {args.lora_ckpt_path}")
        model.load_adapter(args.lora_ckpt_path)
        model = model.to(device)

    model.eval()

    if world_size > 1:
        dist.barrier()

    # ------------------------------------------------------------------ #
    #                        Load VQ-VAE                                  #
    # ------------------------------------------------------------------ #
    height = args.height
    width = args.width

    from diffusers import VQModel
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)
    vqvae.eval()
    vqvae.requires_grad_(False)

    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(height, width)

    if rank == 0:
        print(f"Generate image size: {height}x{width}")
        print(f"VQ sequence length: {seq_len}, newline_every: {newline_every}")

    templates = create_prompt_templates()

    # ------------------------------------------------------------------ #
    #                        Load captions                                #
    # ------------------------------------------------------------------ #
    prompt_file_path = args.prompt_files
    if not os.path.isabs(prompt_file_path):
        prompt_file_path = os.path.join(os.path.dirname(__file__), prompt_file_path)

    if rank == 0:
        print(f"Reading captions from {prompt_file_path}")

    with open(prompt_file_path, 'r') as f:
        val_caption = json.load(f)

    prompt_data_list = []
    for item in val_caption['annotations'][:args.max_prompts]:
        caption = item['caption']
        assert caption is not None and isinstance(caption, str), f"Invalid caption: {caption}"
        prompt_data_list.append(caption)

    if rank == 0:
        print(f"Total prompts: {len(prompt_data_list)}")

    # ------------------------------------------------------------------ #
    #                   Build image mask template                         #
    # ------------------------------------------------------------------ #
    img_mask_token = add_break_line(
        [MASK] * seq_len, token_grid_height, token_grid_width, new_number=NEW_LINE
    )
    img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]

    # ------------------------------------------------------------------ #
    #                   Split prompts by rank                             #
    # ------------------------------------------------------------------ #
    my_prompts = prompt_data_list[rank::world_size]

    # ------------------------------------------------------------------ #
    #                   Resume: count existing images                     #
    # ------------------------------------------------------------------ #
    already_done = 0
    todo_indices = []
    for i in range(len(my_prompts)):
        real_idx = rank + i * world_size
        save_path = os.path.join(args.output_dir, f"{real_idx:05d}.png")
        if os.path.exists(save_path):
            already_done += 1
        else:
            todo_indices.append(i)

    if rank == 0:
        print(f"Prompts for rank 0: {len(my_prompts)} (total across all ranks: {len(prompt_data_list)})")
        print(f"[Resume] Rank 0: {already_done} already done, {len(todo_indices)} remaining")

    if len(todo_indices) == 0:
        if rank == 0:
            print("[Resume] All images already exist — nothing to generate.")
        return

    from tqdm import tqdm
    disable_tqdm = (rank != 0)

    for i in tqdm(
        todo_indices,
        total=len(todo_indices),
        dynamic_ncols=True,
        desc=f"Rank {rank}",
        disable=disable_tqdm,
    ):
        caption = my_prompts[i]
        # Global index: my_prompts[i] == prompt_data_list[rank + i * world_size]
        real_idx = rank + i * world_size

        # Build prompts
        input_prompt, uncon_prompt = generate_text_to_image_prompt(caption, templates)

        con_prompt_token = tokenizer(input_prompt)["input_ids"]
        uncon_prompt_token = tokenizer(uncon_prompt)["input_ids"]

        prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=device).unsqueeze(0)
        uncon_ids = torch.tensor(uncon_prompt_token, device=device).unsqueeze(0)

        code_start = len(con_prompt_token) + 2  # +2 for BOA, BOI

        # Seed per sample: base_seed + global_index
        current_seed = args.seed + real_idx
        setup_seed(current_seed)

        generator = torch.Generator(device=device)
        generator.manual_seed(current_seed)

        # Output path
        filename = f"{real_idx:05d}.png"
        save_path = os.path.join(args.output_dir, filename)
        if os.path.exists(save_path):
            print(f"[Rank {rank}] Skipping {filename} (already exists)")
            continue  # Skip if already exists (double check)

        # Generate VQ tokens
        vq_tokens = generate_image(
            model,
            prompt_ids.clone(),
            seq_len=seq_len,
            newline_every=newline_every,
            timesteps=args.timesteps,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            uncon_ids=uncon_ids,
            code_start=code_start,
            use_cache=args.use_cache,
            cache_ratio=args.cache_ratio,
            refresh_interval=args.refresh_interval,
            warmup_ratio=args.warmup_ratio,
            generator=generator,
            disable_tqdm=True,
        )

        # Decode and save
        out_img = decode_vq_to_image(
            vq_tokens, save_path,
            vae_ckpt=args.vae_ckpt,
            image_height=height,
            image_width=width,
            vqvae=vqvae,
        )
        out_img.save(save_path)

    # ------------------------------------------------------------------ #
    #                          Cleanup                                    #
    # ------------------------------------------------------------------ #

    if rank == 0:
        print(f"Done. Images saved to {args.output_dir}")


if __name__ == '__main__':
    main()
