# -*- coding: utf-8 -*-
"""
OCR Generation Evaluation — Stage 1: BAGEL Image Generation

Reads prompts from the Abirate/english_quotes dataset and generates images
using the BAGEL model.  Writes per-rank meta JSONL + PNG files.

Run with:  conda run -n bagel torchrun --nproc_per_node=N stage1_generate_images.py ...
"""
import os
import sys
import argparse
import json
import gc
import glob
import random
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import set_seed
from accelerate import init_empty_weights
from safetensors.torch import load_file

# Add bagel_train root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache

torch.set_grad_enabled(False)


def append_jsonl(jsonl_path, record):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


def move_generation_input_to_device(generation_input, device):
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def generate_image_bagel(gen_model, vae_model, tokenizer, new_token_ids,
                         prompt, device,
                         num_timesteps=50, cfg_scale=4.0, cfg_interval=None,
                         cfg_renorm_min=0.0, timestep_shift=3.0, resolution=512):
    """Generate a single image using BAGEL's flow-matching generation pipeline."""
    if cfg_interval is None:
        cfg_interval = [0, 1.0]

    num_images = 1
    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0] * num_images
    new_rope = [0] * num_images

    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=[prompt] * num_images,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[(resolution, resolution)] * num_images,
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    cfg_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_newlens = [0] * num_images
    cfg_new_rope = [0] * num_images

    generation_input_cfg = gen_model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope,
        image_sizes=[(resolution, resolution)] * num_images,
    )
    generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = gen_model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift,
                cfg_text_past_key_values=cfg_past_key_values,
                cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                **generation_input,
            )

    image_list = []
    for latent in unpacked_latent:
        latent = latent.reshape(1, resolution // 16, resolution // 16, 2, 2, 16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, resolution // 8, resolution // 8)
        image = vae_model.decode(latent.to(device))
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        tmpimage = Image.fromarray(tmpimage)
        image_list.append(tmpimage)

    return image_list


def main():
    parser = argparse.ArgumentParser(description="OCR Generation — Stage 1: BAGEL Image Generation")
    parser.add_argument("--model_path", type=str, required=True, help="BAGEL model path")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path (optional)")
    parser.add_argument("--output_dir", type=str, default="ocr_generation_results", help="Output directory")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--img_size", type=int, default=512, help="Generated image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples (0=all)")
    parser.add_argument("--dataset_path", type=str, default="Abirate/english_quotes")
    parser.add_argument("--max_latent_size", type=int, default=64, help="Max latent size for VAE")
    args = parser.parse_args()

    # Initialize Distributed
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_all_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "generated_images")
    os.makedirs(images_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    # ===== Stage 1: Image Generation =====
    print(f"[Rank {rank}] ===== Stage 1: Image Generation =====")

    # Load dataset
    dataset = load_dataset(args.dataset_path, split="train", streaming=False)
    args.num_samples = min(args.num_samples, len(dataset)) if args.num_samples else len(dataset)
    dataset = dataset.select(list(range(args.num_samples)))
    print(f"[Rank {rank}] Loaded {args.num_samples} samples from dataset {args.dataset_path}")

    meta_jsonl_path = os.path.join(args.output_dir, f"meta_rank{rank}.jsonl")
    # Load existing meta from ALL rank files to handle GPU count changes on resume
    existing_meta = {}
    for mf in glob.glob(os.path.join(glob.escape(args.output_dir), "meta_rank*.jsonl")):
        with open(mf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                existing_meta[item["index"]] = item

    # Load BAGEL model with VAE (generation mode)
    print(f"[Rank {rank}] Loading BAGEL model (generation mode)...")

    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Load weights directly to rank's GPU — no CPU intermediate copy
    state_dict = load_file(os.path.join(args.model_path, "ema.safetensors"), device=str(device))
    msg = model.load_state_dict(state_dict, strict=False, assign=True)
    if rank == 0:
        print(msg)
    del state_dict

    model = model.to(torch.bfloat16).to(device).eval()

    # LoRA loading (applied to entire Bagel model)
    if args.lora_ckpt_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_ckpt_path, is_trainable=False, torch_device='cpu')
        model = model.to(torch.bfloat16).to(device).eval()
        if rank == 0:
            print(f"Loaded LoRA weights from {args.lora_ckpt_path}")

    vae_model = vae_model.to(device).eval()
    gen_model = model

    # Generation parameters
    cfg_scale = args.cfg_scale
    cfg_interval = [0, 1.0]
    timestep_shift = 3.0
    num_timesteps = 50
    cfg_renorm_min = 0.0

    processed = 0
    for i, sample in tqdm(enumerate(dataset), total=args.num_samples, desc=f"Rank {rank} Stage1"):
        if i % world_size != rank:
            continue

        answer_gt = sample.get('quote', "").replace("\n", " ").strip().replace('\u201c', '"').replace('\u201d', '"')[:120]
        if not answer_gt or not answer_gt.strip():
            continue

        save_path = os.path.join(images_dir, f"{i:05d}.png")
        if os.path.exists(save_path) and i in existing_meta:
            processed += 1
            continue

        set_all_seeds(args.seed)

        prompt_text = (
            f"A Mathpix Markdown format with sharp, legible black text. "
            f"High-resolution typography, top-down view. "
            f"The text is rendered in natural left-to-right, top-to-bottom reading order. "
            f"The text reads:\n"
            f"{answer_gt}"
        )
        prompt_text_save_path = save_path.replace(".png", "_prompt.txt")
        with open(prompt_text_save_path, "w") as f:
            f.write(prompt_text)

        # BAGEL image generation
        samples = generate_image_bagel(
            gen_model, vae_model, tokenizer, new_token_ids,
            prompt=prompt_text,
            device=device,
            num_timesteps=num_timesteps,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            timestep_shift=timestep_shift,
            resolution=args.img_size,
        )
        samples[0].save(save_path)

        meta_record = {"index": i, "gt_answer": answer_gt, "prompt": prompt_text, "image_path": save_path}
        append_jsonl(meta_jsonl_path, meta_record)
        print(f"[Rank {rank}][{i+1}] Image saved: {save_path}")

        processed += 1
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[Rank {rank}] Stage 1 complete. Generated {processed} images.")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
