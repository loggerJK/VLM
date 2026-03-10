# -*- coding: utf-8 -*-
"""
Text-to-image inference script
"""
import os
import json
import argparse
import time
import torch
import random
from transformers import AutoConfig, AutoTokenizer
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line, encode_img_with_paint
from generators.image_generation_generator import generate_image
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates
from peft import PeftModel as PEFT
# No grad
torch.set_grad_enabled(False)

def main():
    parser = argparse.ArgumentParser(description="Text-to-image inference")
    parser.add_argument("--checkpoint", type=str, default='Alpha-VLLM/Lumina-DiMOO', help="Fine-tuned checkpoint path")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--prompt_files", type=str, default=None, help="JSONL file containing prompts")
    parser.add_argument("--painting_mode", type=str, default=None, help="Inpainting for image-inpainting task & outpainting for imahe-extrapolation task")
    parser.add_argument("--painting_image", type=str, default=None, help="Inpainting & outpainting image path")
    parser.add_argument("--mask_h_ratio", type=float, default=1, help="Height ratio for mask region of In/Out paint task")
    parser.add_argument("--mask_w_ratio", type=float, default=0.2, help="Width ratio for mask region of In/Out paint task")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of timesteps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--vae_ckpt", type=str, default="Alpha-VLLM/Lumina-DiMOO", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="results_text_to_image", help="Output directory")
    parser.add_argument("--use-cache", action='store_true', help="Enable caching for faster inference")
    parser.add_argument("--cache_ratio", type=float, default=0.9, help="Ratio of reused tokens, in (0,1); the higher the faster")
    parser.add_argument("--warmup_ratio", type=float, default=0.3, help="Warmup ratio for caching, in [0,1); the lower the faster")
    parser.add_argument("--refresh_interval", type=int, default=5, help="Refresh all cache every `refresh_interval` steps, in (1, timesteps-int(warmup_ratio*timesteps)-1]; the higher the faster")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples per prompt")
    
    # LoRA Arguments
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path")
    
    args = parser.parse_args()

    if args.prompt is None and args.prompt_files is None:
        parser.error("At least one of --prompt or --prompt_files must be provided.")
    
    # Special tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
    EOA = SPECIAL_TOKENS["answer_end"]    # End of Answer
    BOI = SPECIAL_TOKENS["boi"]           # Begin of Image
    EOI = SPECIAL_TOKENS["eoi"]           # End of Image

    # Set initial Random seed if provided, though we will set it again per sample
    if args.seed != 0:
        setup_seed(args.seed)
    
    # Create Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.vae_ckpt, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
    )
    
    if args.lora_ckpt_path:
        print(f"[INFO] Loading LoRA from {args.lora_ckpt_path}")
        model.load_adapter(args.lora_ckpt_path)

    model.eval()

    # Initial image parameters
    if args.painting_mode:
        img = Image.open(args.painting_image)
        width, height = img.size
    else:
        height = args.height
        width = args.width

    # Load VQ-VAE
    from diffusers import VQModel
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)
    vqvae.eval()
    vqvae.requires_grad_(False)
    # Calculate VQ parameters
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(height, width)
    
    print(f"Generate image size: {height}x{width}")
    print(f"Calculated VQ sequence length: {seq_len}")
    print(f"Tokens per line (newline_every): {newline_every}")
    
    # Get prompt templates
    templates = create_prompt_templates()

    prompt_data_list = []
    if args.prompt:
        # Default metadata for single prompt
        prompt_data_list.append({
            "prompt": args.prompt,
            "metadata": {
                "tag": "text-to-image",
                "prompt": args.prompt,
                "include": [],
                "exclude": []
            }
        })
    
    if args.prompt_files:
        print(f"Reading prompts from {args.prompt_files}")
        with open(args.prompt_files, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        prompt_str = data.get("prompt") or data.get("user_prompt")
                        if prompt_str:
                            prompt_data_list.append({
                                "prompt": prompt_str,
                                "metadata": data
                            })
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line: {line}")
    
    if not prompt_data_list:
        print("No prompts found.")
        return

    # build image mask predition (constant if painting mode is same or None)
    if args.painting_mode:
        img_mask_token, img_vis = encode_img_with_paint(img, vqvae=vqvae, mask_h_ratio=args.mask_h_ratio, mask_w_ratio=args.mask_w_ratio, mask_mode=args.painting_mode)
    else:
        img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number = NEW_LINE)
    img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]

    print(f"Total prompts to process: {len(prompt_data_list)}")

    from tqdm import tqdm
    for i, prompt_data in tqdm(enumerate(prompt_data_list), total=len(prompt_data_list), dynamic_ncols=True):
        prompt_text = prompt_data['prompt']
        print(f"\nProcessing prompt [{i+1}/{len(prompt_data_list)}]: {prompt_text}")
        
        # Prepare structured output directory
        prompt_dir = os.path.join(args.output_dir, f"{i:05d}")
        samples_dir = os.path.join(prompt_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        
        # Save metadata
        with open(os.path.join(prompt_dir, "metadata.jsonl"), 'w') as mf:
            json.dump(prompt_data['metadata'], mf, indent=2)

        # Generate prompts using utility function
        input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt_text, templates)

        # build initial sequence
        con_prompt_token = tokenizer(input_prompt)["input_ids"]
        uncon_prompt_token = tokenizer(uncon_prompt)["input_ids"]
        
        prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=device).unsqueeze(0)
        uncon_ids = torch.tensor(uncon_prompt_token, device=device).unsqueeze(0)

        # image satrt index
        code_start = len(con_prompt_token) + 2 
        
        generated_images = [] # Store images for grid
        
        # Loop for multiple samples with different seeds
        for sample_idx in range(args.num_samples):
            # Determine seed for this sample
            if args.seed != 0:
                current_seed = args.seed + sample_idx
            else:
                # If seed is 0 (random), generate a random seed for this sample
                current_seed = random.randint(1, 2**32 - 1)
            
            setup_seed(current_seed)
            print(f"  > Sample [{sample_idx+1}/{args.num_samples}] Seed: {current_seed}")
            
            # Create a generator for reproducibility
            generator = torch.Generator(device='cuda')
            generator.manual_seed(current_seed)

            # Generate VQ tokens
            start_time = time.time()
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
                generator=generator
            )
            
            # Filename: 0000.png, 0001.png...
            filename = f"{sample_idx:04d}.png"
            save_path = os.path.join(samples_dir, filename)
            
            # Decode VQ codes to PNG and save
            out_img = decode_vq_to_image(
                vq_tokens, save_path, 
                vae_ckpt=args.vae_ckpt, 
                image_height=height, 
                image_width=width,
                vqvae=vqvae
            )
            
            final_img = out_img
            
            if args.painting_mode:
                w1, h1 = img_vis.size
                w2, h2 = out_img.size
                canvas = Image.new("RGB", (w1 + w2, max(h1, h2)), "white")
                canvas.paste(img_vis, (0, 0))
                canvas.paste(out_img, (w1, 0))
                concat_path = save_path.replace(".png", "_concat.png")
                canvas.save(concat_path)
                final_img = canvas # Use concatenated image for grid if in painting mode? 
                                   # Usually grid expects result. Let's use the one saved as main result.
            else:
                out_img.save(save_path)
            
            generated_images.append(final_img)
            print(f"    [✓] Saved {save_path}")

            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print(f"    Time for this sample: {elapsed_time:.2f}s")
            
        # Create horizontal grid
        if generated_images:
            try:
                # Assume all images are same size
                w, h = generated_images[0].size
                grid_width = w * len(generated_images)
                grid_img = Image.new("RGB", (grid_width, h))
                for i, img in enumerate(generated_images):
                    grid_img.paste(img, (i * w, 0))
                
                grid_path = os.path.join(prompt_dir, "grid.png")
                grid_img.save(grid_path)
                print(f"    [✓] Saved grid: {grid_path}")
            except Exception as e:
                print(f"    [!] Failed to save grid: {e}")
       

if __name__ == '__main__':
    main()