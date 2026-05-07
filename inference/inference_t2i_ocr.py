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
import re
from transformers import AutoConfig, AutoTokenizer
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line, encode_img_with_paint
from generators.image_generation_generator import generate_image
from utils.prompt_utils import create_prompt_templates
from peft import PeftModel as PEFT

OCR_PROMPT_TEMPLATE = (
    "A Mathpix Markdown format with sharp, legible black text. "
    "High-resolution typography, top-down view. "
    "The text is rendered in natural left-to-right, top-to-bottom reading order. "
    "The text reads: {sentence}"
)


def sanitize_prompt_dirname(prompt_text, max_length=120):
    sanitized = re.sub(r"\s+", "_", prompt_text.strip())
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", sanitized)
    sanitized = sanitized.strip("._")
    if not sanitized:
        sanitized = "prompt"
    return sanitized[:max_length]


def find_subsequence(sequence, subsequence):
    if not subsequence or len(subsequence) > len(sequence):
        return None
    for start_idx in range(len(sequence) - len(subsequence) + 1):
        if sequence[start_idx : start_idx + len(subsequence)] == subsequence:
            return start_idx
    return None


def decode_single_token(tokenizer, token_id):
    try:
        return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode([token_id])


def build_user_prompt_token_metadata(tokenizer, input_prompt, prompt_text, full_token_ids):
    user_tag = "<user>"
    end_tag = "</user>"
    user_start = input_prompt.index(user_tag) + len(user_tag)
    user_end = input_prompt.index(end_tag, user_start)

    token_indices = []
    try:
        encoded = tokenizer(input_prompt, return_offsets_mapping=True)
        encoded_ids = encoded["input_ids"]
        token_offset = 0
        if len(encoded_ids) != len(full_token_ids):
            token_offset = find_subsequence(full_token_ids, encoded_ids)
            if token_offset is None:
                raise ValueError("Failed to align offset-mapped tokens with the full prompt tokenization.")
        offsets = encoded["offset_mapping"]
        for token_idx, (char_start, char_end) in enumerate(offsets):
            if char_end <= char_start:
                continue
            if char_end <= user_start or char_start >= user_end:
                continue
            token_indices.append(token_offset + token_idx)
    except Exception:
        token_indices = []

    if not token_indices:
        user_token_ids = tokenizer(prompt_text)["input_ids"]
        start_idx = find_subsequence(full_token_ids, user_token_ids)
        if start_idx is None:
            user_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            start_idx = find_subsequence(full_token_ids, user_token_ids)
        if start_idx is None:
            raise ValueError("Failed to locate user prompt token span inside the conditional prompt.")
        token_indices = list(range(start_idx, start_idx + len(user_token_ids)))

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
        "char_span": [user_start, user_end],
        "prompt_text": prompt_text,
    }


def build_ocr_generation_prompt(sentence, templates=None):
    if templates is None:
        templates = create_prompt_templates()
    system_prompt = templates["image_generation"]
    user_prompt = OCR_PROMPT_TEMPLATE.format(sentence=sentence)
    input_prompt = "<system>" + system_prompt + "</system>" + "<user>" + user_prompt + "</user>"
    uncon_prompt = "<system>" + system_prompt + "</system>" + "<user>" + "<uncondition>" + "</user>"
    return input_prompt, uncon_prompt, user_prompt


def main():
    parser = argparse.ArgumentParser(description="Text-to-image inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--prompt_files", type=str, default=None, help="TXT file containing one sentence per line")
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
    parser.add_argument("--vae_ckpt", type=str, default="./vae_ckpt", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="results_text_to_image", help="Output directory")
    parser.add_argument("--use-cache", action='store_true', help="Enable caching for faster inference")
    parser.add_argument("--cache_ratio", type=float, default=0.9, help="Ratio of reused tokens, in (0,1); the higher the faster")
    parser.add_argument("--warmup_ratio", type=float, default=0.3, help="Warmup ratio for caching, in [0,1); the lower the faster")
    parser.add_argument("--refresh_interval", type=int, default=5, help="Refresh all cache every `refresh_interval` steps, in (1, timesteps-int(warmup_ratio*timesteps)-1]; the higher the faster")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples per prompt")
    parser.add_argument("--save-attention-maps", dest="save_attention_maps", action="store_true", help="Save conditional attention maps during inference")
    parser.add_argument("--no-save-attention-maps", dest="save_attention_maps", action="store_false", help="Disable conditional attention map saving during inference")
    parser.add_argument("--attention-direction", type=str, choices=["t2i", "i2t", "both"], default="t2i", help="Attention map direction to save")
    parser.add_argument("--manual-attention", dest="manual_attention", action="store_true", help="Force manual attention path even when not saving attention maps")
    parser.add_argument("--no-manual-attention", dest="manual_attention", action="store_false", help="Disable forced manual attention path")
    
    # LoRA Arguments
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path")
    
    parser.set_defaults(save_attention_maps=True)
    parser.set_defaults(manual_attention=False)
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
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
    )
    
    if args.lora_ckpt_path:
        print(f"[INFO] Loading LoRA from {args.lora_ckpt_path}")
        model.load_adapter(args.lora_ckpt_path)
    if hasattr(model, "set_manual_attention"):
        model.set_manual_attention(args.manual_attention)
        print(f"[INFO] manual_attention={args.manual_attention}")
        
    
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
                "tag": "ocr-text-to-image",
                "sentence": args.prompt,
                "include": [],
                "exclude": []
            }
        })
    
    if args.prompt_files:
        print(f"Reading prompts from {args.prompt_files}")
        with open(args.prompt_files, 'r', encoding='utf-8') as f:
            for line in f:
                sentence = line.strip()
                if sentence:
                    prompt_data_list.append({
                        "prompt": sentence,
                        "metadata": {
                            "tag": "ocr-text-to-image",
                            "sentence": sentence,
                        }
                    })
    
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

    for i, prompt_data in enumerate(prompt_data_list):
        sentence = prompt_data['prompt']
        print(f"\nProcessing sentence [{i+1}/{len(prompt_data_list)}]: {sentence}")
        
        # Prepare structured output directory
        prompt_dir = os.path.join(args.output_dir, sanitize_prompt_dirname(sentence))
        samples_dir = os.path.join(prompt_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        
        # Save metadata
        with open(os.path.join(prompt_dir, "metadata.jsonl"), 'w') as mf:
            json.dump(prompt_data['metadata'], mf, indent=2)

        # Generate prompts using utility function
        input_prompt, uncon_prompt, prompt_text = build_ocr_generation_prompt(sentence, templates)

        # build initial sequence
        con_prompt_token = tokenizer(input_prompt)["input_ids"]
        uncon_prompt_token = tokenizer(uncon_prompt)["input_ids"]
        user_prompt_meta = build_user_prompt_token_metadata(
            tokenizer,
            input_prompt,
            prompt_text,
            con_prompt_token,
        )
        
        prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=device).unsqueeze(0)
        uncon_ids = torch.tensor(uncon_prompt_token, device=device).unsqueeze(0)

        # image satrt index
        code_start = len(con_prompt_token) + 2 
        image_key_indices = [
            code_start + offset
            for offset, token_id in enumerate(img_mask_token)
            if token_id != NEW_LINE
        ]

        if args.save_attention_maps:
            with open(os.path.join(prompt_dir, "user_prompt_tokens.json"), "w") as tf:
                json.dump(
                    {
                        "prompt_text": prompt_text,
                        "sentence": sentence,
                        "query_indices": user_prompt_meta["query_indices"],
                        "token_ids": user_prompt_meta["token_ids"],
                        "token_pieces": user_prompt_meta["token_pieces"],
                        "token_decoded": user_prompt_meta["token_decoded"],
                        "char_span": user_prompt_meta["char_span"],
                        "grid_size": [token_grid_height, token_grid_width],
                    },
                    tf,
                    ensure_ascii=False,
                    indent=2,
                )
        
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
            generator = torch.Generator(device=device)
            generator.manual_seed(current_seed)
            attention_capture = None
            if args.save_attention_maps:
                sample_attention_dir = os.path.join(prompt_dir, "attentions", f"{sample_idx:04d}")
                os.makedirs(sample_attention_dir, exist_ok=True)
                attention_capture = {
                    "output_dir": sample_attention_dir,
                    "query_indices": user_prompt_meta["query_indices"],
                    "query_token_ids": user_prompt_meta["token_ids"],
                    "query_token_pieces": user_prompt_meta["token_pieces"],
                    "query_token_decoded": user_prompt_meta["token_decoded"],
                    "image_key_indices": image_key_indices,
                    "grid_size": [token_grid_height, token_grid_width],
                    "prompt_text": prompt_text,
                    "sample_idx": sample_idx,
                    "direction": args.attention_direction,
                }

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
                generator=generator,
                attention_capture=attention_capture,
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
