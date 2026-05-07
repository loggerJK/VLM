# -*- coding: utf-8 -*-
"""
OCR understanding inference script.

Flow: image -> model generates text.
Supports single-GPU (python) and multi-GPU (torchrun) execution.
Supports resume: each rank writes generation results to a per-rank JSONL file.
"""
import os
import sys
import argparse
import json
import gc
import random
import re
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from diffusers import VQModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import SPECIAL_TOKENS, PROMPT_TEMPLATES
from model import LLaDAForMultiModalGeneration
from utils.image_utils import (
    generate_crop_size_list,
    var_edge_pad,
    add_break_line,
    encode_img_with_breaks_fixed,
)
from generators.text_understanding_generator import generate_text_understanding
from utils.prompt_utils import generate_multimodal_understanding_prompt

# Special Tokens
MASK = SPECIAL_TOKENS["mask_token"]
NEW_LINE = SPECIAL_TOKENS["newline_token"]
BOA = SPECIAL_TOKENS["answer_start"]
EOA = SPECIAL_TOKENS["answer_end"]
BOI = SPECIAL_TOKENS["boi"]
EOI = SPECIAL_TOKENS["eoi"]
UNDERSTANDING_PROMPT_TEMPLATE = PROMPT_TEMPLATES["text_understanding"]

torch.set_grad_enabled(False)


def sanitize_prompt_dirname(prompt_text, max_length=120):
    sanitized = re.sub(r"\s+", "_", prompt_text.strip())
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", sanitized)
    sanitized = sanitized.strip("._")
    if not sanitized:
        sanitized = "prompt"
    return sanitized[:max_length]


def decode_single_token(tokenizer, token_id):
    try:
        return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode([token_id])


def extract_answer_query_metadata(tokenizer, generated_ids, code_start, answer_end_token_id):
    generated_ids = generated_ids.tolist()
    answer_ids = generated_ids[code_start:]
    if answer_end_token_id in answer_ids:
        valid_len = answer_ids.index(answer_end_token_id)
    else:
        valid_len = len(answer_ids)

    query_token_ids = answer_ids[:valid_len]
    query_indices = list(range(code_start, code_start + valid_len))
    query_token_pieces = tokenizer.convert_ids_to_tokens(query_token_ids) if hasattr(tokenizer, "convert_ids_to_tokens") else [
        decode_single_token(tokenizer, token_id) for token_id in query_token_ids
    ]
    query_token_decoded = [decode_single_token(tokenizer, token_id) for token_id in query_token_ids]
    return {
        "query_indices": query_indices,
        "query_token_ids": query_token_ids,
        "query_token_pieces": query_token_pieces,
        "query_token_decoded": query_token_decoded,
    }


def load_done_indices(jsonl_path):
    """Load set of already-processed indices from a JSONL file."""
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                done.add(item["index"])
    return done


def append_jsonl(jsonl_path, record):
    """Append a single record to a JSONL file with immediate flush."""
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


def main():
    parser = argparse.ArgumentParser(description="OCR Understanding Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Base model checkpoint path")
    parser.add_argument("--vae_ckpt", type=str, default="Alpha-VLLM/Lumina-DiMOO", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="ocr_understanding_results", help="Output directory")
    parser.add_argument("--steps", type=int, default=128, help="Generation steps")
    parser.add_argument("--gen_length", type=int, default=512, help="Generation length")
    parser.add_argument("--block_length", type=int, default=128, help="Block length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples (None=all)")
    parser.add_argument("--dataset_path", type=str, default="Jiwon-Kang/Llama-Nemotron-VLM-Dataset-v1-OCR4")
    parser.add_argument("--save-attention-maps", dest="save_attention_maps", action="store_true", help="Save query=text, key=image attention maps for generated OCR answers")
    parser.add_argument("--no-save-attention-maps", dest="save_attention_maps", action="store_false", help="Disable attention-map saving")
    parser.add_argument("--attention-save-step-interval", type=int, default=8, help="Save attention maps every N denoising steps")
    parser.add_argument("--manual-attention", dest="manual_attention", action="store_true", help="Force manual attention path")
    parser.add_argument("--no-manual-attention", dest="manual_attention", action="store_false", help="Disable forced manual attention path")
    parser.set_defaults(save_attention_maps=False, manual_attention=False)
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
        print("Distributed environment not detected. Running in single process mode.")

    set_all_seeds(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    # Resume: load already-processed indices for this rank
    jsonl_path = os.path.join(args.output_dir, f"results_rank{rank}.jsonl")
    done_indices = load_done_indices(jsonl_path)
    if done_indices:
        print(f"[Rank {rank}] Resuming: {len(done_indices)} samples already processed, skipping them.")

    # Load Dataset (streaming)
    print(f"[Rank {rank}] Loading dataset: {args.dataset_path}...")
    dataset = load_dataset(args.dataset_path, split="validation", streaming=True)

    # Load Models
    print(f"[Rank {rank}] Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.vae_ckpt, trust_remote_code=True)

    device_map = {"": local_rank} if world_size > 1 else "cuda"
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map=device_map,
    )
    if args.lora_ckpt_path:
        print(f"[Rank {rank}] Loading LoRA weights from {args.lora_ckpt_path}...")
        model.load_adapter(args.lora_ckpt_path)
        
    model = model.to(device)
    if hasattr(model, "set_manual_attention"):
        model.set_manual_attention(args.manual_attention)
        print(f"[Rank {rank}] manual_attention={args.manual_attention}")
    model.eval()
    model.requires_grad_(False)

        
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae", torch_dtype=torch.bfloat16).to(device)
    vqvae.eval()
    vqvae.requires_grad_(False)

    # Inference loop
    print(f"[Rank {rank}] Starting OCR understanding inference...")
    # Pre-compute local indices for this rank (handles remainder correctly)
    local_indices = list(range(rank, args.num_samples, world_size))
    local_indices_set = set(local_indices)
    max_local_idx = max(local_indices) if local_indices else -1

    for i, item in tqdm(enumerate(dataset), total=args.num_samples, desc=f"Rank {rank}"):
        if i > max_local_idx:
            break
        if i not in local_indices_set:
            continue

        # Resume: skip already-processed indices
        if i in done_indices:
            continue

        # Deterministic seed per sample
        set_all_seeds(args.seed)

        image = item['image']
        question = item.get('question', "Extract all text from the image.")
        answer_gt = item.get('answer', "")

        # Image Preprocessing (edge padding, matching train_unified.py validate_ocr)
        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
        image_processed = var_edge_pad(image, crop_size_list=crop_size_list, pad_mode='edge')

        # Encode Image
        input_img_token, (H, W) = encode_img_with_breaks_fixed(image_processed, vqvae)
        img_token = [BOI] + add_break_line(input_img_token[1:-1], H, W, new_number=NEW_LINE) + [EOI]

        # Build prompt (matching validate_ocr pattern)
        instruction = "<system>" + UNDERSTANDING_PROMPT_TEMPLATE + "</system>" + "<user>" + question + "</user>"
        input_ids_raw = tokenizer(instruction)['input_ids']

        # Insert image tokens
        input_token = input_ids_raw[:-1] + img_token + input_ids_raw[-1:]
        image_key_indices = list(range(len(input_ids_raw[:-1]), len(input_ids_raw[:-1]) + len(img_token)))

        # Prepare generation input
        code_start = len(input_token) + 1
        input_token = input_token + [BOA] + [MASK] * args.gen_length
        base_input_ids = torch.tensor(input_token, device=device).unsqueeze(0)

        # Generate
        out = generate_text_understanding(
            model, base_input_ids.clone(),
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=0.0,
            remasking='low_confidence',
            code_start=code_start,
        )

        pred_text = tokenizer.batch_decode(
            out[:, code_start:], skip_special_tokens=True
        )[0].replace("</answer>", "").strip()

        if args.save_attention_maps:
            answer_query_meta = extract_answer_query_metadata(
                tokenizer,
                out[0],
                code_start,
                EOA,
            )
            if answer_query_meta["query_indices"]:
                prompt_dir = os.path.join(args.output_dir, sanitize_prompt_dirname(question[:120]))
                sample_attention_dir = os.path.join(prompt_dir, "attentions", f"{i:05d}")
                os.makedirs(sample_attention_dir, exist_ok=True)
                with open(os.path.join(sample_attention_dir, "answer_tokens.json"), "w", encoding="utf-8") as tf:
                    json.dump(
                        {
                            "question": question,
                            "pred_text": pred_text,
                            "query_indices": answer_query_meta["query_indices"],
                            "query_token_ids": answer_query_meta["query_token_ids"],
                            "query_token_pieces": answer_query_meta["query_token_pieces"],
                            "query_token_decoded": answer_query_meta["query_token_decoded"],
                            "image_key_indices": image_key_indices,
                            "grid_size": [H, W],
                        },
                        tf,
                        ensure_ascii=False,
                        indent=2,
                    )

                attention_capture = {
                    "output_dir": sample_attention_dir,
                    "query_indices": answer_query_meta["query_indices"],
                    "query_token_ids": answer_query_meta["query_token_ids"],
                    "query_token_pieces": answer_query_meta["query_token_pieces"],
                    "query_token_decoded": answer_query_meta["query_token_decoded"],
                    "image_key_indices": image_key_indices,
                    "grid_size": [H, W],
                    "prompt_text": pred_text,
                    "sample_idx": i,
                    "direction": "t2i",
                    "step": 0,
                }
                set_all_seeds(args.seed)
                _ = generate_text_understanding(
                    model,
                    base_input_ids.clone(),
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    cfg_scale=0.0,
                    remasking='low_confidence',
                    code_start=code_start,
                    attention_capture=attention_capture,
                    attention_capture_interval=args.attention_save_step_interval,
                )

        print(f"[Rank {rank}][{i+1}] ==============Prediction==============\n {pred_text}")
        print(f"[Rank {rank}][{i+1}] ==============Ground Truth==============\n {answer_gt}")
        print("=" * 50)

        record = {
            "index": i,
            "question": question,
            "gt_answer": answer_gt,
            "pred_answer": pred_text,
        }
        append_jsonl(jsonl_path, record)

        # Prevent VRAM accumulation
        gc.collect()
        torch.cuda.empty_cache()

    # Synchronize all ranks before finishing
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        print(f"Inference completed. Per-rank results are stored under {args.output_dir}/results_rank*.jsonl")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
