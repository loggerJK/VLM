# -*- coding: utf-8 -*-
"""
OCR Generation Evaluation Script (Multi-GPU, 2-Stage) — Text File Prompts

Evaluates model's ability to generate images containing specified text.
Flow: text from file -> model generates image -> GLM-OCR extracts text -> compare with GT

Reads prompts from a local text file (e.g. word.txt or sentence.txt).
Each line should be prefixed with "# " (e.g. "# Hello").

Supports single-GPU (python) and multi-GPU (torchrun) execution.

2-Stage pipeline to reduce peak VRAM:
  Stage 1: LLaDA + VQVAE generate images (resume via file existence check)
  Stage 2: GLM-OCR extracts text from generated images (resume via per-rank JSONL)
  Final:   Rank 0 aggregates all JSONL files -> metrics
"""
from curses import flash
import os
import sys
import argparse
import json
import gc
import glob
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText, set_seed
from diffusers import VQModel

import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.image_utils import (
    decode_vq_to_image,
    calculate_vq_params,
    add_break_line,
)
from generators.image_generation_generator import generate_image
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates
from utils.generation_utils import setup_seed
from utils.ocr_render import generate_image as generate_ocr_image
from datasets import load_dataset


# Special Tokens
MASK = SPECIAL_TOKENS["mask_token"]
NEW_LINE = SPECIAL_TOKENS["newline_token"]
BOA = SPECIAL_TOKENS["answer_start"]
EOA = SPECIAL_TOKENS["answer_end"]
BOI = SPECIAL_TOKENS["boi"]
EOI = SPECIAL_TOKENS["eoi"]

torch.set_grad_enabled(False)


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


def calculate_metrics(predictions, references, loaded_metrics=None):
    """Calculate OCR text metrics: WER, CER, METEOR, BLEU, Edit Distance, P/R/F1."""
    metrics = {}

    wer_metric = loaded_metrics.get("wer") if loaded_metrics else evaluate.load("wer")
    cer_metric = loaded_metrics.get("cer") if loaded_metrics else evaluate.load("cer")
    meteor_metric = loaded_metrics.get("meteor") if loaded_metrics else evaluate.load("meteor")

    preds_norm = [p.lower().replace("\n", " ").strip() for p in predictions]
    refs_norm = [r.lower().replace("\n", " ").strip() for r in references]

    valid_indices = [i for i, r in enumerate(refs_norm) if len(r.strip()) > 0]
    if not valid_indices:
        return {"wer": 1.0, "cer": 1.0, "meteor": 0.0, "bleu": 0.0,
                "edit_distance": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    preds_norm = [preds_norm[i] for i in valid_indices]
    refs_norm = [refs_norm[i] for i in valid_indices]

    try:
        metrics["wer"] = wer_metric.compute(predictions=preds_norm, references=refs_norm)
    except Exception:
        metrics["wer"] = 1.0

    try:
        metrics["cer"] = cer_metric.compute(predictions=preds_norm, references=refs_norm)
    except Exception:
        metrics["cer"] = 1.0

    try:
        metrics["meteor"] = meteor_metric.compute(predictions=preds_norm, references=refs_norm)["meteor"]
    except Exception:
        metrics["meteor"] = 0.0

    try:
        bleu_scores = []
        for p, r in zip(preds_norm, refs_norm):
            bleu_scores.append(sentence_bleu([r.split()], p.split(), weights=(0.5, 0.5)))
        metrics["bleu"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    except Exception:
        metrics["bleu"] = 0.0

    try:
        edit_dists = []
        for p, r in zip(preds_norm, refs_norm):
            edit_dists.append(nltk.edit_distance(p, r))
        metrics["edit_distance"] = sum(edit_dists) / len(edit_dists) if edit_dists else 0.0
    except Exception:
        metrics["edit_distance"] = 0.0

    try:
        total_p, total_r, total_f1 = 0.0, 0.0, 0.0
        for p, r in zip(preds_norm, refs_norm):
            p_words = set(p.split())
            r_words = set(r.split())
            if not p_words or not r_words:
                continue
            intersection = p_words.intersection(r_words)
            prec = len(intersection) / len(p_words) if p_words else 0.0
            rec = len(intersection) / len(r_words) if r_words else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            total_p += prec
            total_r += rec
            total_f1 += f1
        metrics["precision"] = total_p / len(preds_norm)
        metrics["recall"] = total_r / len(preds_norm)
        metrics["f1"] = total_f1 / len(preds_norm)
    except Exception:
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0

    return metrics


def extract_text_with_glmocr(image_path, ocr_model, ocr_processor):
    """Extract text from image using GLM-OCR (transformers)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]
    inputs = ocr_processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(ocr_model.device)
    inputs.pop("token_type_ids", None)
    generated_ids = ocr_model.generate(**inputs, max_new_tokens=8192)
    output_text = ocr_processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )
    return output_text.strip()


def main():
    parser = argparse.ArgumentParser(description="OCR Generation Evaluation (Text File Prompts)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Base model checkpoint path")
    parser.add_argument("--vae_ckpt", type=str, default="Alpha-VLLM/Lumina-DiMOO", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="ocr_generation_results", help="Output directory")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of generation timesteps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--height", type=int, default=512, help="Generated image height")
    parser.add_argument("--width", type=int, default=512, help="Generated image width")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples (0=all)")
    parser.add_argument("--dataset_path", type=str, default="agentlans/high-quality-english-sentences")

    parser.add_argument("--ocr_model_path", type=str, default="zai-org/GLM-OCR",
                        help="GLM-OCR model path")
    parser.add_argument("--ocr_device", type=str, default=None,
                        help="Device for GLM-OCR (e.g. 'cuda:1'). Defaults to same as generation model.")
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
    images_dir = os.path.join(args.output_dir, "generated_images")
    os.makedirs(images_dir, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    # ================================================================
    # Stage 1: Image Generation (LLaDA + VQVAE only)
    # ================================================================
    print(f"[Rank {rank}] ===== Stage 1: Image Generation =====")

    # # Load prompts from text file
    # print(f"[Rank {rank}] Loading prompts from: {args.prompt_file}...")
    # with open(args.prompt_file, "r") as f:
    #     prompts = [line.strip() for line in f]

    # total_samples = len(prompts)
    # if args.num_samples > 0:
    #     total_samples = min(args.num_samples, total_samples)
    #     prompts = prompts[:total_samples]
    
    # Load dataset 
    dataset = load_dataset(args.dataset_path, split="test", streaming=False)
    # Select Samples
    num_to_select = min(args.num_samples, len(dataset)) if args.num_samples > 0 else len(dataset)
    dataset = dataset.select(list(range(0, num_to_select)))
    args.num_samples = len(dataset)

    print(f"[Rank {rank}] Loaded {args.num_samples} samples from dataset {args.dataset_path}")

    # We need to collect (index, answer_gt) mapping for Stage 2,
    # so save a metadata JSONL during Stage 1
    meta_jsonl_path = os.path.join(args.output_dir, f"meta_rank{rank}.jsonl")
    existing_meta = {}
    if os.path.exists(meta_jsonl_path):
        with open(meta_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                existing_meta[item["index"]] = item

    # Load generation models
    print(f"[Rank {rank}] Loading LLaDA + VQVAE...")
    tokenizer = AutoTokenizer.from_pretrained(args.vae_ckpt, trust_remote_code=True)

    device_map = {"": local_rank} if world_size > 1 else "cuda"
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map=device_map, flash_attention=True
    )
    if args.lora_ckpt_path:
        print(f"[Rank {rank}] Loading LoRA weights from {args.lora_ckpt_path}...")
        model.load_adapter(args.lora_ckpt_path)

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae", torch_dtype=torch.bfloat16).to(device)
    vqvae.eval()
    vqvae.requires_grad_(False)

    # Calculate VQ parameters
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(args.height, args.width)
    if rank == 0:
        print(f"Generate image size: {args.height}x{args.width}")
        print(f"Calculated VQ sequence length: {seq_len}")
        print(f"Tokens per line (newline_every): {newline_every}")

    # Get prompt templates
    templates = create_prompt_templates()

    # Build image mask prediction tokens (constant for all samples)
    img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number=NEW_LINE)
    img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]

    # Stage 1 eval loop
    print(f"[Rank {rank}] Starting image generation...")
    processed = 0
    for i, sample in tqdm(enumerate(dataset), total=args.num_samples, desc=f"Rank {rank} Stage1"):
        if i % world_size != rank:
            continue

        answer_gt = sample.get('text', sample.get('answer', "")).replace("\n", " ").strip()[:120]

        if not answer_gt or not answer_gt.strip():
            continue

        # Truncate long answers to max 1024 tokens
        answer_gt_tokens = tokenizer.encode(answer_gt, add_special_tokens=False)
        if len(answer_gt_tokens) > 1024:
            answer_gt = tokenizer.decode(answer_gt_tokens[:1024], skip_special_tokens=True)

        save_path = os.path.join(images_dir, f"{i:05d}.png")

        # Resume: skip if image already exists
        if os.path.exists(save_path) and i in existing_meta:
            processed += 1
            continue

        # Deterministic seed per sample
        set_all_seeds(args.seed)

        # Build generation prompt (matching train_unified.py OCR gen data transform)
        prompt_text = (
            f"A Mathpix Markdown format with sharp, legible black text. "
            f"High-resolution typography, top-down view. "
            f"The text is rendered in natural left-to-right, top-to-bottom reading order. "
            # f"Use Mathpix Markdown format: tables as LaTeX, mathematical expressions in LaTeX notation, "
            # f"and keep all tables and captions at the end. "
            f"The text reads:\n"
            f"{answer_gt}"
        )
        prompt_text_save_path = save_path.replace(".png", "_prompt.txt")
        with open(prompt_text_save_path, "w") as f:
            f.write(prompt_text)

        # Generate prompt tokens
        input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt_text, templates)
        con_prompt_token = tokenizer(input_prompt)["input_ids"]
        uncon_prompt_token = tokenizer(uncon_prompt)["input_ids"]

        prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=device).unsqueeze(0)
        uncon_ids = torch.tensor(uncon_prompt_token, device=device).unsqueeze(0)

        # Image start index
        code_start = len(con_prompt_token) + 2

        # Set generator seed for reproducibility
        generator = torch.Generator(device='cuda')
        generator.manual_seed(args.seed)

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
            generator=generator,
            disable_tqdm=True,
        )

        # Decode VQ codes to image
        out_img = decode_vq_to_image(
            vq_tokens, save_path,
            vae_ckpt=args.vae_ckpt,
            image_height=args.height,
            image_width=args.width,
            vqvae=vqvae,
        )
        out_img.save(save_path)

        # Save metadata for Stage 2
        meta_record = {
            "index": i,
            "gt_answer": answer_gt,
            "prompt": prompt_text,
            "image_path": save_path,
        }
        append_jsonl(meta_jsonl_path, meta_record)

        print(f"[Rank {rank}][{i+1}] Image saved: {save_path}")

        # Prevent VRAM accumulation
        processed += 1
        gc.collect()
        torch.cuda.empty_cache()

    # Unload Stage 1 models to free VRAM
    print(f"[Rank {rank}] Stage 1 complete. Unloading LLaDA + VQVAE...")
    del model, vqvae
    gc.collect()
    torch.cuda.empty_cache()

    if world_size > 1:
        dist.barrier()

    # ================================================================
    # Stage 2: OCR Extraction (GLM-OCR only)
    # ================================================================
    print(f"[Rank {rank}] ===== Stage 2: OCR Extraction =====")

    # Load all metadata for this rank (from meta JSONL)
    meta_records = []
    if os.path.exists(meta_jsonl_path):
        seen = set()
        with open(meta_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item["index"] not in seen:
                    seen.add(item["index"])
                    meta_records.append(item)
    meta_records.sort(key=lambda x: x["index"])

    # Resume: load already-processed OCR indices
    ocr_jsonl_path = os.path.join(args.output_dir, f"ocr_results_rank{rank}.jsonl")
    done_indices = load_done_indices(ocr_jsonl_path)
    if done_indices:
        print(f"[Rank {rank}] Resuming OCR: {len(done_indices)} samples already processed, skipping them.")

    # Load GLM-OCR model
    ocr_device = args.ocr_device or device
    print(f"[Rank {rank}] Loading GLM-OCR from {args.ocr_model_path} on {ocr_device}...")
    ocr_processor = AutoProcessor.from_pretrained(args.ocr_model_path)
    ocr_model = AutoModelForImageTextToText.from_pretrained(
        args.ocr_model_path, torch_dtype=torch.bfloat16, device_map=ocr_device,
    )
    ocr_model.eval()

    # Stage 2 eval loop
    print(f"[Rank {rank}] Starting OCR extraction on {len(meta_records)} images...")
    for idx, meta in tqdm(enumerate(meta_records), total=len(meta_records), desc=f"Rank {rank} Stage2"):
        i = meta["index"]

        # Resume: skip already-processed
        if i in done_indices:
            continue

        image_path = meta["image_path"]
        if not os.path.exists(image_path):
            print(f"[Rank {rank}] Warning: image not found at {image_path}, skipping index {i}")
            continue

        # Extract text from generated image using GLM-OCR
        ocr_text = extract_text_with_glmocr(image_path, ocr_model, ocr_processor)

        print(f"[Rank {rank}][{i+1}] ==============GT Text============== {meta['gt_answer']}")
        print(f"[Rank {rank}][{i+1}] ==============OCR Extracted==============  {ocr_text}")
        print("=" * 50)

        record = {
            "index": i,
            "gt_answer": meta["gt_answer"],
            "ocr_extracted": ocr_text,
            "prompt": meta["prompt"],
            "image_path": image_path,
        }
        append_jsonl(ocr_jsonl_path, record)

        # Prevent VRAM accumulation
        gc.collect()
        torch.cuda.empty_cache()

    # Unload Stage 2 models
    print(f"[Rank {rank}] Stage 2 complete. Unloading GLM-OCR...")
    del ocr_model, ocr_processor
    gc.collect()
    torch.cuda.empty_cache()

    # Synchronize all ranks before aggregation
    if world_size > 1:
        dist.barrier()

    # ================================================================
    # Final: Metrics Aggregation (rank 0 only)
    # ================================================================
    if rank == 0:
        print("===== Final: Metrics Aggregation =====")

        # Pre-load metrics
        loaded_metrics = {
            "wer": evaluate.load("wer"),
            "cer": evaluate.load("cer"),
            "meteor": evaluate.load("meteor"),
        }

        # Merge all per-rank OCR JSONL files
        results = []
        seen_indices = set()
        for jsonl_file in sorted(glob.glob(os.path.join(args.output_dir, "ocr_results_rank*.jsonl"))):
            with open(jsonl_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if item["index"] not in seen_indices:
                        seen_indices.add(item["index"])
                        results.append(item)

        results.sort(key=lambda x: x["index"])

        if not results:
            print("No valid samples processed.")
        else:
            predictions = [r["ocr_extracted"] for r in results]
            references = [r["gt_answer"] for r in results]

            avg_metrics = calculate_metrics(predictions, references, loaded_metrics=loaded_metrics)

            print("\n" + "=" * 60)
            print("OCR Generation Evaluation Results")
            print("=" * 60)
            print(f"  Total samples: {len(results)}")
            for k, v in avg_metrics.items():
                print(f"  {k}: {v:.4f}")
            print("=" * 60)

            # Save results
            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            with open(os.path.join(args.output_dir, "metrics_summary.json"), "w") as f:
                json.dump(avg_metrics, f, indent=2)

            print(f"Results saved to {args.output_dir}/results.json")
            print(f"Metrics saved to {args.output_dir}/metrics_summary.json")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
