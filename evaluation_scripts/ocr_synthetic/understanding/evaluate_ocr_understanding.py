# -*- coding: utf-8 -*-
"""
OCR Understanding Evaluation Script (Multi-GPU)

Evaluates model's ability to read text from images.
Flow: image -> model generates text -> compare with GT answer
Dataset: agentlans/high-quality-english-sentences (test split)

Supports single-GPU (python) and multi-GPU (torchrun) execution.
Supports resume: each rank writes results to a per-rank JSONL file.
On restart, already-processed indices are skipped automatically.
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
from transformers import AutoTokenizer, set_seed
from diffusers import VQModel

import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from utils.ocr_render import generate_image, generate_image_bytes
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
from copy import deepcopy

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

    preds_norm = [p.lower() for p in predictions]
    refs_norm = [r.lower() for r in references]

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


def main():
    parser = argparse.ArgumentParser(description="OCR Understanding Evaluation")
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
    parser.add_argument("--dataset_path", type=str, default="agentlans/high-quality-english-sentences")
    parser.add_argument("--give_first_token", action="store_true", help="Whether to give the first token of the answer")
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
    dataset = load_dataset(args.dataset_path, split="test", streaming=False)
    # Select Samples
    dataset = dataset.select(list(range(0, min(args.num_samples, len(dataset)))))
    args.num_samples = min(args.num_samples, len(dataset)) if args.num_samples else len(dataset)

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
    model.eval()
    model.requires_grad_(False)

        
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae", torch_dtype=torch.bfloat16).to(device)
    vqvae.eval()
    vqvae.requires_grad_(False)

    # Eval loop
    print(f"[Rank {rank}] Starting OCR understanding evaluation...")
    # Pre-compute local indices for this rank (handles remainder correctly)
    local_indices = list(range(rank, args.num_samples, world_size))
    local_indices_set = set(local_indices)

    for i, item in tqdm(enumerate(dataset), total=args.num_samples, desc=f"Rank {rank}"):
        if i not in local_indices_set:
            continue

        # Resume: skip already-processed indices
        if i in done_indices:
            continue

        # Deterministic seed per sample
        set_all_seeds(args.seed)

        # image = item['image']
        question = item.get('question', "Extract all text from the image in reading order.")
        answer_gt = item.get('text', item.get('answer', "")).replace("\n", " ").strip()[:120]
        answer_input_ids = tokenizer(answer_gt, add_special_tokens=False)['input_ids']
        answer_template = deepcopy(answer_input_ids)  # copy answer_input_ids
        answer_template[1:] = [MASK] * (len(answer_template) - 1)  # Mask all but first token
        
        # Make image
        image = generate_image('# ' + answer_gt, template="clean_light", width=512, height=512, quality=100)
        image_save_path = os.path.join(args.output_dir, f"metadata", f"{i:05d}.png")
        prompt_save_path = os.path.join(args.output_dir, f"metadata", f"{i:05d}.txt")
        os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
        with open(prompt_save_path, "w") as f:
            f.write(f"Question:\n{question}\n\nGround Truth Answer:\n{answer_gt}\n")
        image.save(image_save_path) # Save image for reference

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

        # Prepare generation input
        code_start = len(input_token) + 1
        if args.give_first_token:
            input_token = input_token + [BOA] + answer_template + [EOA]
        else:
            input_token = input_token + [BOA] + [MASK] * len(answer_template) + [EOA]
        input_ids = torch.tensor(input_token, device=device).unsqueeze(0)

        # Generate
        out = generate_text_understanding(
            model, input_ids,
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
        )[0].replace("</answer>", "").strip().replace("\n", " ")

        print(f"\n[Rank {rank}][{i+1}] ==============Prediction==============\n {pred_text}")
        print(f"\n[Rank {rank}][{i+1}] ==============Ground Truth==============\n {answer_gt}")
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

    # Synchronize all ranks before aggregation
    if world_size > 1:
        dist.barrier()

    # Aggregate results from all rank JSONL files (rank 0 only)
    if rank == 0:
        # Pre-load metrics
        loaded_metrics = {
            "wer": evaluate.load("wer"),
            "cer": evaluate.load("cer"),
            "meteor": evaluate.load("meteor"),
        }

        # Merge all per-rank JSONL files
        results = []
        seen_indices = set()
        for jsonl_file in sorted(glob.glob(os.path.join(args.output_dir, "results_rank*.jsonl"))):
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
            predictions = [r["pred_answer"] for r in results]
            references = [r["gt_answer"] for r in results]

            avg_metrics = calculate_metrics(predictions, references, loaded_metrics=loaded_metrics)

            print("\n" + "=" * 60)
            print("OCR Understanding Evaluation Results")
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
