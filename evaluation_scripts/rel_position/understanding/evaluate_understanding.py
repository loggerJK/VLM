# -*- coding: utf-8 -*-
"""
Relative Position Understanding Evaluation Script (Multi-GPU)

Evaluates model's ability to identify relative positions of objects in images.
Flow: image -> model generates position text -> compare with GT answer
Dataset: heez/relative-position-new (validation split)

Supports single-GPU (python) and multi-GPU (torchrun) execution.
Supports resume: each rank writes results to a per-rank JSONL file.
On restart, already-processed indices are skipped automatically.
"""
import os
import sys
import re
import warnings
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

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from config import SPECIAL_TOKENS, PROMPT_TEMPLATES
from model import LLaDAForMultiModalGeneration
from utils.image_utils import (
    generate_crop_size_list,
    var_center_crop,
    add_break_line,
    encode_img_with_breaks_fixed,
)
from generators.text_understanding_generator import generate_text_understanding

# Special Tokens
MASK = SPECIAL_TOKENS["mask_token"]
NEW_LINE = SPECIAL_TOKENS["newline_token"]
BOA = SPECIAL_TOKENS["answer_start"]
EOA = SPECIAL_TOKENS["answer_end"]
BOI = SPECIAL_TOKENS["boi"]
EOI = SPECIAL_TOKENS["eoi"]
UNDERSTANDING_PROMPT_TEMPLATE = PROMPT_TEMPLATES["text_understanding"]

torch.set_grad_enabled(False)

# Position <-> MCQ letter mapping
POSITION_TO_LETTER = {"top-left": "a", "top-right": "b", "bottom-left": "c", "bottom-right": "d"}
LETTER_TO_POSITION = {v: k for k, v in POSITION_TO_LETTER.items()}
MCQ_CHOICES = "a. top-left\nb. top-right\nc. bottom-left\nd. bottom-right"
POSITION_PATTERN = r'(top-left|top-right|bottom-left|bottom-right)'


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


def transform_to_mqa(question: str, answer: str):
    """Transform open-ended question/answer to multiple-choice format.

    Returns (mqa_question, mqa_answer_letter).
    """
    # Strip "Response format: ..." line from question
    mqa_question = re.sub(r'\s*Response format:.*', '', question, flags=re.IGNORECASE).strip()
    # Append MCQ choices
    mqa_question = mqa_question + "\n" + MCQ_CHOICES

    # Extract position from GT answer and map to letter
    match = re.search(POSITION_PATTERN, answer.lower())
    if match:
        mqa_answer_letter = POSITION_TO_LETTER[match.group(1)]
    else:
        warnings.warn(f"Cannot extract position from GT answer: {answer}")
        mqa_answer_letter = "a"  # fallback

    return mqa_question, mqa_answer_letter


def check_mqa_accuracy(ground_truth_letter: str, prediction: str) -> bool:
    """Check if prediction contains the correct MCQ letter."""
    pred_clean = prediction.strip().lower()
    gt_letter = ground_truth_letter.strip().lower()
    if not pred_clean:
        return False
    return pred_clean[0] == gt_letter


def build_position_only_template(answer_text, tokenizer, mask_id):
    """Tokenize answer, mask only the position tokens, return template + length."""
    match = re.search(POSITION_PATTERN, answer_text.lower())
    answer_ids = tokenizer(answer_text, add_special_tokens=False)['input_ids']
    if not match:
        # Fallback: mask everything
        return [mask_id] * len(answer_ids), len(answer_ids)

    position_str = match.group(1)
    pos_ids = tokenizer(position_str, add_special_tokens=False)['input_ids']

    # Find subsequence match and mask those positions
    template = list(answer_ids)
    for start in range(len(answer_ids) - len(pos_ids) + 1):
        if answer_ids[start:start + len(pos_ids)] == pos_ids:
            for j in range(start, start + len(pos_ids)):
                template[j] = mask_id
            break

    return template, len(template)


def check_pos_accuracy(ground_truth: str, prediction: str) -> bool:
    pattern = r'(top-left|top-right|bottom-left|bottom-right)'

    gt_matches = re.findall(pattern, ground_truth.lower())
    pred_matches = re.findall(pattern, prediction.lower())

    if not gt_matches or not pred_matches:
        warnings.warn(f"Cannot parse position — GT: {gt_matches}, Pred: {pred_matches}")
        return False

    if len(pred_matches) > 1:
        warnings.warn(f"Multiple positions detected in prediction: {pred_matches}, returning False")
        return False

    return gt_matches[0] == pred_matches[0]


def main():
    parser = argparse.ArgumentParser(description="Relative Position Understanding Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Base model checkpoint path")
    parser.add_argument("--vae_ckpt", type=str, default="Alpha-VLLM/Lumina-DiMOO", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="rel_position_understanding_results", help="Output directory")
    parser.add_argument("--steps", type=int, default=128, help="Generation steps")
    parser.add_argument("--gen_length", type=int, default=128, help="Generation length")
    parser.add_argument("--block_length", type=int, default=128, help="Block length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples (None=all)")
    parser.add_argument("--dataset_path", type=str, default="heez/relative-position-new")
    parser.add_argument("--mqa", action="store_true", help="Use multiple-choice question format (a/b/c/d)")
    parser.add_argument("--predict_position_only", action="store_true",
                        help="Only mask position tokens in the GT answer, reveal all other tokens")
    args = parser.parse_args()

    if args.mqa and args.predict_position_only:
        parser.error("--mqa and --predict_position_only are mutually exclusive")

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

    # Load Dataset
    print(f"[Rank {rank}] Loading dataset: {args.dataset_path}...")
    dataset = load_dataset(args.dataset_path, split="validation", streaming=False)
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
    print(f"[Rank {rank}] Starting relative position understanding evaluation...")
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

        image = item['image']
        question_orig = item.get('question', '')
        answer_gt = item.get('answer', '')

        # Transform question/answer for MQA mode
        if args.mqa:
            question, answer_gt_eval = transform_to_mqa(question_orig, answer_gt)
        else:
            question = question_orig
            answer_gt_eval = answer_gt

        # Image Preprocessing (center crop, matching train_unified.py validate_position)
        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
        image_processed = var_center_crop(image, crop_size_list=crop_size_list)

        # Encode Image
        input_img_token, (H, W) = encode_img_with_breaks_fixed(image_processed, vqvae)
        img_token = [BOI] + add_break_line(input_img_token[1:-1], H, W, new_number=NEW_LINE) + [EOI]

        # Build prompt (matching validate_position pattern)
        instruction = "<system>" + UNDERSTANDING_PROMPT_TEMPLATE + "</system>" + "<user>" + question + "</user>"
        input_ids_raw = tokenizer(instruction)['input_ids']

        # Insert image tokens
        input_token = input_ids_raw[:-1] + img_token + input_ids_raw[-1:]

        # Prepare generation input
        code_start = len(input_token) + 1

        if args.predict_position_only:
            answer_template, answer_len = build_position_only_template(answer_gt, tokenizer, MASK)
            input_token = input_token + [BOA] + answer_template
            actual_gen_length = answer_len
            actual_block_length = answer_len
            actual_steps = answer_len
        else:
            GEN_LENGTH = args.gen_length if not args.mqa else (args.gen_length if args.gen_length != 128 else 8)
            input_token = input_token + [BOA] + [MASK] * GEN_LENGTH
            actual_gen_length = GEN_LENGTH
            actual_block_length = args.block_length if not args.mqa else (args.block_length if args.block_length != 128 else 8)
            actual_steps = args.steps if not args.mqa else (args.steps if args.steps != 128 else 8)

        input_ids = torch.tensor(input_token, device=device).unsqueeze(0)

        # Generate
        out = generate_text_understanding(
            model, input_ids,
            steps=actual_steps,
            gen_length=actual_gen_length,
            block_length=actual_block_length,
            temperature=args.temperature,
            cfg_scale=0.0,
            remasking='low_confidence',
            code_start=code_start,
        )

        pred_text = tokenizer.batch_decode(
            out[:, code_start:], skip_special_tokens=True
        )[0].replace("</answer>", "").strip()

        if args.mqa:
            correct = check_mqa_accuracy(answer_gt_eval, pred_text)
        else:
            correct = check_pos_accuracy(answer_gt, pred_text)

        print(f"\n[Rank {rank}][{i+1}] ==============Prediction==============\n {pred_text}")
        print(f"\n[Rank {rank}][{i+1}] ==============Ground Truth==============\n {answer_gt}")
        if args.mqa:
            print(f"[Rank {rank}][{i+1}] GT (MQA letter): {answer_gt_eval}")
        print(f"[Rank {rank}][{i+1}] Correct: {correct}")
        print("=" * 50)

        record = {
            "index": i,
            "question": question,
            "gt_answer": answer_gt,
            "pred_answer": pred_text,
            "correct": correct,
        }
        if args.mqa:
            record["gt_answer_mqa"] = answer_gt_eval
        append_jsonl(jsonl_path, record)

        # Prevent VRAM accumulation
        gc.collect()
        torch.cuda.empty_cache()

    # Synchronize all ranks before aggregation
    if world_size > 1:
        dist.barrier()

    # Aggregate results from all rank JSONL files (rank 0 only)
    if rank == 0:
        # Merge all per-rank JSONL files
        results = []
        seen_indices = set()
        for jsonl_file in sorted(glob.glob(os.path.join(glob.escape(args.output_dir), "results_rank*.jsonl"))):
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
            # Compute accuracy using the appropriate checker
            if args.mqa:
                correct_count = sum(
                    1 for r in results
                    if check_mqa_accuracy(r.get("gt_answer_mqa", ""), r["pred_answer"])
                )
            else:
                correct_count = sum(
                    1 for r in results
                    if check_pos_accuracy(r["gt_answer"], r["pred_answer"])
                )
            total_count = len(results)
            accuracy = correct_count / total_count if total_count > 0 else 0.0

            mode_label = "MQA" if args.mqa else ("Position-Only" if args.predict_position_only else "Open-Ended")
            print("\n" + "=" * 60)
            print(f"Relative Position Understanding Evaluation Results ({mode_label})")
            print("=" * 60)
            print(f"  Total samples: {total_count}")
            print(f"  Correct: {correct_count}")
            print(f"  Accuracy: {accuracy:.4f}")
            print("=" * 60)

            # Save results
            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            metrics_summary = {
                "accuracy": accuracy,
                "correct": correct_count,
                "total": total_count,
            }
            with open(os.path.join(args.output_dir, "metrics_summary.json"), "w") as f:
                json.dump(metrics_summary, f, indent=2)

            print(f"Results saved to {args.output_dir}/results.json")
            print(f"Metrics saved to {args.output_dir}/metrics_summary.json")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
