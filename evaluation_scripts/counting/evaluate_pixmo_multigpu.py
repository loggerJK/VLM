# -*- coding: utf-8 -*-
"""
PixMo Counting Evaluation Script (Multi-GPU) — BAGEL

Evaluates model's counting ability on synthetic counting images.
Dataset: Jiwon-Kang/pixmo-count-filtered-imgContained (validation split, 501 samples)

Supports single-GPU (python) and multi-GPU (torchrun) execution.
"""
import os
import sys
import argparse
import json
import gc
import glob
import torch
torch.set_grad_enabled(False)
import random
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from transformers import set_seed
import torch.distributed as dist
from datetime import timedelta

# Add bagel_train root to sys.path so we can import eval.vlm.utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from eval.vlm.utils import load_model_and_tokenizer, build_transform


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


def extract_number(text):
    match = re.search(r'\*\*(\d+)\*\*', text)
    if match:
        return int(match.group(1))

    match = re.search(r'\d+', text)
    if match:
        return int(match.group(0))

    return -1


def extract_number_fixed(text):
    """Extract number from text pattern **number**, number, or English words (zero-ten)."""
    text = text.lower()

    # 1. Try **number**
    match = re.search(r"\*\*(\d+)\*\*", text)
    if match:
        return int(match.group(1))

    # 2. Try English words (zero to ten)
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10
    }
    for word, num in word_to_num.items():
        if re.search(r"\b" + word + r"\b", text):
            return num

    # 3. Try plain digits
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))

    return -1


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pixmo Counting (BAGEL)")
    parser.add_argument("--model_path", type=str, required=True, help="BAGEL model path")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path (optional)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Check if results.json file already exists
    results_json_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_json_path):
        print(f"Results file {results_json_path} already exists. Skipping inference.")
        return

    # Initialize Distributed Training
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

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    # Skip if results already exist
    output_json_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(output_json_path):
        if rank == 0:
            print(f"Results file {output_json_path} already exists. Skipping inference.")
        if world_size > 1:
            dist.destroy_process_group()
        return

    # Load Dataset
    print(f"[Rank {rank}] Loading dataset...")
    try:
        dataset = load_dataset("Jiwon-Kang/pixmo-count-filtered-imgContained", split="validation", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Load BAGEL Model
    print(f"[Rank {rank}] Loading BAGEL model from {args.model_path}...")
    model, tokenizer, new_token_ids = load_model_and_tokenizer(args)
    model = model.to(device)  # Ensure model is on correct rank device
    image_transform = build_transform()

    # LoRA loading (applied to entire Bagel model)
    if args.lora_ckpt_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_ckpt_path, is_trainable=False)
        model = model.to(device).eval()
        if rank == 0:
            print(f"Loaded LoRA weights from {args.lora_ckpt_path}")

    # Resume: load already-processed indices from ALL rank files (robust to GPU count changes)
    jsonl_path = os.path.join(args.output_dir, f"results_rank{rank}.jsonl")
    done_indices = set()
    for f in glob.glob(os.path.join(glob.escape(args.output_dir), "results_rank*.jsonl")):
        done_indices |= load_done_indices(f)
    if done_indices:
        print(f"[Rank {rank}] Resuming: {len(done_indices)} samples already processed, skipping them.")

    print(f"[Rank {rank}] Starting inference...")
    iterable_dataset = dataset

    iterator = tqdm(enumerate(iterable_dataset), total=501, desc=f"Rank {rank}")

    for i, item in iterator:
        if i % world_size != rank:
            continue

        # Resume: skip already-processed indices
        if i in done_indices:
            continue

        set_all_seeds(args.seed)

        image = item['image']
        question = item['question']
        gt_answer = item['answer']
        question = question.replace('**<number>** of', '**<number>**')

        # Ground Truth
        gt_num = extract_number(gt_answer)
        if gt_num == -1:
            try:
                gt_num = int(gt_answer.strip())
            except:
                pass

        if gt_num == -1:
            print(f"Warning: Could not extract number from GT: {gt_answer}")
            continue

        # Ensure image is PIL
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        # BAGEL inference
        text_new = model.chat(
            tokenizer, new_token_ids, image_transform,
            images=[image],
            prompt=question,
            max_length=args.max_new_tokens,
            do_sample=False,
        )

        pred_num = extract_number_fixed(text_new)

        record = {
            "index": i,
            "question": question,
            "gt_answer": gt_answer,
            "pred_answer": text_new,
            "gt_num": gt_num,
            "pred_num": pred_num,
        }
        append_jsonl(jsonl_path, record)

        gc.collect()
        torch.cuda.empty_cache()

    # Synchronize all ranks before aggregation
    if world_size > 1:
        dist.barrier()

    # Aggregate results from all rank JSONL files (rank 0 only)
    if rank == 0:
        results = []
        seen_indices = set()
        for jsonl_file in sorted(glob.glob(os.path.join(glob.escape(args.output_dir), "results_rank*.jsonl"))):
            with open(jsonl_file, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if item["index"] not in seen_indices:
                        seen_indices.add(item["index"])
                        results.append(item)
        results.sort(key=lambda x: x["index"])

        true_labels = [r['gt_num'] for r in results]
        pred_labels = [r['pred_num'] for r in results]

        if not true_labels:
            print(f"No valid samples processed.")
            return

        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"Total Accuracy: {accuracy:.4f}")

        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {os.path.join(args.output_dir, 'results.json')}")

        unique_labels = sorted(list(set(true_labels + pred_labels)))
        valid_labels = [l for l in unique_labels if l >= 0]

        cm = confusion_matrix(true_labels, pred_labels, labels=valid_labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=valid_labels, yticklabels=valid_labels, cmap='viridis')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(f'Confusion Matrix (Acc: {accuracy:.4f})')
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        plt.close()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
