#!/usr/bin/env python3
"""
PixMo Counting Evaluation Script — MMaDA (Multi-GPU)

Evaluates MMaDA's counting ability on the PixMo dataset.
Dataset: Jiwon-Kang/pixmo-count-filtered-imgContained (validation split, 501 samples)

Supports single-GPU (python) and multi-GPU (torchrun) execution.

Usage:
    # Single GPU
    python evaluate_pixmo_multigpu.py \
        --model_path Gen-Verse/MMaDA-8B-MixCoT \
        --vq_model_path showlab/magvitv2

    # Multi-GPU
    torchrun --nproc-per-node=8 --master-port=54321 evaluate_pixmo_multigpu.py \
        --model_path <checkpoint_path> \
        --vq_model_path showlab/magvitv2
"""

import os
import sys
import re
import gc
import glob
import json
import random
import argparse
from datetime import timedelta

import numpy as np
import torch
torch.set_grad_enabled(False)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
import torch.distributed as dist

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Make sure we can import from the mmada root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models import MAGVITv2, MMadaModelLM
from training.prompting_utils import UniversalPrompting
from parquet.my_dataset import image_transform_squash


# ---------------------------------------------------------------------------

def set_all_seeds(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


def extract_number_fixed(text: str) -> int:
    """Extract count from model output.

    Priority: **N** > English word (zero-ten) > plain digit.  Returns -1 on failure.
    """
    text = text.lower()
    match = re.search(r"\*\*(\d+)\*\*", text)
    if match:
        return int(match.group(1))
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    }
    for word, num in word_to_num.items():
        if re.search(r"\b" + word + r"\b", text):
            return num
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return -1


def extract_gt_number(text: str) -> int:
    match = re.search(r'\*\*(\d+)\*\*', text)
    if match:
        return int(match.group(1))
    match = re.search(r'\d+', text)
    if match:
        return int(match.group(0))
    return -1


def load_done_indices(jsonl_path: str) -> set:
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                done.add(item["index"])
    return done


def append_jsonl(jsonl_path: str, record: dict):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="MMaDA model path (local dir or HF hub ID)")
    parser.add_argument("--vq_model_path", type=str, default="showlab/magvitv2")
    parser.add_argument("--output_dir", type=str, default="evaluation_results/counting")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true",
                        help="Rank 0 logs eval metrics + confusion matrix to wandb")
    parser.add_argument("--wandb_project", type=str, default="mmada-counting-eval")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Optional PEFT LoRA adapter dir. If set, base model "
                             "is loaded from --model_path then adapter is merged in.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Distributed setup
    # ------------------------------------------------------------------
    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(days=1))
        device = torch.device("cuda", local_rank)
        print(f"[Rank {rank}] Distributed initialized. World size: {world_size}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running in single-process mode.")

    set_all_seeds(args.seed)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    # Skip if already done
    results_json_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_json_path):
        if rank == 0:
            print(f"Results already exist at {results_json_path}. Skipping inference.")
        if world_size > 1:
            dist.destroy_process_group()
        return

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print(f"[Rank {rank}] Loading dataset Jiwon-Kang/pixmo-count-filtered-imgContained …")
    dataset = load_dataset(
        "Jiwon-Kang/pixmo-count-filtered-imgContained", split="validation", streaming=True
    )
    # Materialise into a list so we can shard by index
    items = list(dataset)
    print(f"[Rank {rank}] Dataset size: {len(items)}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"[Rank {rank}] Loading MMaDA model from {args.model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=512,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=0.0,
        use_reserved_token=True,
    )

    vq_model = MAGVITv2.from_pretrained(args.vq_model_path).to(device).eval()
    vq_model.requires_grad_(False)

    model = MMadaModelLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    if args.lora_path:
        from peft import PeftModel
        if rank == 0:
            print(f"Loading LoRA adapter from {args.lora_path} …")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
    model = model.eval()

    # Resume: collect already-processed indices across all rank files
    jsonl_path = os.path.join(args.output_dir, f"results_rank{rank}.jsonl")
    done_indices = set()
    for f in glob.glob(os.path.join(glob.escape(args.output_dir), "results_rank*.jsonl")):
        done_indices |= load_done_indices(f)
    if done_indices:
        print(f"[Rank {rank}] Resuming: {len(done_indices)} samples already done.")

    # ------------------------------------------------------------------
    # Inference — each rank handles items where i % world_size == rank
    # ------------------------------------------------------------------
    print(f"[Rank {rank}] Starting inference …")

    for i, item in tqdm(enumerate(items), total=len(items), desc=f"Rank {rank}"):
        if i % world_size != rank:
            continue
        if i in done_indices:
            continue

        set_all_seeds(args.seed)

        image = item['image']
        question = item.get('question', item.get('question_count', ''))
        gt_answer = item.get('answer', item.get('answer_count', ''))

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        else:
            image = image.convert('RGB')

        gt_num = extract_gt_number(str(gt_answer))
        if gt_num < 0:
            try:
                gt_num = int(str(gt_answer).strip())
            except Exception:
                pass
        if gt_num < 0:
            print(f"[Rank {rank}] Warning: could not parse GT for item {i}: {gt_answer!r}")
            continue

        try:
            img_tensor = image_transform_squash(
                {'images': image}, resolution=args.resolution
            )['images'].unsqueeze(0).to(device)
            image_tokens = vq_model.get_code(img_tensor) + len(uni_prompting.text_tokenizer)

            # Match inference_mmu.py: apply_chat_template directly after <|eoi|>
            text_token_ids = uni_prompting.text_tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            input_ids = torch.cat([
                torch.full((1, 1), int(uni_prompting.sptids_dict['<|mmu|>']),
                           dtype=torch.long, device=device),
                torch.full((1, 1), int(uni_prompting.sptids_dict['<|soi|>']),
                           dtype=torch.long, device=device),
                image_tokens,
                torch.full((1, 1), int(uni_prompting.sptids_dict['<|eoi|>']),
                           dtype=torch.long, device=device),
                text_token_ids,
            ], dim=1).long()

            prompt_len = input_ids.shape[1]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.mmu_generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    steps=args.max_new_tokens,
                    block_length=args.max_new_tokens,
                    temperature=0.0,
                    remasking='low_confidence',
                )

            pred_text = uni_prompting.text_tokenizer.batch_decode(
                output_ids[:, prompt_len:], skip_special_tokens=True
            )[0]
            pred_num = extract_number_fixed(pred_text)

            append_jsonl(jsonl_path, {
                "index": i,
                "question": question,
                "gt_answer": str(gt_answer),
                "pred_answer": pred_text,
                "gt_num": gt_num,
                "pred_num": pred_num,
            })

        except Exception as e:
            print(f"[Rank {rank}] Error on item {i}: {e}")
            continue

        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Aggregate (rank 0 only)
    # ------------------------------------------------------------------
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        all_results = []
        seen = set()
        for jsonl_file in sorted(glob.glob(
            os.path.join(glob.escape(args.output_dir), "results_rank*.jsonl")
        )):
            with open(jsonl_file) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r["index"] not in seen:
                        seen.add(r["index"])
                        all_results.append(r)

        all_results.sort(key=lambda x: x["index"])

        if not all_results:
            print("No valid results to aggregate.")
            if world_size > 1:
                dist.destroy_process_group()
            return

        with open(results_json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved {len(all_results)} results to {results_json_path}")

        true_labels = [r['gt_num'] for r in all_results]
        pred_labels = [r['pred_num'] for r in all_results]

        accuracy = accuracy_score(true_labels, pred_labels)
        # Lumina parity: include parse failures (-1) in MAD
        deviations = [abs(g - p) for g, p in zip(true_labels, pred_labels)]
        mad = sum(deviations) / len(deviations) if deviations else 0.0

        print(f"\n=== Results ===")
        print(f"Total samples : {len(all_results)}")
        print(f"Accuracy      : {accuracy:.4f}")
        print(f"MAD           : {mad:.4f}")

        summary = {"accuracy": accuracy, "mad": mad, "total": len(all_results)}
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Confusion matrix (0–20 fixed range)
        fixed_labels = list(range(0, 21))
        cm = confusion_matrix(true_labels, pred_labels, labels=fixed_labels)
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=fixed_labels, yticklabels=fixed_labels,
                    cmap='viridis', ax=ax)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title(f'Confusion Matrix (0–20)  Acc: {accuracy:.4f}  MAD: {mad:.4f}')
        cm_path = os.path.join(args.output_dir, "confusion_matrix_0-20.png")
        plt.savefig(cm_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"Confusion matrix (0–20) saved to {cm_path}")

        # Full-range confusion matrix
        unique_labels = sorted(set(true_labels + pred_labels))
        valid_lbls = [l for l in unique_labels if l >= 0]
        cm_full = confusion_matrix(true_labels, pred_labels, labels=valid_lbls)
        fig2, ax2 = plt.subplots(
            figsize=(max(10, len(valid_lbls)), max(8, len(valid_lbls)))
        )
        sns.heatmap(cm_full, annot=True, fmt='d',
                    xticklabels=valid_lbls, yticklabels=valid_lbls,
                    cmap='viridis', ax=ax2)
        ax2.set_xlabel('Predicted label')
        ax2.set_ylabel('True label')
        ax2.set_title(f'Confusion Matrix (full range)  Acc: {accuracy:.4f}')
        cm_full_path = os.path.join(args.output_dir, "confusion_matrix_full.png")
        plt.savefig(cm_full_path, dpi=100, bbox_inches='tight')
        plt.close(fig2)
        print(f"Full confusion matrix saved to {cm_full_path}")

        # WandB logging (rank 0 only, after aggregation)
        if args.use_wandb:
            if not _WANDB_AVAILABLE:
                print("Warning: --use_wandb set but wandb not installed; skipping.")
            else:
                run_name = args.wandb_run_name or os.path.basename(
                    os.path.normpath(args.output_dir)
                )
                wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
                wandb.log({
                    "eval/accuracy": accuracy,
                    "eval/mad": mad,
                    "eval/total": len(all_results),
                    "eval/confusion_matrix": wandb.Image(cm_path),
                    "eval/confusion_matrix_full": wandb.Image(cm_full_path),
                })
                wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
