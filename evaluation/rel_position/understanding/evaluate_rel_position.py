#!/usr/bin/env python3
"""
Relative-Position Understanding Evaluation — MMaDA (Single-GPU)

Evaluates MMaDA on the heez/relative-position-new dataset.
4-way spatial classification: top-left / top-right / bottom-left / bottom-right.

Modes:
    --mode open  (default): free-form generation, regex-extract position keyword
    --mode mqa            : Lumina-style multiple-choice (a/b/c/d), 8-token gen

Usage:
    python evaluate_rel_position.py \
        --model_path Gen-Verse/MMaDA-8B-MixCoT \
        --vq_model_path showlab/magvitv2 \
        --output_dir evaluation_results/rel_position \
        --mode open
"""

import os
import sys
import re
import gc
import json
import random
import argparse

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

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from models import MAGVITv2, MMadaModelLM
from training.prompting_utils import UniversalPrompting
from parquet.my_dataset import image_transform_squash


REL_POSITION_LABELS = ('top-left', 'top-right', 'bottom-left', 'bottom-right')
POSITION_TO_LETTER = {
    'top-left': 'a', 'top-right': 'b', 'bottom-left': 'c', 'bottom-right': 'd',
}
LETTER_TO_POSITION = {v: k for k, v in POSITION_TO_LETTER.items()}
_POS_RE = re.compile(r'(top-left|top-right|bottom-left|bottom-right)')


def set_all_seeds(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


def extract_position(text: str):
    if not text:
        return None
    m = _POS_RE.findall(text.lower())
    return m[0] if m else None


def transform_to_mqa(question: str) -> str:
    """Strip Lumina-style 'Response format: ...' line and append a/b/c/d choices."""
    cleaned = re.sub(r'(?im)^response format:.*$\n?', '', question).strip()
    choices = '\na. top-left\nb. top-right\nc. bottom-left\nd. bottom-right'
    return cleaned + '\nChoose one of the following options:' + choices


def extract_letter(text: str):
    """Return first letter in {a,b,c,d} found in text, else None."""
    if not text:
        return None
    m = re.search(r'(?i)\b([abcd])\b', text)
    if m:
        return m.group(1).lower()
    stripped = text.strip().lower()
    if stripped and stripped[0] in 'abcd':
        return stripped[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="MMaDA model path (local dir or HF hub ID)")
    parser.add_argument("--vq_model_path", type=str, default="showlab/magvitv2",
                        help="MAGVITv2 model path")
    parser.add_argument("--output_dir", type=str, default="evaluation_results/rel_position")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--mode", choices=['open', 'mqa'], default='open',
                        help="open=free-form + regex; mqa=multiple-choice a/b/c/d")
    parser.add_argument("--split", type=str, default="test",
                        help="HF split (default: test=500 samples; validation=100)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit samples (default: all in split)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mmada-rel-position-eval")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Optional PEFT LoRA adapter dir. If set, base model "
                             "is loaded from --model_path then adapter is merged in.")
    args = parser.parse_args()

    if args.mode == 'open':
        gen_len = 64
    else:
        gen_len = 8

    os.makedirs(args.output_dir, exist_ok=True)

    use_wandb = args.use_wandb
    if use_wandb and not _WANDB_AVAILABLE:
        print("Warning: --use_wandb set but wandb not installed; disabling.")
        use_wandb = False
    if use_wandb:
        run_name = args.wandb_run_name or os.path.basename(os.path.normpath(args.output_dir))
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    results_json_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_json_path):
        print(f"Results file {results_json_path} already exists. Skipping inference.")
        if use_wandb:
            wandb.finish()
        return

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset heez/relative-position-new[{args.split}] …")
    dataset = load_dataset("heez/relative-position-new", split=args.split)
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    total = len(dataset)
    print(f"  {total} samples (mode={args.mode}, gen_len={gen_len})")

    print(f"Loading MMaDA model from {args.model_path} …")
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

    print(f"Loading MAGVITv2 from {args.vq_model_path} …")
    vq_model = MAGVITv2.from_pretrained(args.vq_model_path).to(device).eval()
    vq_model.requires_grad_(False)

    model = MMadaModelLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    if args.lora_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {args.lora_path} …")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
    model = model.eval()

    results = []
    for i, item in tqdm(enumerate(dataset), total=total, desc=f"Eval[{args.mode}]"):
        set_all_seeds(args.seed)

        image = item['image']
        question = item.get('question', '')
        gt_position = str(item.get('position', '')).strip().lower()
        if gt_position not in REL_POSITION_LABELS:
            gt_position = extract_position(item.get('answer', '')) or ''
        if not gt_position:
            print(f"Warning: GT not parsable on item {i}; skipping.")
            continue

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        else:
            image = image.convert('RGB')

        prompt_text = transform_to_mqa(question) if args.mode == 'mqa' else question

        try:
            img_tensor = image_transform_squash(
                {'images': image}, resolution=args.resolution
            )['images'].unsqueeze(0).to(device)
            image_tokens = vq_model.get_code(img_tensor) + len(uni_prompting.text_tokenizer)

            text_token_ids = uni_prompting.text_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
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
                    max_new_tokens=gen_len,
                    steps=gen_len,
                    block_length=gen_len,
                    temperature=0.0,
                    remasking='low_confidence',
                )

            pred_text = uni_prompting.text_tokenizer.batch_decode(
                output_ids[:, prompt_len:], skip_special_tokens=True
            )[0]

            if args.mode == 'mqa':
                pred_letter = extract_letter(pred_text)
                pred_position = LETTER_TO_POSITION.get(pred_letter) if pred_letter else None
            else:
                pred_position = extract_position(pred_text)

            results.append({
                "index": i,
                "mode": args.mode,
                "question": question,
                "gt_position": gt_position,
                "pred_position": pred_position,
                "pred_text": pred_text,
            })

        except Exception as e:
            print(f"Error on item {i}: {e}")
            continue

        gc.collect()
        torch.cuda.empty_cache()

    if not results:
        print("No valid results.")
        return

    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {results_json_path}")

    gt_labels = [r['gt_position'] for r in results]
    pred_labels = [r['pred_position'] for r in results]

    correct = sum(1 for g, p in zip(gt_labels, pred_labels) if g == p and p is not None)
    accuracy = correct / len(results)
    parse_fails = sum(1 for p in pred_labels if p is None)

    per_class_total = {l: 0 for l in REL_POSITION_LABELS}
    per_class_correct = {l: 0 for l in REL_POSITION_LABELS}
    for g, p in zip(gt_labels, pred_labels):
        per_class_total[g] = per_class_total.get(g, 0) + 1
        if g == p:
            per_class_correct[g] = per_class_correct.get(g, 0) + 1
    per_class_acc = {
        l: (per_class_correct[l] / per_class_total[l]) if per_class_total[l] > 0 else 0.0
        for l in REL_POSITION_LABELS
    }

    print("\n=== Results ===")
    print(f"Mode          : {args.mode}")
    print(f"Split         : {args.split}")
    print(f"Total samples : {len(results)}")
    print(f"Accuracy      : {accuracy:.4f} ({correct}/{len(results)})")
    print(f"Parse fails   : {parse_fails}")
    print("Per-class accuracy:")
    for l in REL_POSITION_LABELS:
        print(f"  {l:>12}: {per_class_acc[l]:.4f}  ({per_class_correct[l]}/{per_class_total[l]})")

    summary = {
        "mode": args.mode,
        "split": args.split,
        "accuracy": accuracy,
        "correct": correct,
        "parse_fails": parse_fails,
        "total": len(results),
        "per_class_total": per_class_total,
        "per_class_correct": per_class_correct,
        "per_class_accuracy": per_class_acc,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fixed_labels = list(REL_POSITION_LABELS)
    cm_pairs = [(g, p) for g, p in zip(gt_labels, pred_labels) if p is not None]
    if cm_pairs:
        cm_gt = [g for g, _ in cm_pairs]
        cm_pred = [p for _, p in cm_pairs]
        cm = confusion_matrix(cm_gt, cm_pred, labels=fixed_labels)
    else:
        cm = np.zeros((4, 4), dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=fixed_labels, yticklabels=fixed_labels,
                cmap='viridis', ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'RelPosition CM ({args.mode})  Acc: {accuracy:.4f}  ParseFails: {parse_fails}')
    cm_path = os.path.join(args.output_dir, "confusion_matrix_rel_position.png")
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")

    if use_wandb:
        wandb.log({
            "eval/accuracy": accuracy,
            "eval/parse_fails": parse_fails,
            "eval/total": len(results),
            "eval/confusion_matrix": wandb.Image(cm_path),
            **{f"eval/per_class_acc/{l}": per_class_acc[l] for l in REL_POSITION_LABELS},
        })
        wandb.finish()


if __name__ == "__main__":
    main()
