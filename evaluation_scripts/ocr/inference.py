# -*- coding: utf-8 -*-
"""
OCR Understanding Evaluation — Qwen2.5-VL with optional BLIP3o LoRA

Evaluates OCR ability on rendered synthetic text images using Qwen2.5-VL as the
backbone, with optional LoRA adapters converted from BLIP3o checkpoints.

Dataset: agentlans/high-quality-english-sentences (test split)

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
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed
from qwen_vl_utils import process_vision_info
import torch.distributed as dist
from datetime import timedelta
from peft import PeftConfig
from safetensors.torch import load_file

import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu

sys.path.append(os.path.join(
    os.path.dirname(__file__), "..", "..", "qwen-vl-finetune", "qwenvl", "data"
))
from ocr_render import generate_image


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


def load_lora_adapter(model, lora_ckpt_path, lora_mode):
    """Load a BLIP3o LoRA checkpoint into a Qwen2.5-VL model with key conversion.

    Args:
        model: Qwen2_5_VLForConditionalGeneration instance
        lora_ckpt_path: path to the LoRA checkpoint directory
        lora_mode: "und" for understanding LoRA, "gen" for generation LoRA
    """
    peft_config = PeftConfig.from_pretrained(lora_ckpt_path)
    weight_path = os.path.join(lora_ckpt_path, 'adapter_model.safetensors')
    ckpt = load_file(weight_path)

    new_ckpt = {}
    if lora_mode == "und":
        # Understanding LoRA: Qwen keys inside BLIP3o language_model
        for k, v in ckpt.items():
            new_k = k.replace('base_model.model.model.language_model', 'model')
            new_ckpt[new_k] = v
    elif lora_mode == "gen":
        # Generation LoRA: broader BLIP3o model keys, skip non-matching
        for k, v in ckpt.items():
            if 'base_model.model.model' in k:
                new_k = k.replace('base_model.model.model', 'model')
                new_ckpt[new_k] = v
    else:
        raise ValueError(f"Unknown lora_mode: {lora_mode}. Must be 'und' or 'gen'.")

    model.load_adapter(
        peft_config=peft_config,
        adapter_state_dict=new_ckpt,
        adapter_name='default',
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR Understanding (Qwen2.5-VL)")
    parser.add_argument("--model_path", type=str, required=True, help="Qwen2.5-VL model path or HF hub ID")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path (optional)")
    parser.add_argument("--lora_mode", type=str, default="und", choices=["und", "gen"],
                        help="LoRA key conversion mode: 'und' (understanding) or 'gen' (generation)")
    parser.add_argument("--output_dir", type=str, default="ocr_understanding_results", help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max new tokens for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of test samples to evaluate")
    parser.add_argument("--dataset_path", type=str, default="agentlans/high-quality-english-sentences",
                        help="Dataset path")

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

    # Load Dataset
    print(f"[Rank {rank}] Loading dataset: {args.dataset_path}...")
    try:
        dataset = load_dataset(args.dataset_path, split="test", streaming=False)
        args.num_samples = min(args.num_samples, len(dataset)) if args.num_samples else len(dataset)
        dataset = dataset.select(list(range(0, args.num_samples)))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Load Qwen2.5-VL Model
    print(f"[Rank {rank}] Loading Qwen2.5-VL model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{local_rank}",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Optional LoRA loading
    if args.lora_ckpt_path is not None:
        load_lora_adapter(model, args.lora_ckpt_path, args.lora_mode)
        model = model.to(torch.bfloat16).to(device).eval()
        if rank == 0:
            print(f"Loaded LoRA weights from {args.lora_ckpt_path} (mode: {args.lora_mode})")
    else:
        model.eval()

    # Resume: load already-processed indices from ALL rank files (robust to GPU count changes)
    jsonl_path = os.path.join(args.output_dir, f"results_rank{rank}.jsonl")
    done_indices = set()
    for f in glob.glob(os.path.join(glob.escape(args.output_dir), "results_rank*.jsonl")):
        done_indices |= load_done_indices(f)
    if done_indices:
        print(f"[Rank {rank}] Resuming: {len(done_indices)} samples already processed, skipping them.")

    print(f"[Rank {rank}] Starting OCR understanding evaluation...")

    iterator = tqdm(enumerate(dataset), total=args.num_samples, desc=f"Rank {rank}")

    for i, item in iterator:
        if i % world_size != rank:
            continue

        # Resume: skip already-processed indices
        if i in done_indices:
            continue

        set_all_seeds(args.seed)

        question = item.get('question', "Extract all text from the image in reading order.")
        gt_answer = item.get('text', item.get('answer', "")).replace("\n", " ").strip()[:120]
        image = generate_image('# ' + gt_answer, template="clean_light", width=512, height=512, quality=100)

        image_save_path = os.path.join(args.output_dir, "metadata", f"{i:05d}.png")
        prompt_save_path = os.path.join(args.output_dir, "metadata", f"{i:05d}.txt")
        os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
        with open(prompt_save_path, "w") as f:
            f.write(f"Question:\n{question}\n\nGround Truth Answer:\n{gt_answer}\n")
        image.save(image_save_path)

        # Ensure image is PIL
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        # Build messages for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Generation
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        text_new = output_text[0]

        text_new = text_new.strip().replace("\n", " ")
        
        print(f"\n[Rank {rank}][{i+1}] ==============Prediction==============\n {text_new}")
        print(f"\n[Rank {rank}][{i+1}] ==============Ground Truth==============\n {gt_answer}")
        print("=" * 50)

        record = {
            "index": i,
            "question": question,
            "gt_answer": gt_answer,
            "pred_answer": text_new,
        }
        append_jsonl(jsonl_path, record)

        gc.collect()
        torch.cuda.empty_cache()

    # Synchronize all ranks before aggregation
    if world_size > 1:
        dist.barrier()

    # Aggregate results from all rank JSONL files (rank 0 only)
    if rank == 0:
        loaded_metrics = {
            "wer": evaluate.load("wer"),
            "cer": evaluate.load("cer"),
            "meteor": evaluate.load("meteor"),
        }

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

        if not results:
            print(f"No valid samples processed.")
            return

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

        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {os.path.join(args.output_dir, 'results.json')}")

        metrics_path = os.path.join(args.output_dir, "metrics_summary.json")
        with open(metrics_path, "w") as f:
            json.dump(avg_metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
