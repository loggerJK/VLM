# -*- coding: utf-8 -*-
"""
OCR Generation Evaluation — Stage 2: GLM-OCR Extraction + Metrics

Reads meta_rank*.jsonl produced by Stage 1, runs GLM-OCR on each generated
image, then computes aggregate OCR metrics on rank 0.

Run with:  conda run -n glm_ocr torchrun --nproc_per_node=N stage2_ocr_and_metrics.py ...
"""
import os
import argparse
import json
import gc
import glob
import random
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed

import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu

torch.set_grad_enabled(False)


def load_done_indices(jsonl_path):
    done = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                done.add(json.loads(line)["index"])
    return done


def append_jsonl(jsonl_path, record):
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
        bleu_scores = [sentence_bleu([r.split()], p.split(), weights=(0.5, 0.5))
                       for p, r in zip(preds_norm, refs_norm)]
        metrics["bleu"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    except Exception:
        metrics["bleu"] = 0.0
    try:
        edit_dists = [nltk.edit_distance(p, r) for p, r in zip(preds_norm, refs_norm)]
        metrics["edit_distance"] = sum(edit_dists) / len(edit_dists) if edit_dists else 0.0
    except Exception:
        metrics["edit_distance"] = 0.0
    try:
        total_p, total_r, total_f1 = 0.0, 0.0, 0.0
        for p, r in zip(preds_norm, refs_norm):
            p_words, r_words = set(p.split()), set(r.split())
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
    messages = [{"role": "user", "content": [
        {"type": "image", "url": image_path},
        {"type": "text", "text": "Text Recognition:"},
    ]}]
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
    parser = argparse.ArgumentParser(description="OCR Generation — Stage 2: GLM-OCR + Metrics")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory (must contain meta_rank*.jsonl from Stage 1)")
    parser.add_argument("--ocr_model_path", type=str, default="zai-org/GLM-OCR", help="GLM-OCR model path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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

    set_all_seeds(args.seed)

    # ===== Stage 2: OCR Extraction =====
    print(f"[Rank {rank}] ===== Stage 2: OCR Extraction =====")
    # Read ALL rank meta files and deduplicate — handles GPU count changes on resume
    all_meta = {}
    for mf in glob.glob(os.path.join(glob.escape(args.output_dir), "meta_rank*.jsonl")):
        with open(mf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item["index"] not in all_meta:
                    all_meta[item["index"]] = item
    all_meta_records = sorted(all_meta.values(), key=lambda x: x["index"])
    if not all_meta_records:
        print(f"[Rank {rank}] No meta records found in {args.output_dir}. Run Stage 1 first.")
        if world_size > 1:
            dist.destroy_process_group()
        return

    # Round-robin distribute across ranks
    meta_records = [m for idx, m in enumerate(all_meta_records) if idx % world_size == rank]

    ocr_jsonl_path = os.path.join(args.output_dir, f"ocr_results_rank{rank}.jsonl")
    # Load done indices from ALL rank files to handle GPU count changes on resume
    done_indices = set()
    for f in glob.glob(os.path.join(glob.escape(args.output_dir), "ocr_results_rank*.jsonl")):
        done_indices |= load_done_indices(f)

    ocr_processor = AutoProcessor.from_pretrained(args.ocr_model_path)
    ocr_model = AutoModelForImageTextToText.from_pretrained(
        args.ocr_model_path, torch_dtype=torch.bfloat16, device_map=device,
    )
    ocr_model.eval()

    for idx, meta in tqdm(enumerate(meta_records), total=len(meta_records), desc=f"Rank {rank} Stage2"):
        i = meta["index"]
        if i in done_indices:
            continue
        image_path = meta["image_path"]
        if not os.path.exists(image_path):
            continue
        ocr_text = extract_text_with_glmocr(image_path, ocr_model, ocr_processor)
        record = {"index": i, "gt_answer": meta["gt_answer"], "ocr_extracted": ocr_text,
                  "prompt": meta["prompt"], "image_path": image_path}
        append_jsonl(ocr_jsonl_path, record)
        gc.collect()
        torch.cuda.empty_cache()

    del ocr_model, ocr_processor
    gc.collect()
    torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()

    # ===== Final: Metrics =====
    if rank == 0:
        loaded_metrics = {
            "wer": evaluate.load("wer"),
            "cer": evaluate.load("cer"),
            "meteor": evaluate.load("meteor"),
        }
        results = []
        seen_indices = set()
        for jsonl_file in sorted(glob.glob(os.path.join(glob.escape(args.output_dir), "ocr_results_rank*.jsonl"))):
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
        if results:
            predictions = [r["ocr_extracted"] for r in results]
            references = [r["gt_answer"] for r in results]
            avg_metrics = calculate_metrics(predictions, references, loaded_metrics=loaded_metrics)
            print("\n" + "=" * 60)
            print(f"OCR Generation Evaluation Results  |  Total: {len(results)}")
            for k, v in avg_metrics.items():
                print(f"  {k}: {v:.4f}")
            print("=" * 60)
            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            with open(os.path.join(args.output_dir, "metrics_summary.json"), "w") as f:
                json.dump(avg_metrics, f, indent=2)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
