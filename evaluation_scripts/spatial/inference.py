# -*- coding: utf-8 -*-
"""
Relative Position Understanding Evaluation — Qwen2.5-VL with optional BLIP3o LoRA

Evaluates relative position understanding using Qwen2.5-VL as the
backbone, with optional LoRA adapters converted from BLIP3o checkpoints.

Dataset: heez/relative-position-new (validation split)

Supports single-GPU (python) and multi-GPU (torchrun) execution.
"""
import os
import argparse
import json
import gc
import glob
import torch
torch.set_grad_enabled(False)
import random
import numpy as np
import re
import warnings
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed
from qwen_vl_utils import process_vision_info
import torch.distributed as dist
from datetime import timedelta
from peft import PeftConfig
from safetensors.torch import load_file


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


def check_pos_accuracy(ground_truth: str, prediction: str) -> bool:
    gt_matches = re.findall(POSITION_PATTERN, ground_truth.lower())
    pred_matches = re.findall(POSITION_PATTERN, prediction.lower())

    if not gt_matches or not pred_matches:
        warnings.warn(f"Cannot parse position — GT: {gt_matches}, Pred: {pred_matches}")
        return False

    if len(pred_matches) > 1:
        warnings.warn(f"Multiple positions detected in prediction: {pred_matches}, returning False")
        return False

    return gt_matches[0] == pred_matches[0]


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
    parser = argparse.ArgumentParser(description="Evaluate Relative Position Understanding (Qwen2.5-VL)")
    parser.add_argument("--model_path", type=str, required=True, help="Qwen2.5-VL model path or HF hub ID")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path (optional)")
    parser.add_argument("--lora_mode", type=str, default="und", choices=["und", "gen"],
                        help="LoRA key conversion mode: 'und' (understanding) or 'gen' (generation)")
    parser.add_argument("--output_dir", type=str, default="spatial_understanding_results", help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max new tokens for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of validation samples to evaluate")
    parser.add_argument("--dataset_path", type=str, default="heez/relative-position-new", help="Dataset path")

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
        dataset = load_dataset(args.dataset_path, split="validation", streaming=False)
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

    print(f"[Rank {rank}] Starting relative position understanding evaluation...")

    iterator = tqdm(enumerate(dataset), total=args.num_samples, desc=f"Rank {rank}")

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

        correct = check_pos_accuracy(gt_answer, text_new)
        
        print(f"\n[Rank {rank}][{i+1}] ==============Prediction==============\n {text_new}")
        print(f"\n[Rank {rank}][{i+1}] ==============Ground Truth==============\n {gt_answer}")
        print(f"[Rank {rank}][{i+1}] Correct: {correct}")
        print("=" * 50)

        record = {
            "index": i,
            "question": question,
            "gt_answer": gt_answer,
            "pred_answer": text_new,
            "correct": correct,
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

        if not results:
            print(f"No valid samples processed.")
            return

        correct_count = sum(
            1 for r in results
            if check_pos_accuracy(r["gt_answer"], r["pred_answer"])
        )
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        print("\n" + "=" * 60)
        print("Relative Position Understanding Evaluation Results")
        print("=" * 60)
        print(f"  Total samples: {total_count}")
        print(f"  Correct: {correct_count}")
        print(f"  Accuracy: {accuracy:.4f}")
        print("=" * 60)

        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {os.path.join(args.output_dir, 'results.json')}")

        metrics_summary = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": total_count,
        }
        metrics_path = os.path.join(args.output_dir, "metrics_summary.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
