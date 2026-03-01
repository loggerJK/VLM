# -*- coding: utf-8 -*-
"""
OCR Understanding Evaluation Script (Multi-GPU) — BAGEL (Synthetic Images)

Evaluates model's ability to read text from synthetic images.
Flow: render text as image -> model generates text -> compare with GT answer
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
from transformers.trainer_utils import set_seed
from modeling.autoencoder import load_ae
from data.transforms import ImageTransform

import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Add bagel_train root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from eval.vlm.utils import build_transform
from accelerate import init_empty_weights
from safetensors.torch import load_file
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from data.data_utils import add_special_tokens

# Import ocr_render from Lumina-DiMOO utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'Lumina-DiMOO'))
from utils.ocr_render import generate_image as generate_ocr_image

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
    parser = argparse.ArgumentParser(description="OCR Understanding Evaluation — Synthetic (BAGEL)")
    parser.add_argument("--model_path", type=str, required=True, help="BAGEL model path")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path (optional)")
    parser.add_argument("--output_dir", type=str, default="ocr_understanding_results", help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--dataset_path", type=str, default="agentlans/high-quality-english-sentences")
    args = parser.parse_args()

    results_json_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_json_path):
        print(f"Results file already exists at {results_json_path}. Please remove it to run a new evaluation.")
        return

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

    # Resume: load already-processed indices from ALL rank files to handle GPU count changes
    jsonl_path = os.path.join(args.output_dir, f"results_rank{rank}.jsonl")
    done_indices = set()
    for f in glob.glob(os.path.join(glob.escape(args.output_dir), "results_rank*.jsonl")):
        done_indices |= load_done_indices(f)
    if done_indices:
        print(f"[Rank {rank}] Resuming: {len(done_indices)} samples already processed, skipping them.")

    # Load Dataset
    print(f"[Rank {rank}] Loading dataset: {args.dataset_path}...")
    dataset = load_dataset(args.dataset_path, split="test", streaming=False)
    dataset = dataset.select(list(range(0, min(args.num_samples, len(dataset)))))
    args.num_samples = min(args.num_samples, len(dataset)) if args.num_samples else len(dataset)

    # Load BAGEL Model — memory-efficient: meta tensors + direct GPU load
    print(f"[Rank {rank}] Loading BAGEL model from {args.model_path}...")

    # LLM config preparing
    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config preparing
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    # Bagel config preparing
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Load weights directly to rank's GPU — no CPU intermediate copy
    state_dict = load_file(os.path.join(args.model_path, "ema.safetensors"), device=str(device))
    msg = model.load_state_dict(state_dict, strict=False, assign=True)
    print(msg)
    del state_dict

    model = model.to(torch.bfloat16).to(device).eval()

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    image_transform = ImageTransform(980, 224, 14)
    
    # LoRA loading (applied to entire Bagel model)
    if args.lora_ckpt_path is not None:
        from peft import PeftModel
        # model = PeftModel.from_pretrained(model, args.lora_ckpt_path, is_trainable=False, torch_device='cpu')
        model.load_adapter(args.lora_ckpt_path)
        model = model.to(torch.bfloat16).to(device).eval()
        model = model.eval()
        if rank == 0:
            print(f"Loaded LoRA weights from {args.lora_ckpt_path}")

    # Eval loop — pre-compute local indices for this rank
    local_indices = list(range(rank, len(dataset), world_size))
    print(f"[Rank {rank}] Starting OCR understanding evaluation... ({len(local_indices)} samples assigned)")
    for i in tqdm(local_indices, desc=f"Rank {rank}"):
        item = dataset[i]

        # Resume: skip already-processed indices
        if i in done_indices:
            continue

        # Deterministic seed per sample
        set_all_seeds(args.seed)

        question = item.get('question', "Extract all text from the image in reading order.")
        answer_gt = item.get('text', item.get('answer', "")).replace("\n", " ").strip()[:120]

        # Make image from text (synthetic rendering)
        image = generate_ocr_image('# ' + answer_gt, template="clean_light", width=512, height=512, quality=100)
        image_save_path = os.path.join(args.output_dir, "metadata", f"{i:05d}.png")
        prompt_save_path = os.path.join(args.output_dir, "metadata", f"{i:05d}.txt")
        os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
        with open(prompt_save_path, "w") as f:
            f.write(f"Question:\n{question}\n\nGround Truth Answer:\n{answer_gt}\n")
        image.save(image_save_path)

        # BAGEL inference
        pred_text = model.chat(
            tokenizer, new_token_ids, image_transform,
            images=[image],
            prompt=question,
            max_length=args.max_new_tokens,
            do_sample=False,
        )
        pred_text = pred_text.strip().replace("\n", " ")

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

            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            with open(os.path.join(args.output_dir, "metrics_summary.json"), "w") as f:
                json.dump(avg_metrics, f, indent=2)

            print(f"Results saved to {args.output_dir}/results.json")
            print(f"Metrics saved to {args.output_dir}/metrics_summary.json")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
