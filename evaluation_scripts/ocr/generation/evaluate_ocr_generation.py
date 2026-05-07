# -*- coding: utf-8 -*-
"""
OCR Generation Evaluation Script (Multi-GPU, 2-Stage) — Janus

Evaluates model's ability to generate images containing specified text.
Flow: text description -> model generates image -> GLM-OCR extracts text -> compare with GT
Dataset: Jiwon-Kang/Llama-Nemotron-VLM-Dataset-v1-OCR4 (validation split)

2-Stage pipeline to reduce peak VRAM:
  Stage 1: Janus generates images (resume via file existence check)
  Stage 2: GLM-OCR extracts text from generated images (resume via per-rank JSONL)
  Final:   Rank 0 aggregates all JSONL files -> metrics
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
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForImageTextToText, set_seed

import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu

from janus.models import MultiModalityCausalLM, VLChatProcessor
from pytorch_lightning import seed_everything

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
    try: metrics["wer"] = wer_metric.compute(predictions=preds_norm, references=refs_norm)
    except Exception: metrics["wer"] = 1.0
    try: metrics["cer"] = cer_metric.compute(predictions=preds_norm, references=refs_norm)
    except Exception: metrics["cer"] = 1.0
    try: metrics["meteor"] = meteor_metric.compute(predictions=preds_norm, references=refs_norm)["meteor"]
    except Exception: metrics["meteor"] = 0.0
    try:
        bleu_scores = [sentence_bleu([r.split()], p.split(), weights=(0.5, 0.5)) for p, r in zip(preds_norm, refs_norm)]
        metrics["bleu"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    except Exception: metrics["bleu"] = 0.0
    try:
        edit_dists = [nltk.edit_distance(p, r) for p, r in zip(preds_norm, refs_norm)]
        metrics["edit_distance"] = sum(edit_dists) / len(edit_dists) if edit_dists else 0.0
    except Exception: metrics["edit_distance"] = 0.0
    try:
        total_p, total_r, total_f1 = 0.0, 0.0, 0.0
        for p, r in zip(preds_norm, refs_norm):
            p_words, r_words = set(p.split()), set(r.split())
            if not p_words or not r_words: continue
            intersection = p_words.intersection(r_words)
            prec = len(intersection) / len(p_words) if p_words else 0.0
            rec = len(intersection) / len(r_words) if r_words else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            total_p += prec; total_r += rec; total_f1 += f1
        metrics["precision"] = total_p / len(preds_norm)
        metrics["recall"] = total_r / len(preds_norm)
        metrics["f1"] = total_f1 / len(preds_norm)
    except Exception: metrics["precision"] = 0.0; metrics["recall"] = 0.0; metrics["f1"] = 0.0
    return metrics


def extract_text_with_glmocr(image_path, ocr_model, ocr_processor):
    """Extract text from image using GLM-OCR (transformers)."""
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


@torch.inference_mode()
def generate_image_janus(
    mmgpt, vl_chat_processor, prompt,
    temperature=1, parallel_size=1, cfg_weight=5,
    image_token_num_per_image=576, img_size=384, patch_size=16,
):
    """Generate image(s) using Janus autoregressive generation with CFG."""
    device = mmgpt.language_model.model.device
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).to(device)
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(device)

    outputs = None
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds, use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None
        )
        hidden_states = outputs.last_hidden_state
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    return [Image.fromarray(visual_img[i]) for i in range(parallel_size)]


def main():
    parser = argparse.ArgumentParser(description="OCR Generation Evaluation (Janus)")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B", help="Janus model path")
    parser.add_argument("--lora_ckpt_path", type=str, default=None,
                        help="Path to LoRA checkpoint directory (optional)")
    parser.add_argument("--output_dir", type=str, default="ocr_generation_results", help="Output directory")
    parser.add_argument("--cfg_weight", type=float, default=5.0, help="CFG weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--img_size", type=int, default=384, help="Generated image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--dataset_path", type=str, default="Jiwon-Kang/Llama-Nemotron-VLM-Dataset-v1-OCR4")
    parser.add_argument("--ocr_model_path", type=str, default="zai-org/GLM-OCR", help="GLM-OCR model path")
    parser.add_argument("--ocr_device", type=str, default=None, help="Device for GLM-OCR")
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
        rank = 0; world_size = 1; local_rank = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_all_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "generated_images")
    os.makedirs(images_dir, exist_ok=True)
    if world_size > 1: dist.barrier()

    # ===== Stage 1: Image Generation (Janus) =====
    print(f"[Rank {rank}] ===== Stage 1: Image Generation =====")
    dataset = load_dataset(args.dataset_path, split="validation", streaming=True)

    meta_jsonl_path = os.path.join(args.output_dir, f"meta_rank{rank}.jsonl")
    existing_meta = {}
    if os.path.exists(meta_jsonl_path):
        with open(meta_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                item = json.loads(line)
                existing_meta[item["index"]] = item

    print(f"[Rank {rank}] Loading Janus model...")
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = model.to(torch.bfloat16).to(device).eval()

    # Load LoRA weights if provided
    if args.lora_ckpt_path is not None:
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(
            model.language_model, args.lora_ckpt_path
        )
        if rank == 0:
            print(f"Loaded LoRA weights from {args.lora_ckpt_path}")

    local_indices = list(range(rank, len(dataset), world_size))
    for i in tqdm(local_indices, desc=f"Rank {rank} Stage1"):
        item = dataset[i]
        answer_gt = item.get('answer', "")
        if not answer_gt or not answer_gt.strip(): continue

        save_path = os.path.join(images_dir, f"{i:05d}.png")
        if os.path.exists(save_path) and i in existing_meta:
            continue

        seed_everything(args.seed)
        prompt_text = f"An image containing the text: {answer_gt}"

        conversation = [
            {"role": "<|User|>", "content": prompt_text},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation, sft_format=vl_chat_processor.sft_format, system_prompt="",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag

        samples = generate_image_janus(
            model, vl_chat_processor, prompt,
            parallel_size=1, cfg_weight=args.cfg_weight,
            img_size=args.img_size, temperature=args.temperature,
        )
        samples[0].save(save_path)

        meta_record = {"index": i, "gt_answer": answer_gt, "prompt": prompt_text, "image_path": save_path}
        append_jsonl(meta_jsonl_path, meta_record)
        print(f"[Rank {rank}][{i+1}] Image saved: {save_path}")

        gc.collect(); torch.cuda.empty_cache()

    print(f"[Rank {rank}] Stage 1 complete. Unloading Janus...")
    del model; gc.collect(); torch.cuda.empty_cache()
    if world_size > 1: dist.barrier()

    # ===== Stage 2: OCR Extraction (GLM-OCR) =====
    print(f"[Rank {rank}] ===== Stage 2: OCR Extraction =====")
    meta_records = []
    if os.path.exists(meta_jsonl_path):
        seen = set()
        with open(meta_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                item = json.loads(line)
                if item["index"] not in seen: seen.add(item["index"]); meta_records.append(item)
    meta_records.sort(key=lambda x: x["index"])

    ocr_jsonl_path = os.path.join(args.output_dir, f"ocr_results_rank{rank}.jsonl")
    done_indices = load_done_indices(ocr_jsonl_path)
    if done_indices:
        print(f"[Rank {rank}] Resuming OCR: {len(done_indices)} already processed.")

    ocr_device = args.ocr_device or device
    print(f"[Rank {rank}] Loading GLM-OCR from {args.ocr_model_path}...")
    ocr_processor = AutoProcessor.from_pretrained(args.ocr_model_path)
    ocr_model = AutoModelForImageTextToText.from_pretrained(
        args.ocr_model_path, torch_dtype=torch.bfloat16, device_map=ocr_device,
    )
    ocr_model.eval()

    for idx, meta in tqdm(enumerate(meta_records), total=len(meta_records), desc=f"Rank {rank} Stage2"):
        i = meta["index"]
        if i in done_indices: continue
        image_path = meta["image_path"]
        if not os.path.exists(image_path): continue

        ocr_text = extract_text_with_glmocr(image_path, ocr_model, ocr_processor)
        print(f"[Rank {rank}][{i+1}] GT: {meta['gt_answer'][:80]}")
        print(f"[Rank {rank}][{i+1}] OCR: {ocr_text[:80]}")

        record = {"index": i, "gt_answer": meta["gt_answer"], "ocr_extracted": ocr_text,
                  "prompt": meta["prompt"], "image_path": image_path}
        append_jsonl(ocr_jsonl_path, record)
        gc.collect(); torch.cuda.empty_cache()

    del ocr_model, ocr_processor; gc.collect(); torch.cuda.empty_cache()
    if world_size > 1: dist.barrier()

    # ===== Final: Metrics Aggregation =====
    if rank == 0:
        print("===== Final: Metrics Aggregation =====")
        loaded_metrics = {"wer": evaluate.load("wer"), "cer": evaluate.load("cer"), "meteor": evaluate.load("meteor")}
        results = []; seen_indices = set()
        for jsonl_file in sorted(glob.glob(os.path.join(glob.escape(args.output_dir), "ocr_results_rank*.jsonl"))):
            with open(jsonl_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    item = json.loads(line)
                    if item["index"] not in seen_indices: seen_indices.add(item["index"]); results.append(item)
        results.sort(key=lambda x: x["index"])
        if not results:
            print("No valid samples processed.")
        else:
            predictions = [r["ocr_extracted"] for r in results]
            references = [r["gt_answer"] for r in results]
            avg_metrics = calculate_metrics(predictions, references, loaded_metrics=loaded_metrics)
            print("\n" + "=" * 60)
            print(f"OCR Generation Evaluation Results  |  Total: {len(results)}")
            for k, v in avg_metrics.items(): print(f"  {k}: {v:.4f}")
            print("=" * 60)
            with open(os.path.join(args.output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            with open(os.path.join(args.output_dir, "metrics_summary.json"), "w") as f:
                json.dump(avg_metrics, f, indent=2)

    if world_size > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
