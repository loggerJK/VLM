# -*- coding: utf-8 -*-
"""
OCR Generation Evaluation Script (Multi-GPU, 2-Stage) — BAGEL — Quotes Dataset

Reads prompts from the Abirate/english_quotes dataset.

2-Stage pipeline:
  Stage 1: BAGEL generates images
  Stage 2: GLM-OCR extracts text from generated images
  Final:   Rank 0 aggregates -> metrics
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
from transformers import AutoProcessor, AutoModelForImageTextToText, set_seed
from safetensors.torch import load_file

import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Add bagel_train root to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache

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


def move_generation_input_to_device(generation_input, device):
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def generate_image_bagel(gen_model, model, vae_model, tokenizer, new_token_ids,
                         prompt, device,
                         num_timesteps=50, cfg_scale=4.0, cfg_interval=None,
                         cfg_renorm_min=0.0, timestep_shift=3.0, resolution=512):
    """Generate a single image using BAGEL's flow-matching generation pipeline."""
    if cfg_interval is None:
        cfg_interval = [0, 1.0]

    num_images = 1
    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0] * num_images
    new_rope = [0] * num_images

    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=[prompt] * num_images,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[(resolution, resolution)] * num_images,
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    cfg_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_newlens = [0] * num_images
    cfg_new_rope = [0] * num_images

    generation_input_cfg = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope,
        image_sizes=[(resolution, resolution)] * num_images,
    )
    generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = gen_model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift,
                cfg_text_past_key_values=cfg_past_key_values,
                cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                **generation_input,
            )

    image_list = []
    for latent in unpacked_latent:
        latent = latent.reshape(1, resolution // 16, resolution // 16, 2, 2, 16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, resolution // 8, resolution // 8)
        image = vae_model.decode(latent.to(device))
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        tmpimage = Image.fromarray(tmpimage)
        image_list.append(tmpimage)

    return image_list


def main():
    parser = argparse.ArgumentParser(description="OCR Generation Evaluation — Quotes Dataset (BAGEL)")
    parser.add_argument("--model_path", type=str, required=True, help="BAGEL model path")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path (optional)")
    parser.add_argument("--output_dir", type=str, default="ocr_generation_results", help="Output directory")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--img_size", type=int, default=512, help="Generated image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples (0=all)")
    parser.add_argument("--dataset_path", type=str, default="Abirate/english_quotes")
    parser.add_argument("--ocr_model_path", type=str, default="zai-org/GLM-OCR", help="GLM-OCR model path")
    parser.add_argument("--max_latent_size", type=int, default=64, help="Max latent size for VAE")
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

    set_all_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "generated_images")
    os.makedirs(images_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    # ===== Stage 1: Image Generation =====
    print(f"[Rank {rank}] ===== Stage 1: Image Generation =====")

    # Load dataset
    dataset = load_dataset(args.dataset_path, split="train", streaming=False)
    dataset = dataset.select(list(range(len(dataset) - world_size * 3, len(dataset))))
    args.num_samples = min(args.num_samples, len(dataset)) if args.num_samples else len(dataset)
    print(f"[Rank {rank}] Loaded {args.num_samples} samples from dataset {args.dataset_path}")

    meta_jsonl_path = os.path.join(args.output_dir, f"meta_rank{rank}.jsonl")
    # Load existing meta from ALL rank files to handle GPU count changes on resume
    existing_meta = {}
    for mf in glob.glob(os.path.join(glob.escape(args.output_dir), "meta_rank*.jsonl")):
        with open(mf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                existing_meta[item["index"]] = item

    # Load BAGEL model with VAE (generation mode)
    print(f"[Rank {rank}] Loading BAGEL model (generation mode)...")

    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model_state_dict_path = os.path.join(args.model_path, "ema.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    if rank == 0:
        print(msg)
    del model_state_dict

    # LoRA loading (applied to entire Bagel model, before moving to device)
    if args.lora_ckpt_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_ckpt_path, is_trainable=False)
        if rank == 0:
            print(f"Loaded LoRA weights from {args.lora_ckpt_path}")

    model = model.to(device).eval()
    vae_model = vae_model.to(device).eval()
    gen_model = model

    # Generation parameters
    cfg_scale = args.cfg_scale
    cfg_interval = [0, 1.0]
    timestep_shift = 3.0
    num_timesteps = 50
    cfg_renorm_min = 0.0

    processed = 0
    for i, sample in tqdm(enumerate(dataset), total=args.num_samples, desc=f"Rank {rank} Stage1"):
        if i % world_size != rank:
            continue

        answer_gt = sample.get('quote', "").replace("\n", " ").strip().replace('\u201c', '"').replace('\u201d', '"')[:120]
        if not answer_gt or not answer_gt.strip():
            continue

        save_path = os.path.join(images_dir, f"{i:05d}.png")
        if os.path.exists(save_path) and i in existing_meta:
            processed += 1
            continue

        set_all_seeds(args.seed)

        prompt_text = (
            f"A Mathpix Markdown format with sharp, legible black text. "
            f"High-resolution typography, top-down view. "
            f"The text is rendered in natural left-to-right, top-to-bottom reading order. "
            f"The text reads:\n"
            f"{answer_gt}"
        )
        prompt_text_save_path = save_path.replace(".png", "_prompt.txt")
        with open(prompt_text_save_path, "w") as f:
            f.write(prompt_text)

        # BAGEL image generation
        samples = generate_image_bagel(
            gen_model, model, vae_model, tokenizer, new_token_ids,
            prompt=prompt_text,
            device=device,
            num_timesteps=num_timesteps,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            timestep_shift=timestep_shift,
            resolution=args.img_size,
        )
        samples[0].save(save_path)

        meta_record = {"index": i, "gt_answer": answer_gt, "prompt": prompt_text, "image_path": save_path}
        append_jsonl(meta_jsonl_path, meta_record)
        print(f"[Rank {rank}][{i+1}] Image saved: {save_path}")

        processed += 1
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[Rank {rank}] Stage 1 complete. Unloading BAGEL...")
    del model, gen_model, vae_model
    gc.collect()
    torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()

    # ===== Stage 2: OCR Extraction =====
    print(f"[Rank {rank}] ===== Stage 2: OCR Extraction =====")
    meta_records = []
    if os.path.exists(meta_jsonl_path):
        seen = set()
        with open(meta_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item["index"] not in seen:
                    seen.add(item["index"])
                    meta_records.append(item)
    meta_records.sort(key=lambda x: x["index"])

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
