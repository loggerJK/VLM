import pickle
from typing import List, Tuple, Dict, Any, Union
import random
from accelerate import init_empty_weights
import torch
import os
import numpy as np
import torch.nn as nn
import math
import contextlib
import sys
import time
import datetime
import json
from pathlib import Path
import wandb
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
from PIL import Image

# HuggingFace & Diffusers
from datasets import load_dataset, concatenate_datasets
from diffusers import VQModel
from transformers import AutoTokenizer, AutoConfig

# Metrics
import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Project Imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import SPECIAL_TOKENS, PROMPT_TEMPLATES
from model import LLaDAForMultiModalGeneration
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.solvers.finetune import FinetuneSolverBase
from xllmx.data.sampler import FinetuneDistSampler
import xllmx.util as util
import xllmx.util.lr_sched as lr_sched
import xllmx.util.misc as misc
from fairscale.nn.model_parallel import initialize as fs_init

# Image Utils
from utils.image_utils import encode_img_with_breaks, generate_crop_size_list, var_center_crop, var_edge_pad, encode_img_with_breaks_fixed, add_break_line, decode_vq_to_image, calculate_vq_params
# Generation Utils
from generators.text_understanding_generator import generate_text_understanding
from generators.image_generation_generator import generate_image
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates

from transformers import enable_full_determinism
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import functools

# ==============================================================================
# 1. Special Tokens Global Definition
# ==============================================================================
MASK = SPECIAL_TOKENS["mask_token"]
NEW_LINE = SPECIAL_TOKENS["newline_token"]
BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
EOA = SPECIAL_TOKENS["answer_end"]    # End of Answer
BOI = SPECIAL_TOKENS["boi"]           # Begin of Image
EOI = SPECIAL_TOKENS["eoi"]           # End of Image
PAD = 126339                          # Padding token
IMAGE_TOKEN_OFFSET = SPECIAL_TOKENS["image_token_offset"]
UNDERSTANDING_PROMPT_TEMPLATE = PROMPT_TEMPLATES["text_understanding"]


# ==============================================================================
# 2. Helper Functions
# ==============================================================================

def parse_checkpoint_name(ckpt_str: str):
    parts = ckpt_str.split("-")
    epoch = int(parts[0].replace("epoch", ""))
    iteration = None
    global_step = None
    for part in parts[1:]:
        if part.startswith("iter"):
            iteration = int(part.replace("iter", ""))
        elif part.startswith("step"):
            global_step = int(part.replace("step", ""))
    return epoch, iteration, global_step

def mask_codes(codes, sch="cosine", mask = False, editing = False):
    """
    Applies masking to the target tokens for Masked Diffusion Loss.
    """
    r = random.uniform(0, 1)
    if len(codes) <= 5 and mask == False:
        mask_ratio=1.0
    elif sch=="cosine":
        mask_ratio = math.cos(r * math.pi / 2) # cosine scheduler
    elif sch=="linear":
        if r < 0.05:
            r = r + 0.05
        mask_ratio = r
    else:
        mask_ratio = 1.0 
        
    num_to_mask = int(len(codes) * mask_ratio)
    if num_to_mask < 1:
        num_to_mask = 1      
    indices_to_mask = random.sample(range(len(codes)), num_to_mask)
    masked_codes = codes[:]
    labels = [-100] * len(codes)
    for index in indices_to_mask:
        labels[index] = codes[index]
        masked_codes[index] = MASK # Use MASK variable
    return masked_codes, labels

def calculate_metrics(predictions, references, loaded_metrics=None):
    metrics = {}
    
    wer_metric = loaded_metrics.get("wer") if loaded_metrics else evaluate.load("wer")
    cer_metric = loaded_metrics.get("cer") if loaded_metrics else evaluate.load("cer")
    meteor_metric = loaded_metrics.get("meteor") if loaded_metrics else evaluate.load("meteor")
    
    preds_norm = [p.lower() for p in predictions]
    refs_norm = [r.lower() for r in references]
    
    valid_indices = [i for i, r in enumerate(refs_norm) if len(r.strip()) > 0]
    if not valid_indices:
        return {"wer": 1.0, "cer": 1.0, "meteor": 0.0, "bleu": 0.0, "edit_distance": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    preds_norm = [preds_norm[i] for i in valid_indices]
    refs_norm = [refs_norm[i] for i in valid_indices]
    
    try:
        metrics["wer"] = wer_metric.compute(predictions=preds_norm, references=refs_norm)
    except:
        metrics["wer"] = 1.0
        
    try:
        metrics["cer"] = cer_metric.compute(predictions=preds_norm, references=refs_norm)
    except:
        metrics["cer"] = 1.0
        
    try:
        metrics["meteor"] = meteor_metric.compute(predictions=preds_norm, references=refs_norm)["meteor"]
    except:
        metrics["meteor"] = 0.0

    try:
        bleu_scores = []
        for p, r in zip(preds_norm, refs_norm):
            bleu_scores.append(sentence_bleu([r.split()], p.split(), weights=(0.5, 0.5)))
        metrics["bleu"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    except:
        metrics["bleu"] = 0.0
        
    try:
        edit_dists = []
        for p, r in zip(preds_norm, refs_norm):
            edit_dists.append(nltk.edit_distance(p, r))
        metrics["edit_distance"] = sum(edit_dists) / len(edit_dists) if edit_dists else 0.0
    except:
        metrics["edit_distance"] = 0.0

    # Word-level Precision, Recall, F1
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
    except:
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0

    return metrics

# ==============================================================================
# 3. ItemProcessor (OCR)
# ==============================================================================
class ItemProcessorOCR(ItemProcessorBase):
    def __init__(self, tokenizer, max_len, und_image_size=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.und_image_size = und_image_size

    def process_item(self, data_item: dict, training_mode=False) -> Tuple[Any, str, str, str]:
        image = data_item.get('image')
        question = data_item.get('question', "Extract all text from the image.")
        answer = data_item.get('answer', "")
        image_name = data_item.get('image_name', "")

        crop_size_list = generate_crop_size_list((self.und_image_size // 32) ** 2, 32)
        image = var_edge_pad(image, crop_size_list=crop_size_list, pad_mode='edge')

        return (image, question, answer, image_name)

    def predict_item_token_length(self, data_item: dict) -> int:
        return 2048

# ==============================================================================
# 4. Dataset Wrapper
# ==============================================================================
class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, item_processor):
        self.dataset = hf_dataset
        self.item_processor = item_processor
        self.meta_collection = [
            {
                "type": "ocr",
                "len": len(hf_dataset),
                "ratio": 1.0,
                "path": "hf_dataset",
                "item_len_list": [2048] * len(hf_dataset) 
            }
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.item_processor.process_item(item, training_mode=True)

# ==============================================================================
# 5. Solver (Main Training Logic)
# ==============================================================================
class Solver(FinetuneSolverBase):
    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        parser.add_argument("--max_seq_len", default=2048, type=int, help="max token length")
        parser.add_argument("--dropout", type=float, default=0.05, help="dropout rate")
        parser.add_argument("--und_image_size", type=int, default=512, help="Image size for understanding preprocessing")
        parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU Offloading for FSDP")
        parser.add_argument("--validation_interval", type=int, default=100, help="Validation interval in global steps")
        parser.add_argument("--validation_samples", type=int, default=100, help="Number of validation samples to use")
        parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
        parser.add_argument("--wandb_project", type=str, default="lumina-ocr", help="WandB project name")
        parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity name")
        parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
        parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID for resume")
        parser.add_argument("--use_lora", action="store_true", help="Enable LoRA training")
        parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
        parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
        parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
        parser.add_argument("--wo_lm_head", action="store_true", help="Without LM head in LoRA")
        parser.add_argument("--dataset_path", type=str, default="Jiwon-Kang/Llama-Nemotron-VLM-Dataset-v1-OCR4", help="HF Dataset path")
        return parser
    
    def setup_fsdp_sync(self, model, data_parallel, precision, grad_precision=None):
        if data_parallel == "none": return model.cuda()
        if self.dp_rank != 0:
            param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)
        else:
            param_init_fn = None
        cpu_offload = CPUOffload(offload_params=self.args.cpu_offload)
        model = FSDP(
            model,
            auto_wrap_policy=functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list()) if not self.args.use_lora else None,
            process_group=fs_init.get_data_parallel_group(),
            sharding_strategy={"fsdp": ShardingStrategy.FULL_SHARD, "sdp": ShardingStrategy.SHARD_GRAD_OP}[data_parallel],
            mixed_precision=MixedPrecision(
                param_dtype={"fp32": torch.float, "tf32": torch.float, "bf16": torch.bfloat16, "fp16": torch.float16}[precision],
                reduce_dtype={"fp32": torch.float, "tf32": torch.float, "bf16": torch.bfloat16, "fp16": torch.float16}[grad_precision or precision],
            ),
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
            limit_all_gathers=True,
            use_orig_params=True,
            param_init_fn=param_init_fn,
            cpu_offload=cpu_offload,
        )
        torch.cuda.synchronize()
        return model

    def __init__(self, args):
        super().__init__(args)
        
        # Ensure NLTK data and Metrics are ready (Rank 0 only)
        if self.global_rank == 0:
            print("[Solver] Pre-loading NLTK data and metrics...")
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                # Pre-download metric scripts from HF
                evaluate.load("wer")
                evaluate.load("cer")
                evaluate.load("meteor")
            except Exception as e:
                print(f"[Warning] Failed to pre-download metrics: {e}")
        
        dist.barrier() # Wait for Rank 0 to finish
        
        if self.args.use_wandb and self.global_rank == 0:
            wandb_kwargs = {"project": self.args.wandb_project, "entity": self.args.wandb_entity, "config": vars(self.args)}
            if self.args.wandb_run_id:
                wandb_kwargs["id"] = self.args.wandb_run_id
                wandb_kwargs["resume"] = "allow"
            wandb_kwargs["name"] = self.args.wandb_run_name or f"ocr-run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(**wandb_kwargs)
        self.val_table = None
        self.global_step = 0

    def build_model(self):
        init_from = self.args.init_from
        unwrapped_model, tokenizer = self._model_func(init_from)
        if not self.args.use_lora:
            from xllmx.util.tensor_type import promote_param_to_fp32
            for key, param in unwrapped_model.named_parameters():
                param.requires_grad = True
                promote_param_to_fp32(param)
        misc.mark_mp_params(unwrapped_model)
        checkpointing_list = unwrapped_model.get_checkpointing_wrap_module_list() if hasattr(unwrapped_model, 'get_checkpointing_wrap_module_list') and self.args.checkpointing else []
        model = self.setup_fsdp_sync(unwrapped_model, self.args.data_parallel, self.args.precision, self.args.grad_precision)
        misc.broadcast_nonmp_parameters(model)
        if self.args.checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper
            non_reentrant_wrapper = functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=lambda m: m in checkpointing_list)
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        except:
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        return model, tokenizer, optimizer

    def _model_func(self, init_from: str):
        tokenizer = AutoTokenizer.from_pretrained(init_from, trust_remote_code=True)
        dtype = torch.bfloat16 if self.args.precision == "bf16" else torch.float32
        model = LLaDAForMultiModalGeneration.from_pretrained(
            init_from, 
            torch_dtype=dtype, 
            device_map="cpu",
        )
        if hasattr(model.config, 'scale_logits'): model.config.scale_logits = True
        if self.args.checkpointing: model.model.set_activation_checkpointing("whole_layer")
        if self.args.use_lora:
            from peft import LoraConfig, get_peft_model, PeftModel
            for param in model.parameters(): param.requires_grad = False
            if self.args.resume_path is None:
                target_modules = ["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"]
                if self.args.wo_lm_head: target_modules = [tm for tm in target_modules if tm != "ff_out"]
                lora_config = LoraConfig(r=self.args.lora_rank, lora_alpha=self.args.lora_alpha, target_modules=target_modules, lora_dropout=self.args.lora_dropout, bias="none", task_type="CAUSAL_LM")
                model = get_peft_model(model, lora_config)
            else:
                model = PeftModel.from_pretrained(model, self.args.resume_path, torch_dtype=dtype, is_trainable=True)
            model.print_trainable_parameters()
        return model, tokenizer

    def _dataset_func(self):
        # Use environment variable for local dataset directory (default to common path)
        local_dataset_dir = os.getenv('LOCAL_DATASET_DIR', '/mnt/data1/jiwon/Llama-Nemotron-VLM-Dataset-v1/convert_to_hf')
        local_path = os.path.join(local_dataset_dir, self.args.dataset_path.split('/')[-1])

        if os.path.exists(local_path):
            if self.global_rank == 0:
                print(f"[Dataset] Loading from local disk: {local_path}")
            from datasets import load_from_disk
            dataset = load_from_disk(local_path)
        else:
            if self.global_rank == 0:
                print(f"[Dataset] Loading from Hugging Face Hub: {self.args.dataset_path}")
            dataset = load_dataset(self.args.dataset_path)

        train_ds, self.val_ds = dataset["train"], dataset["validation"]
        item_processor = ItemProcessorOCR(self.tokenizer, self.args.max_seq_len, und_image_size=self.args.und_image_size)
        return HFDatasetWrapper(train_ds, item_processor)

    def _item_processor_func(self, tokenizer=None, max_len=None) -> ItemProcessorBase:
        return ItemProcessorOCR(tokenizer or self.tokenizer, max_len or self.args.max_seq_len, und_image_size=self.args.und_image_size)

    def _make_and_save_starting_point(self, save_path: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.args.init_from, trust_remote_code=True)
        base_config = AutoConfig.from_pretrained(self.args.init_from, trust_remote_code=True)
        model = LLaDAForMultiModalGeneration(base_config)
        model.resize_token_embeddings(len(tokenizer))
        if model.model.transformer.ff_out.out_features != len(tokenizer):
             model.model.transformer.ff_out = torch.nn.Linear(4096, len(tokenizer), bias=False)
        original_model = LLaDAForMultiModalGeneration.from_pretrained(self.args.init_from, torch_dtype=torch.bfloat16, device_map="cpu")
        model.load_state_dict(original_model.state_dict())
        model.save_pretrained(save_path); tokenizer.save_pretrained(save_path)

    def resume(self, resume_path: str):
        resume_epoch, resume_iteration, resume_global_step = parse_checkpoint_name(os.path.basename(resume_path))
        if resume_iteration is None: self.start_epoch, self.start_iter = resume_epoch + 1, 0
        else: self.start_epoch, self.start_iter = resume_epoch, resume_iteration + 1
        if resume_global_step: self.global_step = resume_global_step
        else:
            steps_per_epoch = len(self.dataloader_train) // self.args.accum_iter
            self.global_step = resume_epoch * steps_per_epoch + (resume_iteration // self.args.accum_iter if resume_iteration else 0)

    @torch.no_grad()
    def validate(self, epoch):
        """
        Validation with Multi-GPU support.
        Each GPU processes a subset of data, then results are gathered at rank 0.
        """
        import gc

        dist.barrier()

        local_rank = dist.get_rank()
        world_size = dist.get_world_size()

        self.model.eval()

        if not hasattr(self, 'vqvae'):
            dtype = torch.bfloat16 if self.args.precision == "bf16" else torch.float32
            self.vqvae = VQModel.from_pretrained(self.args.init_from, subfolder="vqvae", torch_dtype=dtype).to(f"cuda:{self.global_rank}")
            self.vqvae.eval()

        # Load metrics on all ranks
        loaded_metrics = {"wer": evaluate.load("wer"), "cer": evaluate.load("cer"), "meteor": evaluate.load("meteor")}

        # Distribute validation data across GPUs
        subset_indices = list(range(min(len(self.val_ds), self.args.validation_samples)))
        # subset_indices = list(range(min(len(self.val_ds), 10))) # Debug with 10 samples
        local_dataset_list = []

        for idx, i in enumerate(subset_indices):
            if idx % world_size == local_rank:  # Each GPU gets every Nth sample
                local_dataset_list.append(self.val_ds[i])

        if self.global_rank == 0:
            print(f"[Validation] Total samples: {len(subset_indices)}, Per GPU: ~{len(subset_indices)//world_size}")

        # Process local data
        local_predictions = []
        local_references = []
        local_image_names = []
        local_questions = []
        local_images = []  # Store actual images for wandb.Table

        for idx, item in tqdm(enumerate(local_dataset_list), total=len(local_dataset_list), desc=f"[Rank {local_rank}] Validation"):
            image, question, answer_gt, image_name = item['image'], item['question'], item['answer'], item['image_name']

            crop_size_list = generate_crop_size_list((self.args.und_image_size // 32) ** 2, 32)
            image_processed = var_edge_pad(image, crop_size_list=crop_size_list, pad_mode='edge')
            input_img_token, (H, W) = encode_img_with_breaks_fixed(image_processed, self.vqvae)
            img_token = [BOI] + add_break_line(input_img_token[1:-1], H, W, new_number=NEW_LINE) + [EOI]

            instruction = "<system>" + UNDERSTANDING_PROMPT_TEMPLATE + "</system>" + "<user>" + question + "</user>"
            input_ids_raw = self.tokenizer(instruction)['input_ids']
            input_token = input_ids_raw[:-1] + img_token + input_ids_raw[-1:]
            code_start = len(input_token) + 1
            STEPS_LENGTH= 128
            GEN_LENGTH=  512
            BLOCK_LENGTH= 128
            input_token = input_token + [BOA] + [MASK] * GEN_LENGTH
            input_ids = torch.tensor(input_token, device=f"cuda:{self.global_rank}").unsqueeze(0)

            out = generate_text_understanding(
                self.model, input_ids,
                steps=STEPS_LENGTH,
                gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH,
                temperature=0.0,
                cfg_scale=0.0,
                remasking='low_confidence',
                code_start=code_start
            )

            pred_text = self.tokenizer.batch_decode(out[:, code_start:], skip_special_tokens=True)[0].replace("</answer>", "").strip()
            
            print(f"Prediction: {pred_text} \n Ground Truth: {answer_gt}")
            print("=" * 50)

            # Store results locally
            local_predictions.append(pred_text)
            local_references.append(answer_gt)
            local_image_names.append(image_name)
            local_questions.append(question)
            local_images.append(image_processed)  # Store processed image for wandb.Table

            if local_rank == 0 and idx % 10 == 0:
                print(f"[Rank {local_rank}] Processed {idx+1}/{len(local_dataset_list)} samples")

            # Memory cleanup
            if idx % 2 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Gather results from all GPUs
        all_predictions = [None for _ in range(world_size)]
        all_references = [None for _ in range(world_size)]
        all_image_names = [None for _ in range(world_size)]
        all_questions = [None for _ in range(world_size)]
        all_images = [None for _ in range(world_size)]

        dist.barrier()
        dist.all_gather_object(all_predictions, local_predictions)
        dist.all_gather_object(all_references, local_references)
        dist.all_gather_object(all_image_names, local_image_names)
        dist.all_gather_object(all_questions, local_questions)
        dist.all_gather_object(all_images, local_images)

        # Rank 0: Aggregate and compute metrics
        if self.global_rank == 0:
            # Flatten gathered lists
            predictions = [p for sublist in all_predictions for p in sublist]
            references = [r for sublist in all_references for r in sublist]
            image_names = [n for sublist in all_image_names for n in sublist]
            questions = [q for sublist in all_questions for q in sublist]
            images = [img for sublist in all_images for img in sublist]

            print(f"[Validation] Gathered {len(predictions)} predictions from {world_size} GPUs")

            # Compute overall metrics
            avg_metrics = calculate_metrics(predictions, references, loaded_metrics=loaded_metrics)
            print(f"[Validation] Step {self.global_step} Metrics: WER={avg_metrics['wer']:.4f}, CER={avg_metrics['cer']:.4f}, BLEU={avg_metrics['bleu']:.4f}")

            # Log to WandB
            if self.args.use_wandb:
                # Create new table for this validation step (important: must create new table each time!)
                val_table = wandb.Table(columns=["Step", "Image", "Question", "GT", "Prediction", "WER", "CER"])

                print(f"[Validation] Adding {min(len(predictions), 10)} samples to wandb table")
                print(f"[DEBUG] len(images)={len(images)}, len(predictions)={len(predictions)}")

                # Add sample predictions to table (first 10 samples)
                rows_added = 0
                for i in range(min(len(predictions), 10)):
                    try:
                        ind_metrics = calculate_metrics([predictions[i]], [references[i]], loaded_metrics=loaded_metrics)

                        # Debug first image
                        if i == 0:
                            print(f"[DEBUG] First image type: {type(images[i])}, size: {images[i].size if hasattr(images[i], 'size') else 'N/A'}")

                        val_table.add_data(
                            self.global_step,
                            wandb.Image(images[i]),  # Use wandb.Image() instead of image_name string
                            questions[i][:50] + "..." if len(questions[i]) > 50 else questions[i],
                            references[i][:100] + "..." if len(references[i]) > 100 else references[i],
                            predictions[i][:100] + "..." if len(predictions[i]) > 100 else predictions[i],
                            ind_metrics["wer"],
                            ind_metrics["cer"]
                        )
                        rows_added += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to add row {i} to table: {e}")
                        import traceback
                        traceback.print_exc()

                print(f"[Validation] Successfully added {rows_added}/10 rows to table")

                # Log all validation metrics and table together in one call
                wandb.log({
                    "val/wer": avg_metrics["wer"],
                    "val/cer": avg_metrics["cer"],
                    "val/bleu": avg_metrics["bleu"],
                    "val/meteor": avg_metrics["meteor"],
                    "val/edit_distance": avg_metrics["edit_distance"],
                    "val/precision": avg_metrics["precision"],
                    "val/recall": avg_metrics["recall"],
                    "val/f1": avg_metrics["f1"],
                    "val/samples": val_table,  # Include table in same log call
                    "global_step": self.global_step
                })

        dist.barrier()
        self.model.train()

    def train_one_epoch(self, epoch, start_iter, log_writer=None, metric_logger=None):
        if not hasattr(self, 'vqvae'):
            dtype = torch.bfloat16 if self.args.precision == "bf16" else torch.float32
            self.vqvae = VQModel.from_pretrained(self.args.init_from, subfolder="vqvae", torch_dtype=dtype).to("cuda")
            self.vqvae.eval()
        
        self.model.train(True)
        
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
        
        accum_iter = self.args.accum_iter
        self.optimizer.zero_grad()
        accumulated_loss = 0.0
        
        header = "Epoch: [{}]".format(epoch)
        print_freq = 10
        
        for data_iter_step, batch_data in enumerate(
            metric_logger.log_every(
                self.dataloader_train,
                print_freq,
                header,
                start_iter,
                self.args.batch_size * fs_init.get_data_parallel_world_size(),
            ),
            start=start_iter,
        ):
            images, questions, answers, _ = batch_data
            input_ids_list, labels_list = [], []
        
            for i in range(len(images)):
                img, question, answer = images[i], questions[i], answers[i]
                with torch.no_grad():
                    image_tokens = encode_img_with_breaks(img, self.vqvae)
                
                instruction = "<system>" + UNDERSTANDING_PROMPT_TEMPLATE + "</system>" + "<user>" + question + "</user>"
                
                instruction_token = self.tokenizer(instruction, truncation=True, max_length=self.args.max_seq_len, return_tensors="pt").input_ids[0].tolist()
                
                # print(f"RANK{self.global_rank} Instruction Token Length:", len(instruction_token))
                # print(f"RANK{self.global_rank} Image Token Length:", len(image_tokens))
                instruction_token = instruction_token[:-1] + image_tokens + instruction_token[-1:]
                instruction_label = [-100] * len(instruction_token)
                
                answer_text = answer + "</answer>"
                answer_token = self.tokenizer(answer_text, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()
                
                # print(f"RANK{self.global_rank} Answer Token Length:", len(answer_token))

                # Token length check with minimum answer space guarantee
                MIN_ANSWER_LENGTH = 50  # Minimum answer tokens to prevent loss nan
                BOA_TOKEN = 1

                valid_answer_token_length = self.args.max_seq_len - len(instruction_token) - BOA_TOKEN

                if valid_answer_token_length < MIN_ANSWER_LENGTH:
                    # Instruction too long, truncate to make room for answer
                    print(f"RANK{self.global_rank} WARNING: Instruction too long ({len(instruction_token)} tokens), truncating to make room for answer")
                    max_instruction_length = self.args.max_seq_len - MIN_ANSWER_LENGTH - BOA_TOKEN
                    instruction_token = instruction_token[:max_instruction_length]
                    instruction_label = instruction_label[:max_instruction_length]
                    valid_answer_token_length = MIN_ANSWER_LENGTH

                if len(answer_token) > valid_answer_token_length:
                    print(f"RANK{self.global_rank} Truncating answer: {len(answer_token)} → {valid_answer_token_length}")
                    answer_token = answer_token[:valid_answer_token_length]

                answer_token, answer_label = mask_codes(answer_token)
                
                final_input = instruction_token + [BOA] + answer_token
                final_label = instruction_label + [-100] + answer_label
                
                # print(f"RANK{self.global_rank} Final Input Length after truncation:", len(final_input))
                # print("===================================================")
                
                
                input_ids_list.append(final_input) 
                labels_list.append(final_label)
            
            lr_sched.adjust_learning_rate_epoch(self.optimizer, data_iter_step / len(self.dataloader_train) + epoch, self.args)
            
            with {"bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16), "fp16": torch.cuda.amp.autocast(dtype=torch.float16), "fp32": contextlib.nullcontext(), "tf32": contextlib.nullcontext()}[self.args.precision]:
                loss = self.model(input_ids=input_ids_list, labels=labels_list)
            
            loss_value = loss.item()
            accumulated_loss += loss_value
            
            if not math.isfinite(loss_value):
                print(f"[Rank {self.global_rank}] Loss is {loss_value}, stopping training")
                print(f"[Rank {self.global_rank}] Input IDs (first sample, first 50): {input_ids_list[0][:50]}")
                print(f"[Rank {self.global_rank}] Input IDs (first sample, Last 50): {input_ids_list[0][-50:]}")
                print(f"[Rank {self.global_rank}] Labels (first sample, First 50): {labels_list[0][:50]}")
                print(f"[Rank {self.global_rank}] Labels (first sample, Last 50): {labels_list[0][-50:]}")
                sys.exit(1)
            
            (loss / accum_iter).backward()
            
            if (data_iter_step + 1) % accum_iter == 0:
                if isinstance(self.model, FSDP): 
                    self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
                else: 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad)
                
                self.optimizer.step() 
                self.optimizer.zero_grad(set_to_none=True) 
                self.global_step += 1
                
                if self.global_rank == 0 and self.args.use_wandb:
                    avg_loss = accumulated_loss / accum_iter
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/global_step": self.global_step,
                        "train/epoch": epoch + (data_iter_step / len(self.dataloader_train))
                     })
                if self.global_step % self.args.validation_interval == 0: 
                    self.validate(epoch)
                if self.global_step % self.args.save_iteration_interval == 0: 
                    self.save_checkpoint(epoch, iteration=data_iter_step, global_step=self.global_step)
            
            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def run(self):
        steps_per_epoch = len(self.dataloader_train) // self.args.accum_iter
        self.global_step = self.start_epoch * steps_per_epoch + (self.start_iter // self.args.accum_iter)
        print(f"[Solver] Starting OCR Training from step {self.global_step}")
        
        self.save_checkpoint(self.start_epoch)
        self.validate(self.start_epoch)
        
        self.logger.info(f"Start training for {self.args.epochs} epochs")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.args.epochs):
            self.dataloader_train.sampler.set_epoch(epoch, self.start_iter)

            train_stats = self.train_one_epoch(
                epoch,
                self.start_iter,
                log_writer=self.log_writer,
                metric_logger=self.metric_logger_to_resume,
            )

            # End of Epoch Save (optional, since we now save by step, but good to keep)
            if epoch % self.args.save_interval == 0 or epoch + 1 == self.args.epochs:
                 # Check if we just saved to avoid double saving
                 if self.global_step % self.args.save_iteration_interval != 0:
                    self.save_checkpoint(epoch=epoch)

            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}

            if self.global_rank == 0:
                if self.log_writer is not None:
                    self.log_writer.flush()
                with open(os.path.join(self.args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if self.args.use_wandb:
                    wandb.log(log_stats)

            self.start_iter = 0
            self.metric_logger_to_resume = None

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info("Training time {}".format(total_time_str))

        if self.global_rank == 0 and self.args.use_wandb: wandb.finish()
            
    def save_checkpoint(self, epoch, iteration=None, global_step=None):
        print(f"[Solver] Saving checkpoint at global step {self.global_step}")

        # Check if FSDP or regular model
        is_fsdp = isinstance(self.model, FSDP)

        if is_fsdp:
            util.ckpt.save(
                self.args.output_dir,
                self.global_rank == 0,
                self.model,
                self.optimizer,
                self.tokenizer,
                self.args,
                epoch=epoch,
                iteration=iteration,
                global_step=global_step,
                max_keep=self.args.ckpt_max_keep,
            )
            # Handle rotation (remove old checkpoints)
            util.ckpt.remove_early_ckpts(self.args.output_dir, max_keep=self.args.ckpt_max_keep)
        else:
            # Handle Non-FSDP / LoRA Saving
            if self.global_rank == 0:
                save_name = f"epoch{epoch}"
                if iteration is not None:
                    save_name += f"-iter{iteration}"
                if global_step is not None:
                    save_name += f"-step{global_step}"
                save_dir = os.path.join(self.args.output_dir, save_name)
                os.makedirs(save_dir, exist_ok=True)

                if self.args.use_lora:
                    # Save Adapter Only
                    self.model.save_pretrained(save_dir)
                    print(f"[Solver] Saved LoRA adapter to {save_dir}")
                else:
                    # Save Full Model
                    torch.save(self.model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
                    print(f"[Solver] Saved full model to {save_dir}")

                # Save Tokenizer and Args
                self.tokenizer.save_pretrained(save_dir)
                with open(os.path.join(save_dir, "args.json"), "w") as f:
                    json.dump(vars(self.args), f, indent=2)

        dist.barrier()

if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    util.misc.random_seed(42)
    solver = Solver(args)
    solver.run()