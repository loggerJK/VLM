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
from sklearn.metrics import confusion_matrix
import io
import warnings
from PIL import Image

# HuggingFace & Diffusers
from datasets import load_dataset, concatenate_datasets
from diffusers import VQModel
from transformers import AutoTokenizer, AutoConfig

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
from utils.ocr_render import generate_image as generate_ocr_image
from utils.image_utils import encode_img_with_breaks, generate_crop_size_list, var_center_crop, var_edge_pad, encode_img_with_breaks_fixed, add_break_line, decode_vq_to_image, calculate_vq_params
# Generation Utils
from generators.text_understanding_generator import generate_text_understanding
from generators.image_generation_generator import generate_image
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates

from transformers import enable_full_determinism

# Metrics
import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu

from copy import deepcopy

# ==============================================================================
# 1. Special Tokens Global Definition
# ==============================================================================
MASK = SPECIAL_TOKENS["mask_token"]
NEW_LINE = SPECIAL_TOKENS["newline_token"]
BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
EOA = SPECIAL_TOKENS["answer_end"]    # End of Answer
BOI = SPECIAL_TOKENS["boi"]           # Begin of Image
EOI = SPECIAL_TOKENS["eoi"]           # End of Image
PAD = 126339                          # Padding token (not in config.py, from train.py)
IMAGE_TOKEN_OFFSET = SPECIAL_TOKENS["image_token_offset"]
UNDERSTANDING_PROMPT_TEMPLATE = PROMPT_TEMPLATES["text_understanding"]

# ==============================================================================
# 2. Helper Functions
# ==============================================================================

def parse_checkpoint_name(ckpt_str: str):
    """
    Parse epoch, iteration, and global_step from checkpoint directory name.
    
    Examples:
        - "epoch3" -> (3, None, None)
        - "epoch3-iter120" -> (3, 120, None)
        - "epoch3-iter120-step450" -> (3, 120, 450)
    """
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

def extract_number(text):
    """Extract number from text pattern **number** or just number"""
    match = re.search(r"\*\*(\d+)\*\*", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return -1

def extract_number_fixed(text):
    """Extract number from text pattern **number**, number, or English words (zero-nine)."""
    text = text.lower()
    
    # 1. Try **number**
    match = re.search(r"\*\*(\d+)\*\*", text)
    if match:
        return int(match.group(1))
    
    # 2. Try English words (zero to nine)
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10
    }
    for word, num in word_to_num.items():
        # Match whole word to avoid partial matches (e.g. 'one' in 'bone')
        if re.search(r"\b" + word + r"\b", text):
            return num

    # 3. Try plain digits
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
        
    return -1

import re

def check_pos_accuracy(ground_truth: str, prediction: str) -> bool:
    pattern = r'(top-left|top-right|bottom-left|bottom-right)'
    
    gt_matches = re.findall(pattern, ground_truth.lower())
    pred_matches = re.findall(pattern, prediction.lower())
    
    if not gt_matches or not pred_matches:
        warnings.warn(f"위치를 파싱할 수 없음 — GT: {gt_matches}, Pred: {pred_matches}")
        return False
    
    if len(pred_matches) > 1:
        warnings.warn(f"Prediction에 여러 위치 감지: {pred_matches}, False 리턴")
        return False
    
    return gt_matches[0] == pred_matches[0]

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
# 3. ItemProcessor (CPU Stage)
# ==============================================================================
class ItemProcessorUnderstandingGeneration(ItemProcessorBase):
    """
    Preprocesses raw items from Hugging Face Dataset for Text-to-Image Generation.
    - Resizes/Crops Image (CPU)
    - Extracts Caption/Prompt
    """
    def __init__(self, tokenizer, max_len, und_image_size=512, gen_image_size=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.und_image_size = und_image_size
        self.gen_image_size = gen_image_size

    def process_item(self, data_item: dict, training_mode=False, task='counting') -> Tuple[Any, str]:
        # #agent edited: [3-1] Dataset column mapping (Adjust based on your dataset)
        # Expected format: {'image': PIL.Image, 'text': str} or {'image': ..., 'caption': ...}
        image = data_item.get('image')
        
        # Check key 'descriptions' first for understanding generation
        caption = data_item.get('descriptions', None)
        if isinstance(caption, list):
            caption = caption[0]
        
        # Generation Data
        if caption is not None:
            if task == 'ocr_synthetic': # Generate synthetic OCR image HERE
                answer = data_item.get('answer', '')
                image = generate_ocr_image('# ' + answer, template="clean_light", width=self.gen_image_size, height=self.gen_image_size, quality=100)
            else :
                crop_size_list = generate_crop_size_list((self.gen_image_size // 32) ** 2, 32)
                if task == 'ocr':
                    image = var_edge_pad(image, crop_size_list=crop_size_list, pad_mode='edge')
                else:
                    image = var_center_crop(image, crop_size_list=crop_size_list) # counting, pointing, position
            return (image, caption)

        # Understanding Data
        else:
            if task == 'ocr':
                question = data_item.get('question', "Extract all text from the image.")
                answer = data_item.get('answer', "")
                crop_size_list = generate_crop_size_list((self.und_image_size // 32) ** 2, 32)
                image = var_edge_pad(image, crop_size_list=crop_size_list, pad_mode='edge')
            elif task == 'ocr_synthetic':
                answer = data_item['answer']
                question = "Extract all text from the image."
                image = generate_ocr_image('# ' + answer, template="random", width=self.und_image_size, height=self.und_image_size, quality=100)
            elif task in ['pointing', 'position', 'rel_position']:
                question = data_item.get('question', data_item.get('text', ''))
                answer = str(data_item.get('answer', data_item.get('label', '')))
                crop_size_list = generate_crop_size_list((self.und_image_size // 32) ** 2, 32)
                image = var_center_crop(image, crop_size_list=crop_size_list)
            elif task == 'celeb':
                question = data_item.get('question', '')
                answer = str(data_item.get('answer', ''))
                crop_size_list = generate_crop_size_list((self.und_image_size // 32) ** 2, 32)
                image = var_center_crop(image, crop_size_list=crop_size_list)
            else:  # counting
                question = data_item.get('question_count', data_item.get('text', ''))
                answer = str(data_item.get('answer_count', data_item.get('label', '')))
                crop_size_list = generate_crop_size_list((self.und_image_size // 32) ** 2, 32)
                image = var_center_crop(image, crop_size_list=crop_size_list)

            return (image, question, answer)
            

    def predict_item_token_length(self, data_item: dict) -> int:
        return 1024
    
    

# ==============================================================================
# 4. Dataset Wrapper
# ==============================================================================
class HFDatasetWrapper(torch.utils.data.Dataset):
    """
    Wraps Hugging Face Dataset to be compatible with xllmx Sampler.
    """
    def __init__(self, hf_gen_dataset, hf_und_dataset, item_processor, default_task='counting', mode='both'):
        self.gen_dataset = hf_gen_dataset
        self.und_dataset = hf_und_dataset
        self.item_processor = item_processor
        self.default_task = default_task
        self.mode = mode

        # If gen dataset is None but mode is 'both', fall back to 'und'
        if hf_gen_dataset is None and mode == 'both':
            mode = 'und'
            self.mode = mode

        if dist.get_rank() == 0:
            print(f"[INFO] Dataset Mode: {mode}")
        if mode == 'both':
            self.dataset = concatenate_datasets([hf_und_dataset, hf_gen_dataset])
            self.meta_collection = [
                {
                    "type": "und",
                    "len": len(hf_und_dataset),
                    "ratio": 1.0,
                    "path": "hf_dataset",
                    "item_len_list": [1024] * len(hf_und_dataset) # Used for length clustering only
                },
                {
                    "type": "gen",
                    "len": len(hf_gen_dataset),
                    "ratio": 1.0,
                    "path": "hf_dataset",
                    "item_len_list": [1024] * len(hf_gen_dataset)
                }
            ]
        elif mode == 'und':
            self.dataset = hf_und_dataset
            self.meta_collection = [
                {
                    "type": "und",
                    "len": len(hf_und_dataset),
                    "ratio": 1.0,
                    "path": "hf_dataset",
                    "item_len_list": [1024] * len(hf_und_dataset)
                }
            ]
        elif mode == 'gen':
            self.dataset = hf_gen_dataset
            self.meta_collection = [
                {
                    "type": "gen",
                    "len": len(hf_gen_dataset),
                    "ratio": 1.0,
                    "path": "hf_dataset",
                    "item_len_list": [1024] * len(hf_gen_dataset)
                }
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        for attempt in range(3):
            try:
                item = self.dataset[idx]
                task = item.get('task', self.default_task) # Get task from item, fallback to default_task
                return self.item_processor.process_item(item, training_mode=True, task=task)
            except Exception as e:
                print(f"[HFDataset Try #{attempt}] Failed sample {idx}: {e}")
                idx = random.randint(0, len(self) - 1)
                time.sleep(1)
        item = self.dataset[idx]
        task = item.get('task', self.default_task)
        return self.item_processor.process_item(item, training_mode=True, task=task)


import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.distributed as dist

# ==============================================================================
# 5. Solver (Main Training Logic)
# ==============================================================================
class Solver(FinetuneSolverBase):
    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        parser.add_argument("--max_seq_len", default=1024, type=int, help="max token length")
        parser.add_argument("--dropout", type=float, default=0.05, help="dropout rate")
        parser.add_argument("--und_image_size", type=int, default=512, help="Image size for understanding preprocessing (default: 512)")
        parser.add_argument("--gen_image_size", type=int, default=1024, help="Image size for generation preprocessing (default: 1024)")
        parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU Offloading for FSDP (default: False)")
        
        # Validation
        parser.add_argument("--validation_interval", type=int, default=50, help="Validation interval in global steps")
        parser.add_argument("--validation_as_pointing_format", action="store_true", help="Use pointing format for validation")

        # WandB Arguments
        parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
        parser.add_argument("--wandb_project", type=str, default="lumina-dimoo-finetune", help="WandB project name")
        parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity name")
        parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")

        # LoRA Arguments
        parser.add_argument("--use_lora", action="store_true", help="Enable LoRA training")
        parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
        parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
        parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

        # Training Arguments
        parser.add_argument("--use_compile", action="store_true", help="Enable torch.compile for training speedup")
        parser.add_argument("--wo_lm_head", action="store_true", help="Without LM head in LoRA (for memory saving)")
        
        # Task Argument
        parser.add_argument("--task", type=str, default="counting", choices=["counting", "pointing", "ocr", "ocr_synthetic", "position", "rel_position", "celeb"], help="Task type for understanding")
        parser.add_argument("--dataset_path", type=str, default=None, help="HF Dataset path (used for OCR task)")
        parser.add_argument("--validation_samples", type=int, default=100, help="Number of validation samples to use")
        parser.add_argument("--mode", type=str, default='und', choices=['und', 'gen', 'both'], help="Mode for text understanding/generation/both")
        
        # Wandb Resume
        parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID for resume")
        
        # Dataset arguments
        parser.add_argument("--count_upper_limit", type=int, default=None, help="Upper limit for count. 20 means count<=20.")
        parser.add_argument("--count_lower_limit", type=int, default=None, help="Lower limit for count. 0 means count>=0.")
        parser.add_argument("--eval_everything", action="store_true",
                            help="Run both understanding and generation validation regardless of --mode")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode with smaller dataset and more frequent validation")

        return parser
    
    def _get_fsdp_wrap_policy(self, model):
        if self.args.use_lora:
            # peft's fsdp_auto_wrap_policy creates an OR policy:
            # 1) lambda_policy: wraps trainable leaf modules (lora_A, lora_B) as separate FSDP units
            # 2) transformer_policy: wraps transformer blocks as FSDP units
            # This separates frozen base params from trainable LoRA params,
            # eliminating wasted gradient buffer allocation and reduce-scatter communication.
            #
            # Dynamically resolve block class names from the actual model instance,
            # since only one block type (e.g. LLaDALlamaBlock) is instantiated
            # depending on config.block_type.
            # Dynamic resolution (use if block_type changes):
            # base_model = model.base_model.model
            # block_modules = base_model.get_checkpointing_wrap_module_list()
            # cls_names = ",".join({type(m).__name__ for m in block_modules})
            os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "LLaDALlamaBlock"
            from peft.utils.other import fsdp_auto_wrap_policy
            return fsdp_auto_wrap_policy(model)
        else:
            return functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
            )

    def setup_fsdp_sync(
        self, model: nn.Module, data_parallel: str, precision: str, grad_precision: str = None
    ) -> Union[FSDP, nn.Module]:                                                                             
        if data_parallel == "none":                                                                                                 
           print("[Solver] Data parallel is 'none'. Bypassing FSDP and moving model to CUDA.")       
           return model.cuda()

        print(f"[Solver] setup_fsdp_sync: cpu_offload={self.args.cpu_offload}")
        
        if self.dp_rank == 0:
            param_init_fn = None
        else:
            param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)

        # Set CPU Offload based on argument
        cpu_offload = CPUOffload(offload_params=self.args.cpu_offload)

        model = FSDP(
            model,
            auto_wrap_policy=self._get_fsdp_wrap_policy(model),
            process_group=fs_init.get_data_parallel_group(),
            sharding_strategy={
                "fsdp": ShardingStrategy.FULL_SHARD,
                "sdp": ShardingStrategy.SHARD_GRAD_OP,
            }[data_parallel],
            mixed_precision=MixedPrecision(
                param_dtype={
                    "fp32": torch.float,
                    "tf32": torch.float,
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                }[precision],
                reduce_dtype={
                    "fp32": torch.float,
                    "tf32": torch.float,
                    "bf16": torch.bfloat16,
                    "fp16": torch.float16,
                }[grad_precision or precision],
            ),
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
            limit_all_gathers=True,
            use_orig_params=True,
            param_init_fn=param_init_fn,
            cpu_offload=cpu_offload, # ENABLED
        )
        torch.cuda.synchronize()

        return model

    def __init__(self, args):
        super().__init__(args)
        
        # Pre-download NLTK data and metrics for OCR task
        if self.args.task in ['ocr', 'ocr_synthetic']:
            if self.global_rank == 0:
                print("[Solver] Pre-loading NLTK data and metrics for OCR...")
                try:
                    nltk.download('wordnet', quiet=True)
                    nltk.download('punkt', quiet=True)
                    nltk.download('omw-1.4', quiet=True)
                    evaluate.load("wer")
                    evaluate.load("cer")
                    evaluate.load("meteor")
                except Exception as e:
                    print(f"[Warning] Failed to pre-download metrics: {e}")
            dist.barrier()

        # Initialize WandB with resume support
        if self.args.use_wandb and self.global_rank == 0:
            wandb_kwargs = {
                "project": self.args.wandb_project,
                "entity": self.args.wandb_entity,
                "config": vars(self.args),
                # "mode": "online"
            }

            # If resuming with a run_id, use resume="allow"
            if self.args.wandb_run_id:
                wandb_kwargs["id"] = self.args.wandb_run_id
                wandb_kwargs["resume"] = "allow"
                wandb_kwargs["name"] = self.args.wandb_run_name  # Can be None
                self.logger.info(f"[WandB] Resuming run with ID: {self.args.wandb_run_id}")
            else:
                wandb_kwargs["name"] = self.args.wandb_run_name or f"run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

            wandb.init(**wandb_kwargs)

            # Save run_id for future reference
            if self.args.wandb_run_id is None:
                self.logger.info(f"[WandB] New run created with ID: {wandb.run.id}")
                
        self.global_step = 0

    def build_model(self):
        # Full override of build_model to handle LoRA freezing correctly without relying on FinetuneSolverBase's logic
        # init_from = self.args.resume_path or self.args.init_from
        init_from = self.args.init_from
        
        
        if init_from is None:
            starting_point_path = Path(self.args.output_dir) / "starting_point"
            if dist.get_rank() == 0:
                if (starting_point_path / "config.json").exists():
                    self.logger.info(f"will use existing starting point at {starting_point_path}")
                else:
                    self.logger.info(f"creating starting-point weights at {starting_point_path}")
                    self._make_and_save_starting_point(save_path=str(starting_point_path))
            dist.barrier()
            init_from = str(starting_point_path)

        self.logger.info(f"Start instantiating unwrapped model from {init_from}")
        
        # 1. Instantiate Model (and apply LoRA if enabled)
        unwrapped_model, tokenizer = self._model_func(init_from)

        # 2. Handle Trainable Parameters (Freezing Logic)
        if self.args.use_lora:
            # LoRA adapters are initialized in float32 by default, but FSDP requires
            # uniform dtype within each wrap unit. Cast entire model to the mixed
            # precision dtype so base params (bf16) and LoRA params (fp32) are unified.
            unwrapped_model = unwrapped_model.to(self.mixed_precision_dtype)
        else:
            # Original Logic for Full Finetune
            from xllmx.util.tensor_type import promote_param_to_fp32
            if hasattr(unwrapped_model, "get_trainable_params"):
                trainable_params = dict(unwrapped_model.get_trainable_params())
                for key, param in unwrapped_model.named_parameters():
                    if key in trainable_params:
                        param.requires_grad = True
                        promote_param_to_fp32(param)
                    else:
                        param.requires_grad = False
                        keep_fp32_keywords = ["norm", "lm_head", "embed_tokens"]
                        if any([_ in key for _ in keep_fp32_keywords]):
                            promote_param_to_fp32(param)
                        elif param.is_floating_point():
                            param.data = param.data.to(self.mixed_precision_dtype)
            else:
                # Default: All Trainable
                self.logger.warning(
                    f"model class {type(unwrapped_model)} does not have `get_trainable_params` method,"
                    f"set all params to trainable"
                )
                for key, param in unwrapped_model.named_parameters():
                    param.requires_grad = True
                    promote_param_to_fp32(param)

        self.logger.info("Finish instantiating unwrapped model.")
        
        misc.mark_mp_params(unwrapped_model)
        # misc.print_param_status(unwrapped_model)
        
        # =======================================================
        # total trainable params
        # =======================================================
        train_param_count_local, train_param_count_all = 0, 0
        frozen_param_count_local, frozen_param_count_all = 0, 0
        for name, param in unwrapped_model.named_parameters():
            model_parallel = getattr(param, "model_parallel", False)
            if param.requires_grad:
                if model_parallel:
                    train_param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                else:
                    train_param_count_all += param.numel()
                train_param_count_local += param.numel()
            else:
                if model_parallel:
                    frozen_param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                else:
                    frozen_param_count_all += param.numel()
                frozen_param_count_local += param.numel()
        self.logger.info(
            f"Trainable parameter count : {train_param_count_local} (local rank), {train_param_count_all} (all).\n"
            f"Frozen parameter count : {frozen_param_count_local} (local rank), {frozen_param_count_all} (all)."
            f"Trainable ratio: {train_param_count_all / (train_param_count_all + frozen_param_count_all)*100:.6f}%"
        )


        # 3. Checkpointing (Part 1)
        if self.args.checkpointing:
            checkpointing_list = unwrapped_model.get_checkpointing_wrap_module_list() if hasattr(unwrapped_model, 'get_checkpointing_wrap_module_list') else []
        else:
            checkpointing_list = []

        # 4. FSDP Wrapping
        model = self.setup_fsdp_sync(
            unwrapped_model, self.args.data_parallel, self.args.precision, self.args.grad_precision
        )

        # broadcast non-model-parallel parameters within model parallel group
        misc.broadcast_nonmp_parameters(model)

        # 5. Checkpointing (Part 2)
        if self.args.checkpointing:
            print("apply gradient checkpointing")
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper
            )
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            
            # Helper to check if module is in the list
            # For LoRA, the structure might change, so we might need a more robust check.
            # But get_checkpointing_wrap_module_list returns actual module objects, so it should work if they persist.
            if checkpointing_list:
                target_types = tuple(set(type(m) for m in checkpointing_list))
            else:
                target_types = ()

            def check_fn(submodule):
                return isinstance(submodule, target_types)

            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=check_fn,
            )

        # 6. torch.compile
        if self.args.use_compile:
            print("[Solver] Applying torch.compile (dynamic=True)...")
            model = torch.compile(model, dynamic=True)

        self.logger.info(f"Wrapped model: \n{str(model)}")

        # 7. Optimizer
        try:
            import bitsandbytes as bnb
            print("[Solver] Using bitsandbytes.optim.AdamW8bit Optimizer...")
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, betas=(0.9, 0.95))
        except (ImportError, RuntimeError):
            print("[Solver] bitsandbytes not found or failed to load, falling back to Standard torch.optim.AdamW...")
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, betas=(0.9, 0.95))
        
        return model, tokenizer, optimizer

    def _model_func(self, init_from: str) -> (nn.Module, None):
        """
        Resume cases:
        - LoRA resume: Load base model from init_from, then load adapter from resume_path
        - FSDP resume: Load full model from resume_path via from_pretrained
        """
        tokenizer = AutoTokenizer.from_pretrained(init_from, trust_remote_code=True)
        
        # Determine dtype based on precision arg
        if self.args.precision == "bf16":
            dtype = torch.bfloat16
        elif self.args.precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
            
        print(f"[Solver] Loading model in {dtype} (precision: {self.args.precision})...")
        model = LLaDAForMultiModalGeneration.from_pretrained(init_from, torch_dtype=dtype, device_map="cpu")
        
        # Force scale_logits to True to prevent NaN
        if hasattr(model.config, 'scale_logits'):
            print(f"[Solver] Changing scale_logits from {model.config.scale_logits} to True")
            model.config.scale_logits = True
        
        if self.args.checkpointing:
            print("[Solver] Enabling Activation Checkpointing...")
            model.model.set_activation_checkpointing("whole_layer")

        # --- LoRA Injection ---
        if self.args.use_lora :
            # Freeze base model parameters
            for param in model.parameters():
                param.requires_grad = False
                
            # 새로 Init 하는 경우
            if self.args.resume_path is None:
                print(f"[Solver] Applying LoRA with rank: {self.args.lora_rank}, alpha: {self.args.lora_alpha}, dropout: {self.args.lora_dropout}")
                from peft import LoraConfig, get_peft_model, TaskType
                
                target_modules = ["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"]
                if self.args.wo_lm_head:
                    target_modules = [tm for tm in target_modules if tm != "ff_out"]
                print(f"[Solver] LoRA Target Modules: {target_modules}")
                
                lora_config = LoraConfig(
                    r=self.args.lora_rank,
                    lora_alpha=self.args.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=self.args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM", # Using CAUSAL_LM as generic base, though it's multimodal
                    modules_to_save=[] # Add if needed
                )
                model = get_peft_model(model, lora_config)
                
            
            # Lora만 활성화
            elif self.args.resume_path is not None:
                from peft import PeftModel
                print(f"[Solver] Resuming LoRA from {self.args.resume_path}...")
                # Then load LoRA adapter

                model = PeftModel.from_pretrained(model, self.args.resume_path, torch_dtype=dtype, is_trainable=True, torch_device="cpu")
                # model.load_adapter(self.args.resume_path, is_trainable=True)
            model.print_trainable_parameters()
            
        return model, tokenizer
    
    def resume(self, resume_path: str):
        """
        Resume training from a checkpoint.
        _model_func handles optimizer, epoch/iter/step, and metric_logger.
        """
        self.logger.info(f"[Resume] >>> Entering resume() method")
        self.logger.info(f"[Resume] Resuming from checkpoint: {resume_path}")
        print(f"[Resume] >>> Entering resume() method for: {resume_path}")

        is_fsdp = isinstance(self.model, FSDP)
        self.logger.info(f"[Resume] Model is FSDP: {is_fsdp}")
        print(f"[Resume] Model is FSDP: {is_fsdp}")

        # Parse epoch, iteration, and global_step from checkpoint name
        ckpt_name = os.path.basename(resume_path)
        resume_epoch, resume_iteration, resume_global_step = parse_checkpoint_name(ckpt_name)
        self.logger.info(f"[Resume] Parsed checkpoint: epoch={resume_epoch}, iter={resume_iteration}, step={resume_global_step}")
        print(f"[Resume] Parsed checkpoint: epoch={resume_epoch}, iter={resume_iteration}, step={resume_global_step}")

        # Set start_epoch and start_iter
        if resume_iteration is None:
            self.start_epoch = resume_epoch + 1
            self.start_iter = 0
        else:
            self.start_epoch = resume_epoch
            self.start_iter = resume_iteration + 1

        # Set global_step
        if resume_global_step is not None:
            self.global_step = resume_global_step
        else:
            # Estimate from epoch/iter
            steps_per_epoch = len(self.dataloader_train) // self.args.accum_iter
            self.global_step = resume_epoch * steps_per_epoch
            if resume_iteration:
                self.global_step += resume_iteration // self.args.accum_iter

        self.logger.info(f"[Resume] Will start from epoch={self.start_epoch}, iter={self.start_iter}, global_step={self.global_step}")
        print(f"[Resume] Will start from epoch={self.start_epoch}, iter={self.start_iter}, global_step={self.global_step}")


    def _item_processor_func(self, tokenizer=None, max_len=None) -> ItemProcessorBase:
        # Pass image_size from args
        # return ItemProcessorPointing(tokenizer, max_len, image_size=self.args.image_size)
        return ItemProcessorUnderstandingGeneration(tokenizer, max_len, und_image_size=self.args.und_image_size, gen_image_size=self.args.gen_image_size)
    
    def _setup_validation_prompts(self):
        # All ranks should run this to have consistent prompts for FSDP generation
        if self.args.task == 'ocr':
            if self.global_rank == 0:
                print(f"[Solver] Setting up validation prompts from OCR dataset...")

            val_ds = self.val_ds
            rng = random.Random(42)
            indices = rng.sample(range(len(val_ds)), min(10, len(val_ds)))

            self.validation_prompts = []
            for idx in indices:
                item = val_ds[idx]
                answer = item.get('answer', '')
                caption = f"An image containing the text: {answer}" if answer else "Generate an image."
                self.validation_prompts.append(caption)
                
        elif self.args.task == 'ocr_synthetic':
            self.validation_prompts = []
            val_ds = self.val_ds
            rng = random.Random(42)
            indices = rng.sample(range(len(val_ds)), min(10, len(val_ds)))
            for idx in indices:
                item = val_ds[idx]
                answer = item.get('answer', '')
                caption = (
                    f"A Mathpix Markdown format with sharp, legible black text. "
                    f"High-resolution typography, top-down view. "
                    f"The text is rendered in natural left-to-right, top-to-bottom reading order. "
                    # f"Use Mathpix Markdown format: tables as LaTeX, mathematical expressions in LaTeX notation, "
                    # f"and keep all tables and captions at the end. "
                    f"The text reads:\n"
                    f"{answer}") if answer else "Generate an image."
                self.validation_prompts.append(caption)
                
        elif self.args.task in ['position', 'rel_position']:
            val_ds = self.val_ds
            rng = random.Random(42)
            indices = rng.sample(range(len(val_ds)), min(10, len(val_ds))) # 최대 10개 샘플링

            self.validation_prompts = []
            for idx in indices:
                item = val_ds[idx]
                answer = item.get('answer', '')
                self.validation_prompts.append(answer)

        elif self.args.task == 'celeb':
            persons = ["Heidi", "Samuel", "Elizabeth", "Benjamin", "Gabriel", "Julian"]
            self.validation_prompts = [f"Generate an image of {p}." for p in persons for _ in range(2)]

        else:
            if self.global_rank == 0:
                print(f"[Solver] Setting up validation prompts from heez/pixmo-point-count-gen-und...")

            val_ds = load_dataset('heez/pixmo-point-count-gen-und', split="val_gen")

            rng = random.Random(42)
            indices = rng.sample(range(len(val_ds)), min(10, len(val_ds)))

            self.validation_prompts = []
            for idx in indices:
                item = val_ds[idx]
                caption = item.get('descriptions', "Generate an image.")
                if isinstance(caption, list):
                    caption = caption[0]
                self.validation_prompts.append(caption)
            
        print(f"self.validation_prompts: {self.validation_prompts}")
        
        seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(self.args.gen_image_size, self.args.gen_image_size)
        self.validation_params = {
            "seq_len": seq_len,
            "newline_every": newline_every,
            "token_grid_height": token_grid_height,
            "token_grid_width": token_grid_width
        }

        if self.global_rank == 0:
            print(f"[Solver] Selected {len(self.validation_prompts)} validation prompts:")
            for i, p in enumerate(self.validation_prompts):
                print(f"  {i+1}. {p}")
        
        # save validation prompts to output_dir for reference
        if self.global_rank == 0:
            with open(os.path.join(self.args.output_dir, "validation_prompts.txt"), "w", encoding="utf-8") as f:
                for p in self.validation_prompts:
                    f.write(p + "\n")
        dist.barrier() # Ensure all ranks are ready before moving on

    @torch.no_grad()
    def log_validation_images(self, step):
        # All ranks must participate because model(infer=True) inside generate_image
        # uses collective communication (FSDP AllGather).
        # Prompts are distributed round-robin across ranks for parallel generation.

        if not hasattr(self, 'validation_prompts'):
            self._setup_validation_prompts()

        from utils.generation_utils import setup_seed

        templates = create_prompt_templates()
        print(f"[Rank {self.global_rank}] [Validation] Generating images at step {step}...")
        self.model.eval()

        # Seed 고정: validation마다 동일한 이미지 생성
        setup_seed(42)

        prompts = self.validation_prompts
        world_size = dist.get_world_size()

        # Multi-GPU 프롬프트 분산 (round-robin)
        my_indices = list(range(self.global_rank, len(prompts), world_size))
        max_per_rank = math.ceil(len(prompts) / world_size)

        local_results = []  # list of (prompt_idx, prompt_text, image_bytes)

        for iter_idx in range(max_per_rank):
            is_real = iter_idx < len(my_indices)
            i = my_indices[iter_idx] if is_real else 0  # dummy uses prompt 0
            prompt = prompts[i]
            sample_start_time = time.time()

            input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt, templates)

            con_prompt_token = self.tokenizer(input_prompt)["input_ids"]
            uncon_prompt_token = self.tokenizer(uncon_prompt)["input_ids"]

            img_mask_token = add_break_line(
                [MASK] * self.validation_params["seq_len"],
                self.validation_params["token_grid_height"],
                self.validation_params["token_grid_width"],
                new_number=NEW_LINE,
            )
            img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]

            prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device="cuda").unsqueeze(0)
            uncon_ids = torch.tensor(uncon_prompt_token, device="cuda").unsqueeze(0)

            code_start = len(con_prompt_token) + 2  # +2 for BOA and BOI

            try:
                out_tokens = generate_image(
                    self.model,
                    prompt_ids,
                    seq_len=self.validation_params["seq_len"],
                    newline_every=self.validation_params["newline_every"],
                    timesteps=64,
                    temperature=1.0,
                    cfg_scale=4.0,
                    uncon_ids=uncon_ids,
                    code_start=code_start,
                    refresh_interval=5,
                    warmup_ratio=0.3,
                )

                # Dummy iterations only exist for FSDP synchronization; discard results
                if is_real:
                    img = decode_vq_to_image(
                        out_tokens,
                        save_path=None,
                        vae_ckpt=None,
                        image_height=self.args.gen_image_size,
                        image_width=self.args.gen_image_size,
                        vqvae=self.vqvae,
                    )
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    local_results.append((i, prompt, buf.getvalue()))

                    elapsed = time.time() - sample_start_time
                    print(f"[Rank {self.global_rank}] [Validation] Finished prompt {i+1}/{len(prompts)} (Time: {elapsed:.2f}s)")

            except Exception as e:
                print(f"[Rank {self.global_rank}] [Validation] Error generating image for prompt '{prompt}': {e}")

        # Gather results from all ranks
        all_results = [None] * world_size
        dist.all_gather_object(all_results, local_results)
        gathered = [item for sublist in all_results for item in sublist]

        # Rank 0: log to WandB
        if self.global_rank == 0 and self.args.use_wandb and gathered:
            gathered.sort(key=lambda x: x[0])  # sort by prompt index
            images = []
            for idx, caption, img_bytes in gathered:
                img = Image.open(io.BytesIO(img_bytes))
                images.append(wandb.Image(img, caption=caption))
            wandb.log({"val/generated_images": images})

        self.model.train()
        dist.barrier()
        
    def _dataset_func(self):
        if self.args.task == 'ocr':
            return self._load_ocr_dataset()
        elif self.args.task == 'ocr_synthetic':
            return self._load_synthetic_ocr_dataset()
        elif self.args.task in ['position', 'rel_position']:
            return self._load_position_dataset()
        elif self.args.task == 'celeb':
            return self._load_celeb_dataset()
        else:
            return self._load_counting_pointing_dataset()
        

    def _load_position_dataset(self):
        print("[Solver] Loading Position Dataset...")
        
        if self.args.task == 'position':
            train_ds = load_dataset("heez/quadrant-position-new", split="train", streaming=False)
            self.val_ds_stream = load_dataset("heez/quadrant-position-new", split="validation", streaming=False)
        elif self.args.task == 'rel_position':
            train_ds = load_dataset("heez/relative-position-new", split="train", streaming=False)
            self.val_ds_stream = load_dataset("heez/relative-position-new", split="validation", streaming=False)
        
        self.val_ds = self.val_ds_stream

        # if self.args.count_upper_limit is not None:
        #     train_ds = train_ds.filter(lambda count: count <= self.args.count_upper_limit, input_columns=['count'], num_proc=64)
        #     self.val_ds_stream = self.val_ds_stream.filter(lambda count: count <= self.args.count_upper_limit, input_columns=['count'])
        # if self.args.count_lower_limit is not None:
        #     train_ds = train_ds.filter(lambda count: count >= self.args.count_lower_limit, input_columns=['count'], num_proc=64)
        #     self.val_ds_stream = self.val_ds_stream.filter(lambda count: count >= self.args.count_lower_limit, input_columns=['count'])
        item_processor = self._item_processor_func(tokenizer=self.tokenizer, max_len=self.args.max_seq_len)

        # Descriptions이 존재하는 샘플은 gen 데이터셋, 없는 샘플은 und 데이터셋으로 분리
        train_und_ds = train_ds
        train_gen_ds = train_ds.map(
            lambda answer: {
                'descriptions': answer
            },
            input_columns=['answer'], num_proc=64
        ) # Answer -> Descriptions

        # Understanding validation dataset: train_und_ds에서 100개 샘플 선택
        self.train_und_ds_val = train_und_ds.select(range(min(100, len(train_und_ds))))

        return HFDatasetWrapper(train_gen_ds, train_und_ds, item_processor, default_task=self.args.task, mode=self.args.mode)
        
    def _load_synthetic_ocr_dataset(self):
        print("[Solver] Loading Synthetic OCR Dataset...")
        
        item_processor = self._item_processor_func(tokenizer=self.tokenizer, max_len=self.args.max_seq_len)
        
        raw_dataset = load_dataset("agentlans/high-quality-english-sentences", split="train")

        # Filter 200K
        raw_dataset = raw_dataset.select(list(range(0, 200_000)))
        train_und_ds = raw_dataset.map(
            lambda text: {
                'descriptions': None,
                'answer': text.replace("\n", " ").strip()[:120]
            },
            input_columns=['text'], num_proc=64
        )
        # gen 데이터셋: 'answer' ->  'descriptions'로 매핑하여 생성
        train_gen_ds = train_und_ds.map(
            lambda answer: {
                'descriptions':
                    f"A Mathpix Markdown format with sharp, legible black text. "
                    f"High-resolution typography, top-down view. "
                    f"The text is rendered in natural left-to-right, top-to-bottom reading order. "
                    f"The text reads:\n"
                    f"{answer}"
            },
            input_columns=['answer'], num_proc=64
        )
        
        # Gen Validation Dataset: test split에서 100개 샘플 선택
        raw_dataset_val = load_dataset("agentlans/high-quality-english-sentences", split="test")
        raw_dataset_val = raw_dataset_val.select(list(range(0, min(self.args.validation_samples, len(raw_dataset_val)))))
        self.val_ds = raw_dataset_val.map(
            lambda text: {
                'descriptions': None,
                'answer': text.replace("\n", " ").strip()[:120]
            },
            input_columns=['text'], num_proc=64
        ) # Columns: 'text', 'descriptions', 'answer'
        
        return HFDatasetWrapper(train_gen_ds, train_und_ds, item_processor, default_task='ocr_synthetic', mode=self.args.mode)
        
        

    def _load_ocr_dataset(self):
        print("[Solver] Loading OCR Dataset...")
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

        train_ds = dataset["train"]
        self.val_ds = dataset["validation"]

        item_processor = self._item_processor_func(tokenizer=self.tokenizer, max_len=self.args.max_seq_len)

        # gen 데이터셋: answer를 descriptions로 매핑하여 생성
        train_gen_ds = train_ds.map(
            lambda answer: {'descriptions': f"An image containing the text: {answer}"},
            input_columns=['answer'], num_proc=64
        )
        train_und_ds = train_ds

        # gen 데이터에 answer가 비어있는 경우 필터링
        train_gen_ds = train_gen_ds.filter(
            lambda descriptions: descriptions is not None and descriptions != '',
            num_proc=64, input_columns=['descriptions']
        )
        if len(train_gen_ds) == 0:
            train_gen_ds = None

        return HFDatasetWrapper(train_gen_ds, train_und_ds, item_processor, default_task='ocr', mode=self.args.mode)

    def _load_counting_pointing_dataset(self):
        print("[Solver] Loading Counting/Pointing Datasets...")
        train_ds = load_dataset("heez/pixmo-point-count-gen-und", split="train", streaming=False)

        self.val_ds_stream = load_dataset("heez/pixmo-point-count-gen-und", split="val_und", streaming=False)

        if self.args.count_upper_limit is not None:
            train_ds = train_ds.filter(lambda count: count <= self.args.count_upper_limit, input_columns=['count'], num_proc=64)
            self.val_ds_stream = self.val_ds_stream.filter(lambda count: count <= self.args.count_upper_limit, input_columns=['count'])
        if self.args.count_lower_limit is not None:
            train_ds = train_ds.filter(lambda count: count >= self.args.count_lower_limit, input_columns=['count'], num_proc=64)
            self.val_ds_stream = self.val_ds_stream.filter(lambda count: count >= self.args.count_lower_limit, input_columns=['count'])
        item_processor = self._item_processor_func(tokenizer=self.tokenizer, max_len=self.args.max_seq_len)

        train_gen_ds = train_ds.filter(lambda descriptions: descriptions is not None and descriptions != '', num_proc=64, input_columns=['descriptions'])
        train_und_ds = train_ds.filter(lambda descriptions: descriptions is None or descriptions == '', num_proc=64, input_columns=['descriptions'])

        self.train_und_ds_val = train_und_ds.select(range(min(100, len(train_und_ds))))

        return HFDatasetWrapper(train_gen_ds, train_und_ds, item_processor, default_task=self.args.task, mode=self.args.mode)

    def _load_celeb_dataset(self):
        print("[Solver] Loading Celebrity Recognition Dataset...")
        train_ds = load_dataset("heez/celeb-recognition", split="train", streaming=False)
        self.val_ds = load_dataset("heez/celeb-recognition", split="test", streaming=False)
        self.val_ds_stream = self.val_ds

        item_processor = self._item_processor_func(tokenizer=self.tokenizer, max_len=self.args.max_seq_len)

        train_und_ds = train_ds
        self.train_und_ds_val = train_und_ds.select(range(min(100, len(train_und_ds))))

        return HFDatasetWrapper(None, train_und_ds, item_processor, default_task='celeb', mode=self.args.mode)

    def _make_and_save_starting_point(self, save_path: str) -> None:
        print(f"[Solver] Creating starting point at {save_path}...")
        tokenizer = AutoTokenizer.from_pretrained(self.args.init_from, trust_remote_code=True)
        base_config = AutoConfig.from_pretrained(self.args.init_from, trust_remote_code=True)
        
        # Load model with correct config
        model = LLaDAForMultiModalGeneration(base_config)
        
        # Resize embeddings if needed (usually matches init_from)
        model.resize_token_embeddings(len(tokenizer))
        
        # Ensure output layer matches vocab size
        if model.model.transformer.ff_out.out_features != len(tokenizer):
             model.model.transformer.ff_out = torch.nn.Linear(4096, len(tokenizer), bias=False)

        # Load weights from init_from if available (transfer weights)
        original_model = LLaDAForMultiModalGeneration.from_pretrained(self.args.init_from, torch_dtype=torch.bfloat16, device_map="cpu")
        model.load_state_dict(original_model.state_dict())
        
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("[Solver] Starting point saved.")
        
    @torch.no_grad()
    def validate_position(self, epoch):
        dist.barrier() # Sync before validation
        
        # Rank 준비
        local_rank = dist.get_rank() 
        world_size = dist.get_world_size()
        if self.global_rank == 0:
            task_name = "Relative Positioning" if self.args.task == 'rel_position' else "Quadrant Positioning"
            print(f"\n[Epoch {epoch} | Step {self.global_step}] Running Validation on {task_name}...")
        
        self.model.eval()
        
        # Ensure VQ-VAE is loaded
        if not hasattr(self, 'vqvae'):
            print("[Solver] Loading VQ-VAE for validation...")
             # Reuse logic from train_one_epoch
            if self.args.precision == "bf16":
                dtype = torch.bfloat16
            elif self.args.precision == "fp16":
                raise ValueError("FP16 precision is not supported for VQ-VAE.")
            else:
                dtype = torch.float32
            self.vqvae = VQModel.from_pretrained(self.args.init_from, subfolder="vqvae", torch_dtype=dtype).to(f"cuda:{self.global_rank}")
            self.vqvae.eval()

        correct = 0
        total = 0
        eval_limit = 100
        
        # Rank별로 local_dataset 리스트 준비
        eval_dataset = iter(self.val_ds_stream)
        local_dataset_list = []
        for i, _item in enumerate(eval_dataset):
            if i >= eval_limit:
                break
            if i % world_size == local_rank:
                local_dataset_list.append(_item)
        
        import gc
        local_predictions = []
        local_references = []
        local_questions = []
        local_images = []

        with torch.no_grad():
            # Only rank 0 shows progress bar to avoid clutter
            disable_tqdm = (self.global_rank != 0)
            progress = tqdm(range(len(local_dataset_list)), desc="Validation", unit="sample", disable=disable_tqdm)

            for idx, item in enumerate(local_dataset_list):
                image = item.get('image')
                question = item.get('question')
                answer_gt = item.get('answer')
                

                # Preprocess Image
                crop_size_list = generate_crop_size_list((self.args.und_image_size // 32) ** 2, 32)
                image_processed = var_center_crop(image, crop_size_list=crop_size_list)
                
                # Encode Image
                input_img_token, (H, W) = encode_img_with_breaks_fixed(image_processed, self.vqvae)
                img_token = [BOI] + add_break_line(input_img_token[1:-1], H, W, new_number=NEW_LINE) + [EOI]

                instruction = "<system>" + UNDERSTANDING_PROMPT_TEMPLATE + "</system>" + "<user>" + question + "</user>"
                input_ids_raw = self.tokenizer(instruction)['input_ids']
                input_token = input_ids_raw[:-1] + img_token + input_ids_raw[-1:]
                code_start = len(input_token) + 1
                STEPS_LENGTH = 128
                GEN_LENGTH = 128
                BLOCK_LENGTH = 128
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

                print("=" * 50)
                print(f"Prediction: \n{pred_text}")
                print(f"Ground Truth: \n{answer_gt}")
                

                local_predictions.append(pred_text)
                local_references.append(answer_gt)
                local_questions.append(question)
                local_images.append(image_processed)

                if idx % 2 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        # Gather results from all GPUs
        all_predictions = [None] * world_size
        all_references = [None] * world_size
        all_questions = [None] * world_size
        all_images = [None] * world_size

        dist.barrier()
        dist.all_gather_object(all_predictions, local_predictions)
        dist.all_gather_object(all_references, local_references)
        dist.all_gather_object(all_questions, local_questions)
        dist.all_gather_object(all_images, local_images)
        
        if self.global_rank == 0:
            predictions = [p for sublist in all_predictions for p in sublist]
            references = [r for sublist in all_references for r in sublist]
            questions = [q for sublist in all_questions for q in sublist]
            images = [img for sublist in all_images for img in sublist]


            print(f"[Position Validation] Gathered {len(predictions)} predictions from {world_size} GPUs")

            if self.args.use_wandb:
                # 준비
                val_table = wandb.Table(columns=["Step", "Image", "Question", "Answer", "GT", "Correctness"])
                acc_list = []
                
                for pred, ref, q, img in zip(predictions, references, questions, images):
                    acc_bool = check_pos_accuracy(pred, ref)
                    acc_list.append(acc_bool)
                    val_table.add_data(self.global_step, wandb.Image(img), q, pred, ref, acc_bool)
                
                # Calculate overall accuracy
                acc_list = np.array(acc_list)
                accuracy = acc_list.mean()
                print(f"[Positioning Validation] Accuracy: {accuracy:.4f} ({acc_list.sum()}/{len(acc_list)})")
                    
                # Log    
                wandb.log({
                    "val/pos_verbose": val_table,
                    "val/pos_acc": accuracy
                })
                
        dist.barrier() # Ensure all ranks are done before moving on
        self.model.train()
                

    @torch.no_grad()
    def validate(self, epoch, format="counting", split="val"):
        dist.barrier() # Sync before validation
        
        local_rank = dist.get_rank() 
        world_size = dist.get_world_size()
        # local_device =  torch.device(f"cuda:{local_rank}")
        
        # Turn off lora
        # self.model.disable_adapter_layers() if self.args.use_lora else None
        
        if self.global_rank == 0:
            print(f"\n[Epoch {epoch} | Step {self.global_step}] Running Validation on CountBenchQA ({split})...")
        
        self.model.eval()
        
        # Ensure VQ-VAE is loaded
        if not hasattr(self, 'vqvae'):
            print("[Solver] Loading VQ-VAE for validation...")
             # Reuse logic from train_one_epoch
            if self.args.precision == "bf16":
                dtype = torch.bfloat16
            elif self.args.precision == "fp16":
                raise ValueError("FP16 precision is not supported for VQ-VAE.")
            else:
                dtype = torch.float32
            self.vqvae = VQModel.from_pretrained(self.args.init_from, subfolder="vqvae", torch_dtype=dtype).to(f"cuda:{self.global_rank}")
            self.vqvae.eval()

        correct = 0
        total = 0
        eval_limit = 100
        
        details_buffer = []
        gt_counts = []
        pred_counts = []
        
        # Rank별로 local_dataset 리스트 준비
        if split == "val":
            # All ranks iterate, but effectively they process the same data if not sharded. 
            # For FSDP generation, they MUST run the same inputs to keep internal states synced.
            eval_dataset = iter(self.val_ds_stream)
            local_dataset_list = []
            for i, _item in enumerate(eval_dataset):
                if i >= eval_limit:
                    break
                if i % world_size == local_rank:
                    # print(f"[Rank {local_rank}] Adding sample {i} to local eval set.")
                    local_dataset_list.append(_item)
        elif split == "train":
            # use self.train_und_ds_val for quick validation on training data
            eval_dataset = self.train_und_ds_val
            local_dataset_list = []
            for i, _item in enumerate(eval_dataset):
                if i % world_size == local_rank:
                    local_dataset_list.append(_item)
        else:
            raise ValueError(f"Unknown split: {split}")
        
        
        # WandB Table - create new table each validation call
        val_table = None
        if self.args.use_wandb and self.global_rank == 0:
            val_table = wandb.Table(columns=["Step", "Accuracy", "Details"])

        with torch.no_grad():
            count = 0
            # Only rank 0 shows progress bar to avoid clutter
            disable_tqdm = (self.global_rank != 0)
            progress = tqdm(range(len(local_dataset_list)), desc=f"Validation ({split})", unit="sample", disable=disable_tqdm)
            
            for item in local_dataset_list:
                # if count >= eval_limit:
                #     break
                
                image = item.get('image')
                
                from io import BytesIO
                from PIL import Image
                # if image is dict with 'bytes' key, convert to PIL Image
                if isinstance(image, dict) and 'bytes' in image:
                    image = Image.open(BytesIO(image['bytes'])).convert("RGB")
                
                if split == "train" :
                    if self.args.task == "counting":
                        question = item.get('question_count', '')
                elif split == "val":
                    question = item.get('question', '')
                question = question.replace('**<number>** of', '**<number>**') # Deprecated old format fix
                        
                gt_count = item.get('count') # Pointing task might not have count, handle gracefully if needed or assume mixed dataset
                
                if format == 'pointing':
                    label = item.get('label', '<object>')
                    question_point_example = f'''<points x1="<coordinate of  x1>" y1="<coordinate of  y1>" x2="<coordinate of  x2>" y2="<coordinate of  y2>" ... x_n="<coordinate of  x_n>" y_n="<coordinate of  y_n>" alt="{label}">{label}</points>.'''.strip()
                    question_count_example = f"There are **<number>** {label} in the image.".strip()
                    
                    question = (
                        f"Locate all {label}. How many {label} are there in the image?. "
                        f"Response Example : {question_point_example}. {question_count_example}"
                    )

                if image is None: continue
                if gt_count is None: continue # Skip if no GT count available

                # Preprocess Image
                crop_size_list = generate_crop_size_list((self.args.und_image_size // 32) ** 2, 32)
                image_processed = var_center_crop(image, crop_size_list=crop_size_list)
                
                # Encode Image
                input_img_token, (H, W) = encode_img_with_breaks_fixed(image_processed, self.vqvae)
                img_token = add_break_line(input_img_token[1:-1], H, W, new_number=NEW_LINE)
                img_token = [BOI] + img_token + [EOI]
                

                instruction = "<system>You are a multimodal model that can process both text and images. Answer the following question based on the provided images.</system>" + \
                              "<user>" + question + "</user>"
                
                input_ids_raw = self.tokenizer(instruction)['input_ids']
                
                # Insert Image Token
                input_token = input_ids_raw[:-1] + img_token + input_ids_raw[-1:]
                
                # Prepare Generation Input
                code_start = len(input_token) + 1
                STEPS_LENGTH= 20 if format == "counting" else 128
                GEN_LENGTH= 20 if format == "counting" else 512
                BLOCK_LENGTH= 20 if format == "counting" else 128
                input_token = input_token + [BOA] + GEN_LENGTH * [MASK] # gen_length=10 for short answer
                input_ids = torch.tensor(input_token, device=f"cuda:{self.global_rank}").unsqueeze(0)
                
                # Generate (All ranks must call this!)
                out_new = generate_text_understanding(
                    self.model, input_ids,
                    steps=STEPS_LENGTH, # reduced steps for validation speed
                    gen_length=GEN_LENGTH,
                    block_length=BLOCK_LENGTH, # one block
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking='low_confidence',
                    code_start=code_start
                )
                
                answer = self.tokenizer.batch_decode(out_new[:, code_start:], skip_special_tokens=True)[0]
                pred_count = extract_number_fixed(answer)
                
                gt_count_val = int(gt_count)
                pred_count_val = int(pred_count)
                is_correct = bool(pred_count_val == gt_count_val)
                
                gt_counts.append(gt_count_val)
                pred_counts.append(pred_count_val)

                if is_correct:
                    correct += 1
                total += 1
                
                res_str = f"\n[{count + 1}] \nQuestion: {question} \nGT: {gt_count_val} \nPred: {pred_count_val} \nCorrect: {is_correct} \nAns: {answer}"
                print(res_str)
                print("-" * 50)
                details_buffer.append(res_str)
                
                count += 1
                progress.update(1)
                
                # Prevent VRAM accumulation
                if count % 2 == 0:
                    import gc; gc.collect()
                    torch.cuda.empty_cache()
            
            progress.close()
            
            # Gather 1) pred_counts and 2) gt_counts 3) details_buffer from all ranks to rank 0 for confusion matrix
            all_pred_counts = [None for _ in range(world_size)] 
            all_gt_counts = [None for _ in range(world_size)] 
            all_details_buffer = [None for _ in range(world_size)] 

            dist.barrier()
            dist.all_gather_object(all_pred_counts, pred_counts)
            dist.all_gather_object(all_gt_counts, gt_counts)
            dist.all_gather_object(all_details_buffer, details_buffer)
            
            # if self.global_rank == 0:
            #     print(f"all_pred_counts: {all_pred_counts}")
            #     print(f"all_gt_counts: {all_gt_counts}")
            #     print(f"all_details_buffer: {all_details_buffer}")
            
            pred_counts = [count for sublist in all_pred_counts for count in sublist] 
            gt_counts = [count for sublist in all_gt_counts for count in sublist]
            details_buffer = [detail for sublist in all_details_buffer for detail in sublist]
            
            # if self.global_rank == 0:
            #     print(f"pred_counts gathered:\n {pred_counts}")
            #     print(f"gt_counts gathered:\n {gt_counts}")
            #     print(f"details_buffer gathered:\n {details_buffer}")
                    
            if self.global_rank == 0:
                # Calculate Accuracy
                total = len(gt_counts)
                correct = sum(1 for g, p in zip(gt_counts, pred_counts) if g == p and p >= 0)
                
                accuracy = correct / total if total > 0 else 0
                
                # Calculate Mean Average Deviation
                deviations = [abs(g - p) for g, p in zip(gt_counts, pred_counts)]
                mean_avg_deviation = sum(deviations) / len(deviations) if deviations else 0.0
                
                print(f"[Split {split}] Validation Accuracy ({format}): {accuracy:.4f} ({correct}/{total})")
                print(f"[Split {split}] Mean Average Deviation ({format}): {mean_avg_deviation:.4f}")
                
                # Generate Confusion Matrix
                try:
                    # Filter valid predictions (non-negative) for matrix
                    valid_indices = [i for i, p in enumerate(pred_counts) if p >= 0]
                    valid_gt = [gt_counts[i] for i in valid_indices]
                    valid_pred = [pred_counts[i] for i in valid_indices]
                    
                    # Determine labels range
                    labels = sorted(list(set(valid_gt + valid_pred)))
                    cm = confusion_matrix(valid_gt, valid_pred, labels=labels)
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=labels, yticklabels=labels)
                    plt.xlabel('Predicted Count')
                    plt.ylabel('Target Count')
                    plt.title(f'Confusion Matrix (Acc: {accuracy:.4f}, MAD: {mean_avg_deviation:.4f})')
                    
                    # Save to buffer
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    cm_image = Image.open(buf)
                    
                    plt.close()
                    
                    # Fixed range 0-20 version 
                    labels_fixed_range = list(range(0, 21)) # Fixed range for counting 0-20
                    cm_fixed = confusion_matrix(valid_gt, valid_pred, labels=labels_fixed_range)
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(cm_fixed, annot=True, fmt='d', cmap='viridis', xticklabels=labels_fixed_range, yticklabels=labels_fixed_range)
                    plt.xlabel('Predicted Count')
                    plt.ylabel('Target Count')  
                    plt.title(f'Confusion Matrix 0-20 (Acc: {accuracy:.4f}, MAD: {mean_avg_deviation:.4f})')
                    # Save to buffer
                    buf_fixed = io.BytesIO()
                    plt.savefig(buf_fixed, format='png')
                    buf_fixed.seek(0)
                    cm_image_fixed = Image.open(buf_fixed)
                    plt.close()
                    
                except Exception as e:
                    print(f"Error generating confusion matrix: {e}")
                    cm_image = None
                    cm_image_fixed = None
                    
                if self.args.use_wandb:
                    all_details = "\n".join(details_buffer)
                    if val_table is not None:
                        val_table.add_data(self.global_step, accuracy, all_details)

                    log_data = {
                        f"{split}/accuracy" if format == "counting" else f"{split}/accuracy_pointing": accuracy,
                        f"{split}/mean_avg_deviation" if format == "counting" else f"{split}/mean_avg_deviation_pointing": mean_avg_deviation,
                        f"{split}/epoch": epoch,
                        "global_step": self.global_step,
                        f"{split}/predictions": val_table
                    }
                    
                    if cm_image is not None:
                        log_data[f"{split}/confusion_matrix" if format == "counting" else f"{split}/confusion_matrix_pointing"] = wandb.Image(cm_image, caption=f"Confusion Matrix Epoch {epoch}")
                    if cm_image_fixed is not None:
                        log_data[f"{split}/confusion_matrix_0-20" if format == "counting" else f"{split}/confusion_matrix_0-20_pointing"] = wandb.Image(cm_image_fixed, caption=f"Confusion Matrix 0-20 Epoch {epoch}")
                    
                    wandb.log(log_data)
                
                # Cleanup
                del eval_dataset
                torch.cuda.empty_cache()
                self.model.train()

    @torch.no_grad()
    def validate_ocr(self, epoch):
        """OCR validation with multi-GPU support and text metrics."""
        import gc

        dist.barrier()
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()

        self.model.eval()

        if not hasattr(self, 'vqvae'):
            dtype = torch.bfloat16 if self.args.precision == "bf16" else torch.float32
            self.vqvae = VQModel.from_pretrained(self.args.init_from, subfolder="vqvae", torch_dtype=dtype).to(f"cuda:{self.global_rank}")
            self.vqvae.eval()

        loaded_metrics = {"wer": evaluate.load("wer"), "cer": evaluate.load("cer"), "meteor": evaluate.load("meteor")}

        subset_indices = list(range(min(len(self.val_ds), self.args.validation_samples)))
        local_dataset_list = []
        for idx, i in enumerate(subset_indices):
            if idx % world_size == local_rank:
                local_dataset_list.append(self.val_ds[i])

        if self.global_rank == 0:
            print(f"[OCR Validation] Total samples: {len(subset_indices)}, Per GPU: ~{len(subset_indices)//world_size}")

        local_predictions = []
        local_references = []
        local_questions = []
        local_images = []

        for idx, item in tqdm(enumerate(local_dataset_list), total=len(local_dataset_list), desc=f"[Rank {local_rank}] OCR Validation", disable=(local_rank != 0)):
            question = item.get('question', "Extract all text from the image.")
            answer_gt = item.get('answer', "")

            if self.args.task == 'ocr_synthetic':
                answer_input_ids = self.tokenizer(answer_gt, add_special_tokens=False)['input_ids']
                answer_template = deepcopy(answer_input_ids)  # copy answer_input_ids
                answer_template[1:] = [MASK] * (len(answer_template) - 1)  # Mask all but first token
                image_processed = generate_ocr_image('# ' + answer_gt, template="clean_light", width=self.args.und_image_size, height=self.args.und_image_size, quality=100)
            else:
                image = item['image']
                from io import BytesIO
                from PIL import Image
                if isinstance(image, dict) and 'bytes' in image:
                    image = Image.open(BytesIO(image['bytes'])).convert("RGB")
                crop_size_list = generate_crop_size_list((self.args.und_image_size // 32) ** 2, 32)
                image_processed = var_edge_pad(image, crop_size_list=crop_size_list, pad_mode='edge')
            input_img_token, (H, W) = encode_img_with_breaks_fixed(image_processed, self.vqvae)
            img_token = [BOI] + add_break_line(input_img_token[1:-1], H, W, new_number=NEW_LINE) + [EOI]

            instruction = "<system>" + UNDERSTANDING_PROMPT_TEMPLATE + "</system>" + "<user>" + question + "</user>"
            input_ids_raw = self.tokenizer(instruction)['input_ids']
            input_token = input_ids_raw[:-1] + img_token + input_ids_raw[-1:]
            code_start = len(input_token) + 1
            if self.args.task == 'ocr_synthetic':
                input_token = input_token + [BOA] + answer_template + [EOA]
                STEPS_LENGTH = len(answer_template) 
                GEN_LENGTH = len(answer_template)
                BLOCK_LENGTH = len(answer_template)
            else :
                STEPS_LENGTH = 128
                GEN_LENGTH = 512 if self.args.task == 'ocr' else 128
                BLOCK_LENGTH = 128
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

            local_predictions.append(pred_text)
            local_references.append(answer_gt)
            local_questions.append(question)
            local_images.append(image_processed)

            if idx % 2 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Gather results from all GPUs
        all_predictions = [None] * world_size
        all_references = [None] * world_size
        all_questions = [None] * world_size
        all_images = [None] * world_size

        dist.barrier()
        dist.all_gather_object(all_predictions, local_predictions)
        dist.all_gather_object(all_references, local_references)
        dist.all_gather_object(all_questions, local_questions)
        dist.all_gather_object(all_images, local_images)

        if self.global_rank == 0:
            predictions = [p for sublist in all_predictions for p in sublist]
            references = [r for sublist in all_references for r in sublist]
            questions = [q for sublist in all_questions for q in sublist]
            images = [img for sublist in all_images for img in sublist]

            print(f"[OCR Validation] Gathered {len(predictions)} predictions from {world_size} GPUs")

            avg_metrics = calculate_metrics(predictions, references, loaded_metrics=loaded_metrics)
            print(f"[OCR Validation] Step {self.global_step} Metrics: WER={avg_metrics['wer']:.4f}, CER={avg_metrics['cer']:.4f}, BLEU={avg_metrics['bleu']:.4f}")

            if self.args.use_wandb:
                val_table = wandb.Table(columns=["Step", "Image", "Question", "GT", "Prediction", "WER", "CER"])

                rows_added = 0
                for i in range(min(len(predictions), 10)):
                    try:
                        ind_metrics = calculate_metrics([predictions[i]], [references[i]], loaded_metrics=loaded_metrics)
                        val_table.add_data(
                            self.global_step,
                            wandb.Image(images[i]),
                            questions[i][:50] + "..." if len(questions[i]) > 50 else questions[i],
                            references[i][:100] + "..." if len(references[i]) > 100 else references[i],
                            predictions[i][:100] + "..." if len(predictions[i]) > 100 else predictions[i],
                            ind_metrics["wer"],
                            ind_metrics["cer"]
                        )
                        rows_added += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to add row {i} to table: {e}")

                wandb.log({
                    "val/wer": avg_metrics["wer"],
                    "val/cer": avg_metrics["cer"],
                    "val/bleu": avg_metrics["bleu"],
                    "val/meteor": avg_metrics["meteor"],
                    "val/edit_distance": avg_metrics["edit_distance"],
                    "val/precision": avg_metrics["precision"],
                    "val/recall": avg_metrics["recall"],
                    "val/f1": avg_metrics["f1"],
                    "val/samples": val_table,
                    "global_step": self.global_step
                })

        dist.barrier()
        self.model.train()

    @torch.no_grad()
    def validate_celeb(self, epoch):
        """Celebrity recognition validation with multi-GPU support."""
        import gc

        dist.barrier()
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()

        if self.global_rank == 0:
            print(f"\n[Epoch {epoch} | Step {self.global_step}] Running Celebrity Recognition Validation...")

        self.model.eval()

        if not hasattr(self, 'vqvae'):
            dtype = torch.bfloat16 if self.args.precision == "bf16" else torch.float32
            self.vqvae = VQModel.from_pretrained(self.args.init_from, subfolder="vqvae", torch_dtype=dtype).to(f"cuda:{self.global_rank}")
            self.vqvae.eval()

        correct = 0
        total = 0
        eval_limit = min(len(self.val_ds), self.args.validation_samples)

        subset_indices = list(range(eval_limit))
        local_dataset_list = []
        for idx, i in enumerate(subset_indices):
            if idx % world_size == local_rank:
                try:
                    local_dataset_list.append(self.val_ds[i])
                except Exception as e:
                    print(f"[Celeb Val] Skipping corrupted sample {i}: {e}")
                    continue

        if self.global_rank == 0:
            print(f"[Celeb Validation] Total samples: {len(subset_indices)}, Per GPU: ~{len(subset_indices)//world_size}")

        local_predictions = []
        local_references = []
        local_questions = []
        local_images = []

        with torch.no_grad():
            disable_tqdm = (self.global_rank != 0)
            progress = tqdm(range(len(local_dataset_list)), desc=f"[Rank {local_rank}] Celeb Validation", unit="sample", disable=disable_tqdm)

            for idx, item in enumerate(local_dataset_list):
                try:
                    image = item.get('image')
                    question = item.get('question', '')
                    answer_gt = item.get('answer', '')

                    crop_size_list = generate_crop_size_list((self.args.und_image_size // 32) ** 2, 32)
                    image_processed = var_center_crop(image, crop_size_list=crop_size_list)

                    input_img_token, (H, W) = encode_img_with_breaks_fixed(image_processed, self.vqvae)
                    img_token = [BOI] + add_break_line(input_img_token[1:-1], H, W, new_number=NEW_LINE) + [EOI]

                    instruction = "<system>" + UNDERSTANDING_PROMPT_TEMPLATE + "</system>" + "<user>" + question + "</user>"
                    input_ids_raw = self.tokenizer(instruction)['input_ids']
                    input_token = input_ids_raw[:-1] + img_token + input_ids_raw[-1:]
                    code_start = len(input_token) + 1
                    STEPS_LENGTH = 128
                    GEN_LENGTH = 128
                    BLOCK_LENGTH = 128
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

                    print("=" * 50)
                    print(f"Prediction: \n{pred_text}")
                    print(f"Ground Truth: \n{answer_gt}")

                    local_predictions.append(pred_text)
                    local_references.append(answer_gt)
                    local_questions.append(question)
                    local_images.append(image_processed)

                    if idx % 2 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[Celeb Val] Error processing sample {idx}: {e}")
                    progress.update(1)
                    continue

                progress.update(1)
            progress.close()

        all_predictions = [None] * world_size
        all_references = [None] * world_size
        all_questions = [None] * world_size

        dist.barrier()
        dist.all_gather_object(all_predictions, local_predictions)
        dist.all_gather_object(all_references, local_references)
        dist.all_gather_object(all_questions, local_questions)

        if self.global_rank == 0:
            predictions = [p for sublist in all_predictions for p in sublist]
            references = [r for sublist in all_references for r in sublist]
            questions = [q for sublist in all_questions for q in sublist]

            correct = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
            total = len(predictions)
            accuracy = correct / total if total > 0 else 0.0

            print(f"[Celeb Validation] Step {self.global_step} Accuracy: {accuracy:.4f} ({correct}/{total})")

            if self.args.use_wandb:
                val_table = wandb.Table(columns=["Step", "Question", "GT", "Prediction", "Correct"])
                for i in range(min(len(predictions), 20)):
                    is_correct = predictions[i].strip().lower() == references[i].strip().lower()
                    val_table.add_data(
                        self.global_step,
                        questions[i][:100],
                        references[i],
                        predictions[i],
                        is_correct,
                    )

                wandb.log({
                    "val/celeb_accuracy": accuracy,
                    "val/celeb_samples": val_table,
                    "global_step": self.global_step,
                })

        dist.barrier()
        self.model.train()

    def run(self):
        # Check for NaNs in parameters
        print("[Solver] Checking model parameters for NaNs...")
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"[Solver] FATAL: Parameter {name} contains NaNs!")
                sys.exit(1)
            if torch.isinf(param).any():
                print(f"[Solver] FATAL: Parameter {name} contains Infs!")
                sys.exit(1)
        print("[Solver] Model parameters are clean.")
        
        # Ensure VQ-VAE is loaded for validation
        if not hasattr(self, 'vqvae'):
             if self.args.precision == "bf16":
                dtype = torch.bfloat16
             elif self.args.precision == "fp16":
                raise ValueError("FP16 precision is not supported for VQ-VAE.")
             else:
                dtype = torch.float32
             self.vqvae = VQModel.from_pretrained(self.args.init_from, subfolder="vqvae", torch_dtype=dtype).to(f"cuda:{self.global_rank}")
             print(f"[Solver] Loaded VQ-VAE for validation for rank {self.global_rank}.")
             self.vqvae.eval()

        # Initialize global step (estimate)
        steps_per_epoch = len(self.dataloader_train) // self.args.accum_iter
        self.global_step = self.start_epoch * steps_per_epoch
        if self.start_iter > 0:
            self.global_step += self.start_iter // self.args.accum_iter
            
        print(f"[Solver] Starting from Global Step: {self.global_step}")

        # Initial Validation
        run_und = self.args.eval_everything or self.args.mode in ['und', 'both']
        run_gen = self.args.eval_everything or self.args.mode in ['gen', 'both']

        if run_und and self.args.wandb_run_id is None:
            if self.args.task in ['ocr', 'ocr_synthetic']: # OCR
                self.validate_ocr(self.start_epoch)
            elif self.args.task in ['position', 'rel_position']:
                self.validate_position(self.start_epoch)
            elif self.args.task == 'celeb':
                self.validate_celeb(self.start_epoch)
            else: # Counting, Pointing
                self.validate(self.start_epoch, format="counting", split="train")
                self.validate(self.start_epoch, format="counting", split="val")
                if self.args.validation_as_pointing_format:
                    self.validate(self.start_epoch, format="pointing", split="train")
                    self.validate(self.start_epoch, format="pointing", split="val")

        if run_gen and self.args.wandb_run_id is None:
            if self.global_rank == 0:
                print("[Solver] Logging validation images on wandb...")
            self.log_validation_images(self.global_step)

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
        
        if self.global_rank == 0 and self.args.use_wandb:
            wandb.finish()

    def save_checkpoint(self, epoch, iteration=None, global_step=None):
        print(f"[Solver] Saving checkpoint at global step {self.global_step}")
        
        # Check if FSDP or regular model
        is_fsdp = isinstance(self.model, FSDP)
        
        if is_fsdp and self.args.use_lora:
            # FSDP + LoRA: gather full state dict, then save only LoRA adapter weights
            save_name = f"epoch{epoch}"
            if iteration is not None:
                save_name += f"-iter{iteration}"
            if global_step is not None:
                save_name += f"-step{global_step}"
            save_dir = os.path.join(self.args.output_dir, save_name)
            os.makedirs(save_dir, exist_ok=True)

            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            with FSDP.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                full_state_dict = self.model.state_dict()
                if self.global_rank == 0:
                    # Filter to LoRA keys only
                    lora_state_dict = {
                        k: v for k, v in full_state_dict.items()
                        if "lora_" in k or "modules_to_save" in k
                    }
                    torch.save(lora_state_dict, os.path.join(save_dir, "adapter_model.bin"))
                    # Save adapter config if available
                    peft_model = self.model.module if hasattr(self.model, 'module') else self.model
                    if hasattr(peft_model, 'peft_config'):
                        for adapter_name, peft_cfg in peft_model.peft_config.items():
                            peft_cfg.save_pretrained(save_dir)
                            break
                    self.tokenizer.save_pretrained(save_dir)
                    with open(os.path.join(save_dir, "args.json"), "w") as f:
                        json.dump(vars(self.args), f, indent=2)
                    print(f"[Solver] Saved LoRA adapter (FSDP gathered) to {save_dir}")

            util.ckpt.remove_early_ckpts(self.args.output_dir, max_keep=self.args.ckpt_max_keep)

        elif is_fsdp:
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
                
        
        dist.barrier() # Sync

    def train_one_epoch(self, epoch: int, start_iter: int, log_writer=None, metric_logger=None):
        if not hasattr(self, 'vqvae'):
            if self.global_rank == 0:
                print(f"[Solver] Loading VQ-VAE to {torch.cuda.current_device()}...")
            
            if self.args.precision == "bf16":
                dtype = torch.bfloat16
            elif self.args.precision == "fp16":
                raise ValueError("FP16 precision is not supported for VQ-VAE as it causes NaNs. Please use BF16 or FP32.")
            else:
                dtype = torch.float32

            self.vqvae = VQModel.from_pretrained(self.args.init_from, subfolder="vqvae", torch_dtype=dtype).to("cuda")
            self.vqvae.eval()

        self.model.train(True)
        if metric_logger is None:
            metric_logger = misc.MetricLogger(delimiter="  ")
            metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

        header = "Epoch: [{}]".format(epoch)
        print_freq = 10
        accum_iter = self.args.accum_iter

        self.optimizer.zero_grad()
        
        accumulated_loss = 0.0 # Initialize loss accumulator

        # Pre-create Image Processor for Generation tasks to avoid redundant overhead
        from diffusers.image_processor import VaeImageProcessor
        vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False)

        # Import prompt utils for generation
        from utils.prompt_utils import create_prompt_templates, generate_text_to_image_prompt
        templates = create_prompt_templates()

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
            # Check task type by number of columns in batch_data
            is_generation = (len(batch_data) == 2)

            input_ids_list = []
            labels_list = []

            if is_generation:
                images_tuple, captions_tuple = batch_data
            else:
                images_tuple, questions_tuple, answers_tuple = batch_data

            for i in range(len(images_tuple)):
                img = images_tuple[i]
                
                if is_generation:
                    caption = captions_tuple[i]
                    
                    with torch.no_grad():
                        x = image_processor.preprocess(img).to(device=self.vqvae.device, dtype=self.vqvae.dtype)
                        latents = self.vqvae.encode(x).latents
                        B, C, H, W = latents.shape
                        quantized = self.vqvae.quantize(latents)[2][2] + IMAGE_TOKEN_OFFSET # Offset
                        image_tokens_raw = quantized.reshape(B, H, W).flatten().tolist()
                    
                    masked_image_tokens, image_labels = mask_codes(image_tokens_raw)

                    masked_image_tokens = add_break_line(masked_image_tokens, H, W, NEW_LINE)
                    image_labels = add_break_line(image_labels, H, W, -100)

                    input_prompt, uncond_prompt = generate_text_to_image_prompt(caption, templates)
                    if np.random.rand() < 0.1:
                        # 10% uncond
                        input_prompt = uncond_prompt
                    
                    instruction_token = self.tokenizer(input_prompt, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()

                    # 이미지 토큰 보장: instruction을 줄여서 max_seq_len에 이미지가 온전히 들어가도록
                    image_fixed_len = len(masked_image_tokens) + 4  # BOA, BOI, EOI, EOA
                    max_instruction_len = self.args.max_seq_len - image_fixed_len
                    if len(instruction_token) > max_instruction_len:
                        # </user> 토큰 보존: 끝에서 분리 → 앞부분만 truncate → 재결합
                        end_user_tokens = self.tokenizer("</user>", add_special_tokens=False).input_ids
                        n_end = len(end_user_tokens)
                        instruction_token = instruction_token[:max_instruction_len - n_end] + instruction_token[-n_end:]

                    instruction_label = [-100] * len(instruction_token)

                    final_input = instruction_token + [BOA] + [BOI] + masked_image_tokens + [EOI] + [EOA]                
                    final_label = instruction_label + [-100] + [-100] + image_labels + [-100] + [-100]

                else:
                    question = questions_tuple[i]
                    answer = answers_tuple[i]

                    with torch.no_grad():
                        image_tokens = encode_img_with_breaks(img, self.vqvae)

                    if self.args.task in ['ocr', 'ocr_synthetic']:
                        instruction = "<system>" + UNDERSTANDING_PROMPT_TEMPLATE + "</system>" + \
                                      "<user>" + question + "</user>"
                    else:
                        instruction = "<system>You are a multimodal model that can process both text and images. Answer the following question based on the provided images.</system>" + \
                                      "<user>" + question + "</user>"
                    instruction_token = self.tokenizer(instruction, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()

                    instruction_token = instruction_token[:-1] + image_tokens + instruction_token[-1:]
                    instruction_label = [-100] * len(instruction_token)

                    answer_text = answer + "</answer>"
                    answer_token = self.tokenizer(answer_text, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()

                    # Unified MIN_ANSWER_LENGTH truncation for all understanding tasks
                    MIN_ANSWER_LENGTH = 50
                    BOA_TOKEN = 1
                    valid_answer_token_length = self.args.max_seq_len - len(instruction_token) - BOA_TOKEN
                    if valid_answer_token_length < MIN_ANSWER_LENGTH:
                        max_instruction_length = self.args.max_seq_len - MIN_ANSWER_LENGTH - BOA_TOKEN
                        instruction_token = instruction_token[:max_instruction_length]
                        instruction_label = instruction_label[:max_instruction_length]
                        valid_answer_token_length = MIN_ANSWER_LENGTH
                    if len(answer_token) > valid_answer_token_length:
                        answer_token = answer_token[:valid_answer_token_length]

                    answer_token, answer_label = mask_codes(answer_token)

                    final_input = instruction_token + [BOA] + answer_token
                    final_label = instruction_label + [-100] + answer_label

                # task_type = "Gen" if is_generation else "Und"
                # if self.global_rank == 0 :
                #     print(f"[{task_type}]: instruction len: {len(instruction_token)}, answer len: {len(answer_token) if not is_generation else 'N/A'}, masked image len: {len(image_tokens_raw) if is_generation else 'N/A'}, final input len: {len(final_input)}")
                

                input_ids_list.append(final_input)
                labels_list.append(final_label)

            examples = input_ids_list
            labels = labels_list
            
            lr_sched.adjust_learning_rate_epoch(
                self.optimizer, data_iter_step / len(self.dataloader_train) + epoch, self.args
            )

            with {
                "bf16": torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda'),
                "fp16": torch.amp.autocast(dtype=torch.float16, device_type='cuda'),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                c_loss = self.model(input_ids=examples, labels=labels)

            loss = c_loss
            loss_value = loss.item()
            accumulated_loss += loss_value # Accumulate loss

            if not math.isfinite(loss_value):
                if self.global_rank == 0:
                    print(f"[Rank {self.global_rank}] Loss is {loss_value}, stopping training")
                    print(f"[Rank {self.global_rank}] Input IDs (first sample, first 50): {examples[0][:50]}")
                    print(f"[Rank {self.global_rank}] Labels (first sample, first 50): {labels[0][:50]}")
                sys.exit(1)

            effective_loss = loss / accum_iter
            effective_loss.backward()

            if (data_iter_step + 1) % accum_iter == 0:
                if isinstance(self.model, FSDP):
                    self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad)
                
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                
                # Increment Global Step
                self.global_step += 1

                # --- WandB Logging per Step ---
                if self.global_rank == 0 and self.args.use_wandb:
                    avg_loss = accumulated_loss / accum_iter
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "train/global_step": self.global_step,
                        "train/epoch": epoch + (data_iter_step / len(self.dataloader_train))
                     })
                    accumulated_loss = 0.0 # Reset accumulator
                
                # --- Step-based Validation ---
                if self.global_step % self.args.validation_interval == 0:
                    run_und = self.args.eval_everything or self.args.mode in ['und', 'both']
                    run_gen = self.args.eval_everything or self.args.mode in ['gen', 'both']

                    if run_und:
                        if self.args.task in ['ocr', 'ocr_synthetic']:
                            self.validate_ocr(epoch)
                        elif self.args.task in ['position', 'rel_position']:
                            self.validate_position(epoch)
                        elif self.args.task == 'celeb':
                            self.validate_celeb(epoch)
                        else:
                            self.validate(epoch, split="train")
                            self.validate(epoch, split="val")
                            if self.args.validation_as_pointing_format:
                                self.validate(epoch, format="pointing", split="train")
                                self.validate(epoch, format="pointing", split="val")

                    if run_gen and self.args.use_wandb:
                        if self.global_rank == 0:
                            print("[Solver] Logging validation images on wandb...")
                        self.log_validation_images(self.global_step)

                # --- Step-based Saving ---
                if self.global_step % self.args.save_iteration_interval == 0:
                     self.save_checkpoint(epoch, iteration=data_iter_step, global_step=self.global_step)

            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        print(f"Averaged stats: {metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    util.misc.random_seed(42)
    solver = Solver(args)
    solver.run()
