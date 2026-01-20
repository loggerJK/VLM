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
from PIL import Image

# HuggingFace & Diffusers
from datasets import load_dataset, concatenate_datasets
from diffusers import VQModel
from transformers import AutoTokenizer, AutoConfig

# Project Imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import SPECIAL_TOKENS # Import Special Tokens
from model import LLaDAForMultiModalGeneration
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.solvers.finetune import FinetuneSolverBase
from xllmx.data.sampler import FinetuneDistSampler
import xllmx.util as util
import xllmx.util.lr_sched as lr_sched
import xllmx.util.misc as misc
from fairscale.nn.model_parallel import initialize as fs_init

# Image Utils
from utils.image_utils import encode_img_with_breaks, generate_crop_size_list, var_center_crop, encode_img_with_breaks_fixed, add_break_line, decode_vq_to_image, calculate_vq_params
# Generation Utils
from generators.text_understanding_generator import generate_text_understanding
from generators.image_generation_generator import generate_image
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates

from transformers import enable_full_determinism


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
            crop_size_list = generate_crop_size_list((self.gen_image_size // 32) ** 2, 32)
            image = var_center_crop(image, crop_size_list=crop_size_list)
            return (image, caption)
        
        # Understanding Data
        else:
            image = data_item.get('image')
            
            if task == 'pointing':
                question = data_item.get('question', data_item.get('text', ''))
                answer = str(data_item.get('answer', data_item.get('label', '')))
            else: # Default or 'counting'
                question = data_item.get('question_count', data_item.get('text', ''))
                answer = str(data_item.get('answer_count', data_item.get('label', '')))

            # Image Preprocessing (Crop) - Changed to 256 for stability
            crop_size_list = generate_crop_size_list((self.und_image_size // 32) ** 2, 32)
            image = var_center_crop(image, crop_size_list=crop_size_list)

            # Return Tuple to prevent Collate_fn (zip) from mixing dictionary keys
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
        # Gen, Und 데이터셋은 argument로 항상 들어온다고 가정
        self.gen_dataset = hf_gen_dataset
        self.und_dataset = hf_und_dataset
        self.item_processor = item_processor
        self.default_task = default_task # 'counting' or 'pointing' for understanding
        self.mode = mode # 'und', 'gen', or 'both'
        
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
        item = self.dataset[idx]
        task = item.get('task', self.default_task) # Get task from item, fallback to default_task
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
        parser.add_argument("--wo_lm_head", action="store_true", help="Without LM head in LoRA (for memory saving)")
        
        # Task Argument
        parser.add_argument("--task", type=str, default="counting", choices=["counting", "pointing"], help="Task type for understanding")
        parser.add_argument("--mode", type=str, default='und', choices=['und', 'gen', 'both'], help="Mode for text understanding/generation/both")
        
        # Wandb Resume
        parser.add_argument("--wandb_run_id", type=str, default=None, help="WandB run ID for resume")
        
        # Dataset arguments
        parser.add_argument("--count_upper_limit", type=int, default=None, help="Upper limit for count. 20 means count<=20.")
        parser.add_argument("--count_lower_limit", type=int, default=None, help="Lower limit for count. 0 means count>=0.")
        
        return parser
    
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

        # Handle LoRA wrapping policies if needed, but standard FSDP policy often works if layers are standard linear
        # If LoRA is used, FSDP wraps the PeftModel
        
        model = FSDP(
            model,
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
            ) if not self.args.use_lora else None, # Disable custom wrap policy for LoRA for now, let FSDP handle it or use default
            # Note: For LoRA + FSDP, explicit wrapping is often better, but for now we try default or 'none' for debugging.
            # If using 'none', this method returns early.
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
                
        self.val_table = None
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
            # LoRA handles freezing internally in _model_func (get_peft_model), so we just log.
            # Ensure we don't accidentally unfreeze everything.
            from xllmx.util.tensor_type import promote_param_to_fp32
            
            # Explicitly ensure only LoRA params are trainable (double check)
            # and promote trainable params to fp32 if needed (though BF16 training usually keeps them BF16)
            # Actually, standard practice for LoRA is mixed precision. 
            # If args.precision is bf16, we usually keep LoRA in bf16 or fp32.
            
            # print_param_status will be called later to verify.
            pass 
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
            def check_fn(submodule):
                return submodule in checkpointing_list

            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=check_fn,
            )

        self.logger.info(f"Wrapped model: \n{str(model)}")

        # 6. Optimizer
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
        if self.global_rank == 0:
            print(f"[Solver] Setting up validation prompts from heez/pixmo-point-count-gen-und...")
        
        # Try loading validation split
        val_ds = load_dataset('heez/pixmo-point-count-gen-und', split="val_gen")

        # Sample 10 random indices with fixed seed for consistency
        rng = random.Random(42) 
        indices = rng.sample(range(len(val_ds)), min(10, len(val_ds)))
        
        self.validation_prompts = []
        # text_keys = ['descriptions', 'text', 'caption', 'prompt']
        
        for idx in indices:
            item = val_ds[idx]
            # caption = ""
            # for k in text_keys:
            #     if k in item:
            #         val = item[k]
            #         if isinstance(val, list):
            #              caption = val[0]
            #         else:
            #              caption = val
            #         break
            # if not caption:
            #     caption = "Generate an image."
            caption = item.get('descriptions', "Generate an image.")
            if isinstance(caption, list):
                caption = caption[0]
            self.validation_prompts.append(caption)
            
        # print(f"self.validation_prompts: {self.validation_prompts}")
        
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
        # uses collective communication (FSDP).
        
        if not hasattr(self, 'validation_prompts'):
             self._setup_validation_prompts()
             # Distribute prompts to all ranks so they all know what to generate
             # (Simplified: since we used fixed seed in _setup_validation_prompts, all ranks should have same prompts if they all called it)
        
        # Ensure all ranks call _setup_validation_prompts if they don't have it
        # Actually, in _setup_validation_prompts I had 'if self.global_rank != 0: return'. 
        # I should change that so all ranks have the same fixed prompts.
        templates = create_prompt_templates()
        print(f"[Rank {self.global_rank}] [Validation] Generating images at step {step}...")
        self.model.eval()
        
        prompts = self.validation_prompts
        
        images = []
        for i, prompt in enumerate(prompts):
            sample_start_time = time.time()
            # system_prompt = "Generate an image according to the text prompt."       # template
            # instruction = f"<system>{system_prompt}</system><user>{prompt}</user>"
            
            input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt, templates)

            # build initial sequence
            con_prompt_token = self.tokenizer(input_prompt)["input_ids"]
            uncon_prompt_token = self.tokenizer(uncon_prompt)["input_ids"]

            img_mask_token = add_break_line([MASK] * self.validation_params["seq_len"], self.validation_params["token_grid_height"], self.validation_params["token_grid_width"], new_number = NEW_LINE)
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

                # Only rank 0 decodes and prepares WandB image
                if self.global_rank == 0:
                    img = decode_vq_to_image(
                        out_tokens, 
                        save_path=None, 
                        vae_ckpt=None, 
                        image_height=self.args.gen_image_size, 
                        image_width=self.args.gen_image_size, 
                        vqvae=self.vqvae
                    )
                    images.append(wandb.Image(img, caption=prompt))
                    
                    sample_elapsed_time = time.time() - sample_start_time
                    print(f"[Rank 0] [Validation] Finished prompt {i+1}/{len(prompts)} (Time: {sample_elapsed_time:.2f}s)")
                
            except Exception as e:
                print(f"[Rank {self.global_rank}] [Validation] Error generating image for prompt '{prompt}': {e}")
        
        if self.global_rank == 0 and self.args.use_wandb and images:
            wandb.log({"val/generated_images": images}, step=step)
            
        self.model.train()
        dist.barrier() # Sync after validation
        
    def _dataset_func(self):
        print("[Solver] Loading Hugging Face Datasets...")
        # LOCAL_TRAIN_DIR = os.getenv('LOCAL_TRAIN_DIR', None)
        # LOCAL_VAL_DIR = os.getenv('LOCAL_VAL_DIR', None)
        # if LOCAL_TRAIN_DIR and os.path.exists(LOCAL_TRAIN_DIR):
        #     from datasets import load_from_disk
        #     train_ds = load_from_disk(LOCAL_TRAIN_DIR)
        # else:
        #     print(f"[Solver] Loading training dataset from Hugging Face Hub... : Jiwon-Kang/pixmo-point-count-concat_0-20-qaFixed")
        #     train_ds = load_dataset("Jiwon-Kang/pixmo-point-count-concat_0-20-qaFixed", split="train")
        
        train_ds = load_dataset("heez/pixmo-point-count-gen-und", split="train", streaming=False)
            
        
        # Validation Dataset (Map-style) - Stored in self.val_ds
        # if LOCAL_VAL_DIR and os.path.exists(LOCAL_VAL_DIR):
        #     from datasets import load_from_disk
        #     self.val_ds = load_from_disk(os.path.join(LOCAL_VAL_DIR, "validation"))
        #     # Limit to 100 for speed if needed, or keep full
        #     if len(self.val_ds) > 100:
        #         self.val_ds = self.val_ds.select(range(100))
        # else:
        #     # Load only first 100 samples
        #     self.val_ds = load_dataset("Jiwon-Kang/pixmo-count-filtered-imgContained", split="validation[:100]", streaming=False)
        self.val_ds_stream = load_dataset("heez/pixmo-point-count-gen-und", split="val_und", streaming=False)
        
        if args.count_upper_limit is not None:
            train_ds = train_ds.filter(lambda count: count <= args.count_upper_limit, input_columns=['count'], num_proc=64)
            self.val_ds_stream = self.val_ds_stream.filter(lambda count: count <= args.count_upper_limit, input_columns=['count'])
        if args.count_lower_limit is not None:
            train_ds = train_ds.filter(lambda count: count >= args.count_lower_limit, input_columns=['count'], num_proc=64)
            self.val_ds_stream = self.val_ds_stream.filter(lambda count: count >= args.count_lower_limit, input_columns=['count'])
        item_processor = self._item_processor_func(tokenizer=self.tokenizer, max_len=self.args.max_seq_len)
        
        train_gen_ds = train_ds.filter(lambda descriptions: descriptions is not None and descriptions != '', num_proc=64, input_columns=['descriptions'])
        train_und_ds = train_ds.filter(lambda descriptions: descriptions is None or descriptions == '', num_proc=64, input_columns=['descriptions'])
        
        self.train_und_ds_val = train_und_ds.select(range(min(100, len(train_und_ds))))
        
        return HFDatasetWrapper(train_gen_ds, train_und_ds, item_processor, default_task=self.args.task, mode=self.args.mode)

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
        
        
        # WandB Table
        if self.val_table is None and self.args.use_wandb and self.global_rank == 0:
            self.val_table = wandb.Table(columns=["Step", "Accuracy", "Details"])

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
                    if self.val_table is not None:
                        self.val_table.add_data(self.global_step, accuracy, all_details)
                    
                    log_data = {
                        f"{split}/accuracy" if format == "counting" else f"{split}/accuracy_pointing": accuracy, 
                        f"{split}/mean_avg_deviation" if format == "counting" else f"{split}/mean_avg_deviation_pointing": mean_avg_deviation,
                        f"{split}/epoch": epoch,
                        "global_step": self.global_step,
                        f"{split}/predictions": self.val_table
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

        # Initial Validation (Unconditional)
        # self.save_checkpoint(epoch=self.start_epoch, iteration=0, global_step=self.global_step)
        if self.args.mode in ['und', 'both']:
            self.validate(self.start_epoch, format="counting", split="train")
            self.validate(self.start_epoch, format="counting", split="val")
            if self.args.validation_as_pointing_format:
                self.validate(self.start_epoch, format="pointing", split="train")
                self.validate(self.start_epoch, format="pointing", split="val")
            
        if self.args.mode in ['gen', 'both'] and self.args.use_wandb:
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
                    
                    instruction_token = self.tokenizer(input_prompt, truncation=True, max_length=512, padding=False, return_tensors="pt").input_ids[0].tolist()
                    instruction_label = [-100] * len(instruction_token)

                    final_input = instruction_token + [BOA] + [BOI] + masked_image_tokens + [EOI] + [EOA]                
                    final_label = instruction_label + [-100] + [-100] + image_labels + [-100] + [-100]

                else:
                    question = questions_tuple[i]
                    answer = answers_tuple[i]

                    with torch.no_grad():
                        image_tokens = encode_img_with_breaks(img, self.vqvae)
                    
                    instruction = "<system>You are a multimodal model that can process both text and images. Answer the following question based on the provided images.</system>" + \
                                    "<user>" + question + "</user>"
                    instruction_token = self.tokenizer(instruction, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()
                    
                    instruction_token = instruction_token[:-1] + image_tokens + instruction_token[-1:]
                    instruction_label = [-100] * len(instruction_token)

                    answer_text = answer + "</answer>"
                    answer_token = self.tokenizer(answer_text, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()
                    
                    answer_token, answer_label = mask_codes(answer_token)

                    final_input = instruction_token + [BOA] + answer_token
                    final_label = instruction_label + [-100] + answer_label
                
                if len(final_input) > self.args.max_seq_len:
                    final_input = final_input[:self.args.max_seq_len]
                    final_label = final_label[:self.args.max_seq_len]
                    
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
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
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
                    if self.args.mode in ['und', 'both']:
                        self.validate(epoch, split="train")
                        self.validate(epoch, split="val")
                        if self.args.validation_as_pointing_format:
                            self.validate(epoch, format="pointing", split="train")
                            self.validate(epoch, format="pointing", split="val")
                        
                    if self.args.mode in ['gen', 'both'] and self.args.use_wandb:
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