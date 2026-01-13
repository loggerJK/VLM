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
from datasets import load_dataset
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
from utils.image_utils import encode_img_with_breaks, generate_crop_size_list, var_center_crop, encode_img_with_breaks_fixed, add_break_line
# Generation Utils
from generators.text_understanding_generator import generate_text_understanding

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

# ==============================================================================
# 2. Helper Functions
# ==============================================================================

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
class ItemProcessorPointing(ItemProcessorBase):
    """
    Preprocesses raw items from Hugging Face Dataset.
    ONLY performs CPU-bound tasks (Image Cropping, Text Extraction).
    VQ-VAE encoding is deferred to the GPU training loop.
    """
    def __init__(self, tokenizer, max_len, image_size=256, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_size = image_size

    def process_item(self, data_item: dict, training_mode=False, task='counting') -> Tuple[Any, str, str]:
        image = data_item.get('image')
        
        if task == 'pointing':
            question = data_item.get('question', data_item.get('text', ''))
            answer = str(data_item.get('answer', data_item.get('label', '')))
        else: # Default or 'counting'
            question = data_item.get('question_count', data_item.get('text', ''))
            answer = str(data_item.get('answer_count', data_item.get('label', '')))

        # Image Preprocessing (Crop) - Changed to 256 for stability
        crop_size_list = generate_crop_size_list((self.image_size // 32) ** 2, 32)
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
    def __init__(self, hf_dataset, item_processor, default_task='counting'):
        self.dataset = hf_dataset
        self.item_processor = item_processor
        self.default_task = default_task
        self.meta_collection = [
            {
                "type": "default",
                "len": len(hf_dataset),
                "ratio": 1.0,
                "path": "hf_dataset",
                "item_len_list": [1024] * len(hf_dataset)
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
        parser.add_argument("--image_size", type=int, default=256, help="Image size for preprocessing (default: 256)")
        parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU Offloading for FSDP (default: False)")
        
        # Validation
        parser.add_argument("--validation_interval", type=int, default=50, help="Validation interval in global steps")

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

        # Task Argument
        parser.add_argument("--task", type=str, default="counting", choices=["counting", "pointing"], help="Task type")
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
        if self.args.use_wandb and self.global_rank == 0:
            wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                name=self.args.wandb_run_name or f"run-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=vars(self.args),
                mode="online"
            )
        self.val_table = None
        self.global_step = 0

    def build_model(self):
        # Full override of build_model to handle LoRA freezing correctly without relying on FinetuneSolverBase's logic
        init_from = self.args.resume_path or self.args.init_from
        
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
        if self.args.use_lora:
            print(f"[Solver] Applying LoRA with rank: {self.args.lora_rank}, alpha: {self.args.lora_alpha}, dropout: {self.args.lora_dropout}")
            from peft import LoraConfig, get_peft_model, TaskType
            
            target_modules = ["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"]
            print(f"[Solver] LoRA Target Modules: {target_modules}")
            
            # Freeze base model parameters
            for param in model.parameters():
                param.requires_grad = False
            
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
            model.print_trainable_parameters()
            
        return model, tokenizer

    def _item_processor_func(self, tokenizer=None, max_len=None) -> ItemProcessorBase:
        # Pass image_size from args
        return ItemProcessorPointing(tokenizer, max_len, image_size=self.args.image_size)

    def _dataset_func(self):
        print("[Solver] Loading Hugging Face Datasets...")
        LOCAL_TRAIN_DIR = os.getenv('LOCAL_TRAIN_DIR', None)
        LOCAL_VAL_DIR = os.getenv('LOCAL_VAL_DIR', None)
        if LOCAL_TRAIN_DIR and os.path.exists(LOCAL_TRAIN_DIR):
            from datasets import load_from_disk
            train_ds = load_from_disk(LOCAL_TRAIN_DIR)
        else:
            print(f"[Solver] Loading training dataset from Hugging Face Hub... : Jiwon-Kang/pixmo-point-count-concat_0-20-qaFixed")
            train_ds = load_dataset("Jiwon-Kang/pixmo-point-count-concat_0-20-qaFixed", split="train")
        
        # Validation Dataset (Streaming) - Stored in self.val_ds_stream
        if LOCAL_VAL_DIR and os.path.exists(LOCAL_VAL_DIR):
            from datasets import load_from_disk
            self.val_ds_stream = load_from_disk(os.path.join(LOCAL_VAL_DIR, "validation"))
        else:
            self.val_ds_stream = load_dataset("Jiwon-Kang/pixmo-count-filtered-imgContained", split="validation", streaming=True)

        item_processor = self._item_processor_func(tokenizer=self.tokenizer, max_len=self.args.max_seq_len)
        return HFDatasetWrapper(train_ds, item_processor, default_task=self.args.task)

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
    def validate(self, epoch):
        dist.barrier() # Sync before validation
        
        # Turn off lora
        # self.model.disable_adapter_layers() if self.args.use_lora else None
        
        if self.global_rank == 0:
            print(f"\n[Epoch {epoch} | Step {self.global_step}] Running Validation on CountBenchQA...")
        
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
            self.vqvae = VQModel.from_pretrained(self.args.init_from, subfolder="vqvae", torch_dtype=dtype).to("cuda")
            self.vqvae.eval()

        correct = 0
        total = 0
        eval_limit = 100 
        
        # All ranks iterate, but effectively they process the same data if not sharded. 
        # For FSDP generation, they MUST run the same inputs to keep internal states synced.
        eval_dataset = iter(self.val_ds_stream)
        details_buffer = []
        gt_counts = []
        pred_counts = []
        
        # WandB Table
        if self.val_table is None and self.args.use_wandb and self.global_rank == 0:
            self.val_table = wandb.Table(columns=["Step", "Accuracy", "Details"])

        with torch.no_grad():
            count = 0
            # Only rank 0 shows progress bar to avoid clutter
            disable_tqdm = (self.global_rank != 0)
            progress = tqdm(range(eval_limit), desc="Validation", unit="sample", disable=disable_tqdm)
            
            for item in eval_dataset:
                if count >= eval_limit:
                    break
                
                image = item.get('image')
                question = item.get('question', '')
                gt_count = item.get('count') # Pointing task might not have count, handle gracefully if needed or assume mixed dataset
                label = item.get('label', '<object>')
                question = question.replace('**<number>** of', '**<number>**')

                if image is None: continue
                if gt_count is None: continue # Skip if no GT count available

                # Preprocess Image
                crop_size_list = generate_crop_size_list((self.args.image_size // 32) ** 2, 32)
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
                GEN_LENGTH= 20
                input_token = input_token + [BOA] + GEN_LENGTH * [MASK] # gen_length=10 for short answer
                input_ids = torch.tensor(input_token, device="cuda").unsqueeze(0)
                
                # Generate (All ranks must call this!)
                out_new = generate_text_understanding(
                    self.model, input_ids,
                    steps=GEN_LENGTH, # reduced steps for validation speed
                    gen_length=GEN_LENGTH,
                    block_length=GEN_LENGTH, # one block
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking='low_confidence',
                    code_start=code_start
                )
                
                # Only Rank 0 processes results for logging
                if self.global_rank == 0:
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
                    
            if self.global_rank == 0:
                accuracy = correct / total if total > 0 else 0
                
                # Calculate Mean Average Deviation
                deviations = [abs(g - p) for g, p in zip(gt_counts, pred_counts)]
                mean_avg_deviation = sum(deviations) / len(deviations) if deviations else 0.0
                
                print(f"Validation Accuracy: {accuracy:.4f} ({correct}/{total})")
                print(f"Mean Average Deviation: {mean_avg_deviation:.4f}")
                
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
    
                if self.args.use_wandb:
                    all_details = "\n".join(details_buffer)
                    if self.val_table is not None:
                        self.val_table.add_data(self.global_step, accuracy, all_details)
                    
                    log_data = {
                        "val/accuracy": accuracy, 
                        "val/mean_avg_deviation": mean_avg_deviation,
                        "val/epoch": epoch,
                        "global_step": self.global_step,
                        "val/predictions": self.val_table
                    }
                    
                    if cm_image is not None:
                        log_data["val/confusion_matrix"] = wandb.Image(cm_image, caption=f"Confusion Matrix Epoch {epoch}")
                    if cm_image_fixed is not None:
                        log_data["val/confusion_matrix_0-20"] = wandb.Image(cm_image_fixed, caption=f"Confusion Matrix 0-20 Epoch {epoch}")
                    
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
             self.vqvae = VQModel.from_pretrained(self.args.init_from, subfolder="vqvae", torch_dtype=dtype).to("cuda")
             self.vqvae.eval()

        # Initialize global step (estimate)
        steps_per_epoch = len(self.dataloader_train) // self.args.accum_iter
        self.global_step = self.start_epoch * steps_per_epoch
        if self.start_iter > 0:
            self.global_step += self.start_iter // self.args.accum_iter
            
        print(f"[Solver] Starting from Global Step: {self.global_step}")

        # Initial Validation (Unconditional)
        # self.save_checkpoint(epoch=self.start_epoch, iteration=0, global_step=self.global_step)
        self.validate(epoch=self.start_epoch)

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
                
                # Handle rotation (remove old checkpoints)
                util.ckpt.remove_early_ckpts(self.args.output_dir, max_keep=self.args.ckpt_max_keep)
        
        dist.barrier() # Sync

    def train_one_epoch(self, epoch: int, start_iter: int, log_writer=None, metric_logger=None):
        if not hasattr(self, 'vqvae'):
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
            # Unpack batch columns (collate_fn=zip gives tuple of tuples)
            # images_tuple: (img1, img2, ...)
            # questions_tuple: (q1, q2, ...)
            # answers_tuple: (a1, a2, ...)
            images_tuple, questions_tuple, answers_tuple = batch_data
            
            input_ids_list = []
            labels_list = []

            # Iterate over batch elements
            for i in range(len(images_tuple)):
                img = images_tuple[i]
                question = questions_tuple[i]
                answer = answers_tuple[i]

                # A. Encode Image (GPU)
                with torch.no_grad():
                    image_tokens = encode_img_with_breaks(img, self.vqvae)
                
                # B. Prepare Text & Tokens
                instruction = "<system>You are a multimodal model that can process both text and images. Answer the following question based on the provided images.</system>" + \
                              "<user>" + question + "</user>"
                instruction_token = self.tokenizer(instruction, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()
                
                # Insert Image Tokens (before EOS)
                instruction_token = instruction_token[:-1] + image_tokens + instruction_token[-1:] # Insert before </user>
                instruction_label = [-100] * len(instruction_token)

                # Answer
                answer_text = answer + "</answer>"
                answer_token = self.tokenizer(answer_text, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()
                
                answer_token, answer_label = mask_codes(answer_token)

                # Combine with BOA
                final_input = instruction_token + [BOA] + answer_token
                final_label = instruction_label + [-100] + answer_label
                
                if len(final_input) > self.args.max_seq_len:
                    final_input = final_input[:self.args.max_seq_len]
                    final_label = final_label[:self.args.max_seq_len]
                # print(f"instruction_token length: {len(instruction_token)}, answer_token length: {len(answer_token)}")
                # print(f"final_input : {len(final_input)}, final_label : {len(final_label)}")

                input_ids_list.append(final_input)
                labels_list.append(final_label)

            # Pass lists directly to model (it handles padding)
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
                print(f"[Rank {self.global_rank}] Loss is {loss_value}, stopping training")
                print(f"[Rank {self.global_rank}] Input IDs (first sample, first 50): {examples[0][:50]}")
                print(f"[Rank {self.global_rank}] Labels (first sample, first 50): {labels[0][:50]}")
                print(f"[Rank {self.global_rank}] Input Min/Max: {min([min(x) for x in examples])}/{max([max(x) for x in examples])}")
                sys.exit(1)

            effective_loss = loss / accum_iter
            effective_loss.backward()

            # Gradient Accumulation Step
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
                     self.validate(epoch)
                     
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