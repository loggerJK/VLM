import pickle
from typing import List, Tuple, Dict, Any
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

# HuggingFace & Diffusers
from datasets import load_dataset
from diffusers import VQModel
from transformers import AutoTokenizer, AutoConfig

# #agent edited: [1] Import PEFT for LoRA
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

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
from generators.image_generation_generator import generate_image

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

# ==============================================================================
# 2. Helper Functions
# ==============================================================================

def mask_codes(codes, sch="cosine", mask = False, editing = False):
    """
    Applies masking to the target tokens for Masked Diffusion Loss.
    Adapted from train.py
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

# ==============================================================================
# 3. ItemProcessor (CPU Stage)
# ==============================================================================
# #agent edited: [3] ItemProcessor for Text-to-Image Generation
class ItemProcessorGeneration(ItemProcessorBase):
    """
    Preprocesses raw items from Hugging Face Dataset for Text-to-Image Generation.
    - Resizes/Crops Image (CPU)
    - Extracts Caption/Prompt
    """
    def __init__(self, tokenizer, max_len, image_size=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_size = image_size

    def process_item(self, data_item: dict, training_mode=False) -> Tuple[Any, str]:
        # #agent edited: [3-1] Dataset column mapping (Adjust based on your dataset)
        # Expected format: {'image': PIL.Image, 'text': str} or {'image': ..., 'caption': ...}
        image = data_item.get('image')
        
        text_keys = [ 'descriptions', 'text', 'caption', 'prompt']
        caption = ""
        for k in text_keys:
            if k in data_item:
                caption = data_item[k]
                break
        
        # Fallback if no caption found
        if not caption:
            caption = "Generate an image."

        # Image Preprocessing (Crop) - Using 512 or 256 depending on config
        # T2I often uses fixed resolution or aspect ratio bucketing. 
        # Here we use var_center_crop logic from existing codebase.
        crop_size_list = generate_crop_size_list((self.image_size // 32) ** 2, 32)
        image = var_center_crop(image, crop_size_list=crop_size_list)

        return (image, caption)

    def predict_item_token_length(self, data_item: dict) -> int:
        return 1024

# ==============================================================================
# 4. Dataset Wrapper
# ==============================================================================
class HFDatasetWrapper(torch.utils.data.Dataset):
    """
    Wraps Hugging Face Dataset to be compatible with xllmx Sampler.
    """
    def __init__(self, hf_dataset, item_processor):
        self.dataset = hf_dataset
        self.item_processor = item_processor
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
        return self.item_processor.process_item(item, training_mode=True)


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
        parser.add_argument("--image_size", type=int, default=512, help="Image size for preprocessing")
        parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU Offloading for FSDP")
        
        # #agent edited: [4] Dataset Arguments
        parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face Dataset Name (e.g. 'lambdalabs/pokemon-blip-captions')")
        parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")

        # WandB Arguments
        parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
        parser.add_argument("--wandb_project", type=str, default="lumina-generation", help="WandB project name")
        parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity name")
        parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
        
        # #agent edited: [5] LoRA Arguments
        parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
        parser.add_argument("--lora_rank", type=int, default=128, help="LoRA Rank")
        parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
        parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA Dropout")
        parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out"], help="Target modules for LoRA")
        parser.add_argument("--validation_interval", type=int, default=10, help="Validation interval in steps")
        return parser
    
    def setup_fsdp_sync(
        self, model: nn.Module, data_parallel: str, precision: str, grad_precision: str = None
    ) -> FSDP:
        print(f"[Solver] setup_fsdp_sync: cpu_offload={self.args.cpu_offload}")
        
        if self.dp_rank == 0:
            param_init_fn = None
        else:
            param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)

        # Set CPU Offload
        cpu_offload = CPUOffload(offload_params=self.args.cpu_offload)

        # #agent edited: [6] FSDP Wrapping Policy adjustment
        # LoRA models have different structure, need careful wrapping.
        # For simplicity, we use the default policy or custom based on class names.
        
        model = FSDP(
            model,
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list() if hasattr(model, 'get_fsdp_wrap_module_list') else False,
            ),
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
            cpu_offload=cpu_offload, 
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
                config=vars(self.args)
            )

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

        
        # Optimizer Setup
        try:
            import bitsandbytes as bnb
            print("[Solver] Using bitsandbytes.optim.AdamW8bit Optimizer...")
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, betas=(0.9, 0.95))
        except ImportError:
            print("[Solver] bitsandbytes not found, falling back to Standard torch.optim.AdamW...")
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, betas=(0.9, 0.95))
        
        return model, tokenizer, optimizer

    def _model_func(self, init_from: str) -> (LLaDAForMultiModalGeneration, None):
        tokenizer = AutoTokenizer.from_pretrained(init_from, trust_remote_code=True)
        
        if self.args.precision == "bf16":
            dtype = torch.bfloat16
        elif self.args.precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
            
        print(f"[Solver] Loading model in {dtype} (precision: {self.args.precision})...")
        model = LLaDAForMultiModalGeneration.from_pretrained(init_from, torch_dtype=dtype, device_map="cpu")
        
        if hasattr(model.config, 'scale_logits'):
            model.config.scale_logits = True
        
        if self.args.checkpointing:
            print("[Solver] Enabling Gradient Checkpointing...")
            model.model.set_activation_checkpointing("whole_layer")

        if self.args.use_lora:
            for param in model.parameters():
                param.requires_grad = False  # Freeze all parameters
                
            print(f"[Solver] Applying LoRA Adapter (rank={self.args.lora_rank}, alpha={self.args.lora_alpha})...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=self.args.lora_rank, 
                lora_alpha=self.args.lora_alpha, 
                lora_dropout=self.args.lora_dropout,
                target_modules=self.args.lora_target_modules 
            )
            model = get_peft_model(model, peft_config)
            
            if self.global_rank == 0:
                model.print_trainable_parameters()
        else:
            print("[Solver] Full Fine-tuning Mode")
            if self.args.checkpointing:
                print("[Solver] Enabling Activation Checkpointing...")
                model.model.set_activation_checkpointing("whole_layer")
            
        return model, tokenizer

    def _item_processor_func(self, tokenizer=None, max_len=None) -> ItemProcessorBase:
        return ItemProcessorGeneration(tokenizer, max_len, image_size=self.args.image_size)

    def _dataset_func(self):
        # #agent edited: [8] Load user-specified HF Dataset
        print(f"[Solver] Loading Hugging Face Dataset: {self.args.dataset_name} ({self.args.dataset_split})...")
        train_ds = load_dataset(self.args.dataset_name, split=self.args.dataset_split)
        
        item_processor = self._item_processor_func(tokenizer=self.tokenizer, max_len=self.args.max_seq_len)
        return HFDatasetWrapper(train_ds, item_processor)

    def _make_and_save_starting_point(self, save_path: str) -> None:
        print(f"[Solver] Creating starting point at {save_path}...")
        tokenizer = AutoTokenizer.from_pretrained(self.args.init_from, trust_remote_code=True)
        base_config = AutoConfig.from_pretrained(self.args.init_from, trust_remote_code=True)
        model = LLaDAForMultiModalGeneration(base_config)
        model.resize_token_embeddings(len(tokenizer))
        
        if model.model.transformer.ff_out.out_features != len(tokenizer):
             model.model.transformer.ff_out = torch.nn.Linear(4096, len(tokenizer), bias=False)

        original_model = LLaDAForMultiModalGeneration.from_pretrained(self.args.init_from, torch_dtype=torch.bfloat16, device_map="cpu")
        model.load_state_dict(original_model.state_dict())
        
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("[Solver] Starting point saved.")

    def _setup_validation_prompts(self):
        # All ranks should run this to have consistent prompts for FSDP generation
        if self.global_rank == 0:
            print(f"[Solver] Setting up validation prompts from {self.args.dataset_name}...")
        
        try:
            # Try loading validation split
            val_ds = load_dataset(self.args.dataset_name, split="validation")
        except Exception:
            try:
                if self.global_rank == 0: print("[Solver] Validation split not found, trying 'test' split...")
                val_ds = load_dataset(self.args.dataset_name, split="test")
            except Exception:
                if self.global_rank == 0: print("[Solver] Test split not found, falling back to 'train' split.")
                val_ds = load_dataset(self.args.dataset_name, split="train")

        # Sample 10 random indices with fixed seed for consistency
        rng = random.Random(42) 
        indices = rng.sample(range(len(val_ds)), min(10, len(val_ds)))
        
        self.validation_prompts = []
        text_keys = ['descriptions', 'text', 'caption', 'prompt']
        
        for idx in indices:
            item = val_ds[idx]
            caption = ""
            for k in text_keys:
                if k in item:
                    val = item[k]
                    if isinstance(val, list):
                         caption = val[0]
                    else:
                         caption = val
                    break
            if not caption:
                caption = "Generate an image."
            self.validation_prompts.append(caption)
            
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
        
        print(f"[Rank {self.global_rank}] [Validation] Generating images at step {step}...")
        self.model.eval()
        
        prompts = self.validation_prompts
        
        images = []
        for prompt in prompts:
            system_prompt = "Generate an image according to the text prompt."
            instruction = f"<system>{system_prompt}</system><user>{prompt}</user>"
            
            # Prepare Input
            instruction_ids = self.tokenizer(instruction, return_tensors="pt").input_ids.to("cuda")
            if instruction_ids[0, -1] == self.tokenizer.eos_token_id:
                instruction_ids = instruction_ids[:, :-1]
                
            input_ids = torch.cat([
                instruction_ids,
                torch.tensor([[BOA, BOI]], device="cuda")
            ], dim=1)
            
            code_start = input_ids.shape[1]
            
            # Append MASK tokens for generation
            _, _, h_grid, w_grid = calculate_vq_params(self.args.image_size, self.args.image_size)
            mask_tokens = [MASK] * (h_grid * w_grid)
            mask_with_breaks = add_break_line(mask_tokens, h_grid, w_grid, NEW_LINE)
            
            # Add MASKs + EOI + EOA
            input_ids = torch.cat([
                input_ids,
                torch.tensor([mask_with_breaks + [EOI, EOA]], device="cuda")
            ], dim=1)
            
            # Unconditional ids: try to match instruction format but with empty/generic user
            uncon_instruction = f"<system>{system_prompt}</system><user></user>"
            uncon_ids = self.tokenizer(uncon_instruction, return_tensors="pt").input_ids.to("cuda")
            if uncon_ids[0, -1] == self.tokenizer.eos_token_id:
                uncon_ids = uncon_ids[:, :-1]
            uncon_ids = torch.cat([uncon_ids, torch.tensor([[BOA, BOI]], device="cuda")], dim=1)

            try:
                # Generate (All ranks participate)
                out_tokens = generate_image(
                    self.model,
                    input_ids,
                    uncon_ids=uncon_ids,
                    code_start=code_start,
                    cfg_scale=4.0,
                    timesteps=64,
                    mask_token_id=MASK,
                    newline_id=NEW_LINE
                )
                
                # Only rank 0 decodes and prepares WandB image
                if self.global_rank == 0:
                    img = decode_vq_to_image(
                        out_tokens, 
                        save_path=None, 
                        vae_ckpt=None, 
                        image_height=self.args.image_size, 
                        image_width=self.args.image_size, 
                        vqvae=self.vqvae
                    )
                    images.append(wandb.Image(img, caption=prompt))
                
            except Exception as e:
                print(f"[Rank {self.global_rank}] [Validation] Error generating image for prompt '{prompt}': {e}")
        
        if self.global_rank == 0 and self.args.use_wandb and images:
            wandb.log({"val/generated_images": images}, step=step)
            
        self.model.train()
        dist.barrier() # Sync after validation

    def run(self):
        print("[Solver] Checking model parameters for NaNs...")
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"[Solver] FATAL: Parameter {name} contains NaNs!")
                sys.exit(1)
        
        # Load VQ-VAE for Training
        if not hasattr(self, 'vqvae'):
             if self.args.precision == "bf16":
                dtype = torch.bfloat16
             elif self.args.precision == "fp16":
                # raise ValueError("FP16 precision is not supported for VQ-VAE.")
                dtype = torch.float32 # Fallback to float32 for VQVAE
             else:
                dtype = torch.float32
             self.vqvae = VQModel.from_pretrained(self.args.init_from, subfolder="vqvae", torch_dtype=dtype).to("cuda")
             self.vqvae.eval()

        # Initialize global step
        steps_per_epoch = len(self.dataloader_train) // self.args.accum_iter
        self.global_step = self.start_epoch * steps_per_epoch
        if self.start_iter > 0:
            self.global_step += self.start_iter // self.args.accum_iter
        print(f"[Solver] Starting from Global Step: {self.global_step}")

        # Setup Validation Prompts
        self._setup_validation_prompts()

        # Initial validation logging to verify it works
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
        # save where
        print(f"[Solver] Checkpoint output dir: {self.args.output_dir}")
        
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
            # VQ-VAE loading logic (redundant safety)
            dtype = torch.float32 if self.args.precision != "bf16" else torch.bfloat16
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
        accumulated_loss = 0.0 
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
            images_tuple, captions_tuple = batch_data
            
            input_ids_list = []
            labels_list = []

            for i in range(len(images_tuple)):
                img = images_tuple[i]
                caption = captions_tuple[i]

                with torch.no_grad():
                    # encode_img_with_breaks returns tokens with newline separator
                    image_tokens = encode_img_with_breaks(img, self.vqvae)
                
                # 2. Mask Image Tokens (for training target)
                masked_image_tokens, image_labels = mask_codes(image_tokens)

                # 3. Prepare Prompt (Instruction + User Prompt)
                # Template: <system>Generate an image according to the text prompt.</system><user>{caption}</user>
                system_prompt = "Generate an image according to the text prompt."
                instruction = f"<system>{system_prompt}</system><user>{caption}</user>"
                
                instruction_token = self.tokenizer(instruction, truncation=True, max_length=512, padding=False, return_tensors="pt").input_ids[0].tolist()
                final_input = instruction_token + [BOA] + [BOI] + masked_image_tokens + [EOI] + [EOA]                
                
                instruction_label = [-100] * len(instruction_token)
                final_label = instruction_label + [-100] + [-100] + image_labels + [-100] + [-100]
                
                # Truncate if too long
                if len(final_input) > self.args.max_seq_len:
                    final_input = final_input[:self.args.max_seq_len]
                    final_label = final_label[:self.args.max_seq_len]

                input_ids_list.append(final_input)
                labels_list.append(final_label)

            examples = input_ids_list
            labels = labels_list
            
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[self.args.precision]:
                c_loss = self.model(examples, labels)

            loss = c_loss
            loss_value = loss.item()
            accumulated_loss += loss_value # Accumulate loss


            if not math.isfinite(loss_value):
                print(f"[Rank {self.global_rank}] Loss is {loss_value}, stopping training")
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
                        "train/epoch": epoch
                     })
                     accumulated_loss = 0.0 # Reset accumulator
                print(f"global step --- {self.global_step}")
                if self.global_step % self.args.validation_interval == 0 and self.args.use_wandb:
                    print("[Solver] Logging validation images on wandb...")
                    self.log_validation_images(self.global_step)

                # --- Step-based Saving ---
                if self.global_step % self.args.save_iteration_interval == 0:
                     self.save_checkpoint(epoch, iteration=data_iter_step, global_step=self.global_step)

            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

            # if self.global_rank == 0:
            #     if self.args.use_wandb:
            #         wandb.log({
            #             "train/loss": loss_value,
            #             "train/lr": self.optimizer.param_groups[0]["lr"],
            #             "train/epoch": epoch + data_iter_step / len(self.dataloader_train),
            #         }, step=data_iter_step + len(self.dataloader_train) * epoch)
            #         if (data_iter_step + 1) % accum_iter == 0:
            #              wandb.log({"train/grad_norm": grad_norm}, step=data_iter_step + len(self.dataloader_train) * epoch)

        metric_logger.synchronize_between_processes()
        print(f"Averaged stats: {metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    util.misc.random_seed(42)
    solver = Solver(args)
    solver.run()