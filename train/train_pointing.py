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
from utils.image_utils import encode_img_with_breaks, generate_crop_size_list, var_center_crop


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

    def process_item(self, data_item: dict, training_mode=False) -> Tuple[Any, str, str]:
        image = data_item.get('image')
        question = data_item.get('question', data_item.get('text', ''))
        answer = str(data_item.get('answer', data_item.get('label', '')))

        # Image Preprocessing (Crop)
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
        parser.add_argument("--image_size", type=int, default=256, help="Image size for preprocessing (default: 256)")
        parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU Offloading for FSDP (default: False)")
        
        # WandB Arguments
        parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
        parser.add_argument("--wandb_project", type=str, default="lumina-dimoo-finetune", help="WandB project name")
        parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity name")
        parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
        return parser
    
    def setup_fsdp_sync(
        self, model: nn.Module, data_parallel: str, precision: str, grad_precision: str = None
    ) -> FSDP:
        print(f"[Solver] setup_fsdp_sync: cpu_offload={self.args.cpu_offload}")
        
        if self.dp_rank == 0:
            param_init_fn = None
        else:
            param_init_fn = lambda x: x.to_empty(device=torch.cuda.current_device(), recurse=False)

        # Set CPU Offload based on argument
        cpu_offload = CPUOffload(offload_params=self.args.cpu_offload)

        model = FSDP(
            model,
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
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
                config=vars(self.args)
            )

    def build_model(self):
        # 1. Call super build_model to get model and tokenizer
        model, tokenizer, _ = super().build_model()
        
        # 2. Use 8-bit AdamW for memory efficiency
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
            
        return model, tokenizer

    def _item_processor_func(self, tokenizer=None, max_len=None) -> ItemProcessorBase:
        # Pass image_size from args
        return ItemProcessorPointing(tokenizer, max_len, image_size=self.args.image_size)

    def _dataset_func(self):
        print("[Solver] Loading Hugging Face Datasets...")
        train_ds = load_dataset("Jiwon-Kang/pixmo-point-count-concat_0-20-qaFixed", split="train")
        
        try:
            val_ds = load_dataset("Jiwon-Kang/pixmo-count-filtered-imgContained", split="train")
            self.dataset_val = HFDatasetWrapper(val_ds, self._item_processor_func(self.tokenizer, self.args.max_seq_len))
        except Exception as e:
            print(f"Warning: Could not load validation set: {e}")
            self.dataset_val = None

        item_processor = self._item_processor_func(tokenizer=self.tokenizer, max_len=self.args.max_seq_len)
        return HFDatasetWrapper(train_ds, item_processor)

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
                util.ckpt.save(
                    self.args.output_dir,
                    self.global_rank == 0,
                    self.model,
                    self.optimizer,
                    self.tokenizer,
                    self.args,
                    epoch=epoch,
                    max_keep=self.args.ckpt_max_keep,
                )

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
                print(f"instruction_token length: {len(instruction_token)}, answer_token length: {len(answer_token)}")
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
                c_loss = self.model(examples, labels)

            loss = c_loss
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print(f"[Rank {self.global_rank}] Loss is {loss_value}, stopping training")
                print(f"[Rank {self.global_rank}] Input IDs (first sample, first 50): {examples[0][:50]}")
                print(f"[Rank {self.global_rank}] Labels (first sample, first 50): {labels[0][:50]}")
                print(f"[Rank {self.global_rank}] Input Min/Max: {min([min(x) for x in examples])}/{max([max(x) for x in examples])}")
                sys.exit(1)

            effective_loss = loss / accum_iter
            effective_loss.backward()

            if (data_iter_step + 1) % accum_iter == 0:
                self.model.clip_grad_norm_(max_norm=self.args.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            torch.cuda.synchronize()
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        print(f"Averaged stats: {metric_logger}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    solver.run()