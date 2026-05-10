# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import os
import re
import wandb
import yaml
import bitsandbytes as bnb
from dataclasses import dataclass, field
from time import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper,
)
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from tqdm import tqdm
import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu

def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def qwen2_flop_coefficients(config) -> tuple[float, float]:
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    dense_token_factor = 6.0 * dense_N
    attn_factor = 12.0 * head_dim * num_attention_heads * num_hidden_layers
    return dense_token_factor, attn_factor


def detect_peak_tflops(default_tflops: float) -> float:
    """Guess per-device BF16 TFLOPs from GPU name; fall back to default when unknown."""
    try:
        import torch
        device_name = torch.cuda.get_device_name()
    except (ImportError, RuntimeError):
        return default_tflops

    name = device_name.upper()
    if "MI300X" in name:
        tflops = 1336.0
    elif any(tag in name for tag in ("H100", "H800", "H200")):
        tflops = 989.0
    elif any(tag in name for tag in ("A100", "A800")):
        tflops = 312.0
    elif "L40" in name:
        tflops = 181.05
    elif "L20" in name:
        tflops = 119.5
    elif "H20" in name:
        tflops = 148.0
    elif "910B" in name:
        tflops = 354.0
    elif "RTX 3070 TI" in name:
        tflops = 21.75
    else:
        tflops = default_tflops
    return tflops


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    task: str = field(
        default=None,
        metadata={
            "help": "Task type for multi-task training. When set, overrides dataset_config_file. "
                    "Choices: counting, ocr, ocr_synthetic, celeb, rel_position. "
                    "Leave None to use dataset_config_file."
        }
    )
    mode: str = field(
        default="und",
        metadata={"help": "Task mode: 'und' (understanding only), 'gen' (generation only), 'both'."}
    )
    hf_dataset_path: str = field(
        default=None,
        metadata={"help": "HuggingFace dataset path for OCR task (e.g. 'naver-clova-ix/cord-v2')."}
    )
    validation_interval: int = field(
        default=500,
        metadata={"help": "Run validation every N training steps. Set 0 to disable."}
    )
    validation_samples: int = field(
        default=100,
        metadata={"help": "Number of samples to use for validation."}
    )
    eval_everything: bool = field(
        default=False,
        metadata={"help": "Run understanding validation regardless of --mode (e.g. even in gen-only mode)."}
    )
    eval_before_training: bool = field(
        default=False,
        metadata={"help": "Run validation once before training starts (at step 0)."}
    )
    val_gen_resolution: int = field(
        default=512,
        metadata={"help": "Resolution for validation image generation (default: 512)."}
    )
    val_gen_num_images: int = field(
        default=5,
        metadata={"help": "Number of images to generate during validation."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="never",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        # default=2000,
        default=0,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    peak_device_tflops: float = field(
        default=0.0,
        metadata={"help": "Per-GPU peak BF16 TFLOPs used to compute MFU; leave at 0 to auto-detect."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=False,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )

    # --- LoRA ---
    use_lora: bool = field(
        default=False,
        metadata={"help": "Enable LoRA (Low-Rank Adaptation) fine-tuning."}
    )
    lora_rank: int = field(
        default=128,
        metadata={"help": "LoRA rank (r)."}
    )
    lora_alpha: int = field(
        default=256,
        metadata={"help": "LoRA scaling alpha."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability."}
    )
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "Comma-separated list of module names to apply LoRA to."}
    )
    lora_ckpt_path: str = field(
        default=None,
        metadata={"help": "Path to a saved LoRA adapter checkpoint to resume from."}
    )


def build_task_dataset_meta(task, mode, hf_dataset_path=None):
    """Build dataset_meta dict dynamically from task and mode, instead of YAML.

    Returns a dict in the same format as a parsed YAML config file, suitable
    for passing to ``DataConfig(grouped_datasets=...)``.
    """
    UND_IMAGE_ARGS = {
        "image_stride": 14,
        "max_image_size": 980,
        "min_image_size": 378,
        "max_pixels": 2_007_040,
    }
    GEN_IMAGE_ARGS = {
        "image_stride": 16,
        "max_image_size": 1024,
        "min_image_size": 512,
    }

    dataset_meta = {}

    if task == "counting":
        if mode in ("und", "both"):
            dataset_meta["counting_und"] = {
                "dataset_names": ["pixmo"],
                "image_transform_args": dict(UND_IMAGE_ARGS),
                "is_mandatory": True,
                "num_used_data": [-1],
                "weight": 1,
            }
        if mode in ("gen", "both"):
            dataset_meta["counting_gen"] = {
                "dataset_names": ["pixmo"],
                "image_transform_args": dict(GEN_IMAGE_ARGS),
                "is_mandatory": mode == "gen",
                "num_used_data": [-1],
                "weight": 1,
            }

    elif task == "ocr":
        extra = {}
        if hf_dataset_path:
            extra["hf_dataset_path"] = hf_dataset_path
        if mode in ("und", "both"):
            dataset_meta["ocr_und"] = {
                "dataset_names": ["ocr"],
                "image_transform_args": dict(UND_IMAGE_ARGS),
                "is_mandatory": True,
                "num_used_data": [-1],
                "weight": 1,
                **extra,
            }
        if mode in ("gen", "both"):
            dataset_meta["ocr_gen"] = {
                "dataset_names": ["ocr"],
                "image_transform_args": dict(GEN_IMAGE_ARGS),
                "is_mandatory": mode == "gen",
                "num_used_data": [-1],
                "weight": 1,
                **extra,
            }

    elif task == "ocr_synthetic":
        if mode in ("und", "both"):
            dataset_meta["ocr_synthetic_und"] = {
                "dataset_names": ["sentences"],
                "image_transform_args": dict(UND_IMAGE_ARGS),
                "is_mandatory": True,
                "num_used_data": [-1],
                "weight": 1,
            }
        if mode in ("gen", "both"):
            dataset_meta["ocr_synthetic_gen"] = {
                "dataset_names": ["sentences"],
                "image_transform_args": dict(GEN_IMAGE_ARGS),
                "is_mandatory": mode == "gen",
                "num_used_data": [-1],
                "weight": 1,
            }

    elif task == "celeb":
        if mode in ("und", "both"):
            dataset_meta["celeb_und"] = {
                "dataset_names": ["celeb"],
                "image_transform_args": dict(UND_IMAGE_ARGS),
                "is_mandatory": True,
                "num_used_data": [-1],
                "weight": 1,
            }

    elif task == "rel_position":
        if mode == "und":
            dataset_meta["rel_position_und"] = {
                "dataset_names": ["rel_position"],
                "image_transform_args": dict(UND_IMAGE_ARGS),
                "is_mandatory": True,
                "num_used_data": [-1],
                "weight": 1,
            }
        elif mode == "gen":
            dataset_meta["rel_position_gen"] = {
                "dataset_names": ["rel_position"],
                "image_transform_args": dict(GEN_IMAGE_ARGS),
                "is_mandatory": True,
                "num_used_data": [-1],
                "weight": 1,
            }
        else:
            raise ValueError(
                f"rel_position supports 'und' or 'gen' mode only; got {mode!r}."
            )

    else:
        raise ValueError(f"Unknown task: {task!r}. Expected counting, ocr, ocr_synthetic, celeb, or rel_position.")

    if not dataset_meta:
        raise ValueError(f"No datasets configured for task={task!r}, mode={mode!r}")

    return dataset_meta


def _load_validation_dataset(task, mode, hf_dataset_path=None, num_samples=100):
    """Load a small validation split for the given task."""
    from datasets import load_dataset as hf_load_dataset

    if task == "counting":
        if mode == 'und':
            ds = hf_load_dataset("heez/pixmo-point-count-gen-und", split="val_und")
        elif mode == 'gen':
            ds = hf_load_dataset("heez/pixmo-point-count-gen-und", split="val_gen")
        else:
            raise ValueError(f"Invalid mode {mode} for counting task validation")
        # Use the last N samples as validation
        n = min(num_samples, len(ds))
        ds = ds.select(range(len(ds) - n, len(ds)))
        return ds

    elif task == "ocr":
        assert hf_dataset_path is not None, "hf_dataset_path required for OCR validation"
        ds = hf_load_dataset(hf_dataset_path, split="validation")
        n = min(num_samples, len(ds))
        ds = ds.select(range(n))
        return ds

    elif task == "ocr_synthetic":
        ds = hf_load_dataset("agentlans/high-quality-english-sentences", split="test")
        n = min(num_samples, len(ds))
        ds = ds.select(range(n))
        return ds

    elif task == "celeb":
        ds = hf_load_dataset("heez/celeb-recognition", split="test")
        n = min(num_samples, len(ds))
        ds = ds.select(range(n))
        return ds

    elif task == "rel_position":
        if mode not in ("und", "gen"):
            return None
        ds = hf_load_dataset("heez/relative-position-new", split="validation")
        n = min(num_samples, len(ds))
        ds = ds.select(range(n))
        return ds

    return None


def _setup_val_gen_prompts(val_ds, task, num_prompts=5):
    """Extract text prompts from val_ds for generation validation."""
    import random
    rng = random.Random(42)
    indices = rng.sample(range(len(val_ds)), min(num_prompts, len(val_ds)))
    prompts = []
    for idx in indices:
        item = val_ds[idx]
        if task == "counting":
            caption = item['descriptions']
            print(f"prompt: {caption}")
            if isinstance(caption, list):
                caption = caption[0]
        elif task == "ocr":
            answer = item.get("text", item.get("answer", "Hello World"))
            caption = f"An image containing the text: {answer}"
        elif task == "ocr_synthetic":
            answer = item.get("text", "Hello World").replace("\n", " ").strip()[:120]
            caption = (
                "A clean image with sharp, legible black text on white background. "
                f"The text reads: {answer}"
            )
        elif task == "celeb":
            persons = ["Heidi", "Samuel", "Elizabeth", "Benjamin", "Gabriel", "Julian"]
            prompts = [f"Generate an image of {p}." for p in persons for _ in range(2)]
            return prompts
        elif task == "rel_position":
            caption = item.get("answer", "")
        else:
            caption = "A beautiful landscape photograph."
        prompts.append(caption)
        
    return prompts


@torch.no_grad()
def validate_generation(
    model, vae_model, tokenizer, new_token_ids,
    prompts, resolution, device, logger,
    num_timesteps=50, cfg_scale=4.0, timestep_shift=3.0,
):
    """Generate images from text prompts and return wandb.Image list for logging."""
    from modeling.bagel.qwen2_navit import NaiveCache
    from PIL import Image as PILImage

    model = model.module if hasattr(model, 'module') else model
    model.eval()
    num_hidden_layers = model.config.llm_config.num_hidden_layers
    images = []

    for i, prompt in enumerate(prompts):
        try:
            # 1. Encode text prompt into KV cache
            past_key_values = NaiveCache(num_hidden_layers)
            newlens = [0]
            new_rope = [0]

            generation_input, newlens, new_rope = model.prepare_prompts(
                curr_kvlens=newlens, curr_rope=new_rope,
                prompts=[prompt], tokenizer=tokenizer,
                new_token_ids=new_token_ids,
            )
            generation_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in generation_input.items()}

            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = model.forward_cache_update_text(
                    past_key_values, **generation_input
                )

            # 2. Prepare latent generation input
            generation_input = model.prepare_vae_latent(
                curr_kvlens=newlens, curr_rope=new_rope,
                image_sizes=[(resolution, resolution)],
                new_token_ids=new_token_ids,
            )
            generation_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in generation_input.items()}

            # 3. Prepare CFG (text-only unconditional)
            cfg_past_key_values = NaiveCache(num_hidden_layers)
            cfg_newlens = [0]
            cfg_new_rope = [0]
            generation_input_cfg = model.prepare_vae_latent_cfg(
                curr_kvlens=cfg_newlens, curr_rope=cfg_new_rope,
                image_sizes=[(resolution, resolution)],
            )
            generation_input_cfg = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                    for k, v in generation_input_cfg.items()}

            # 4. Generate image (flow denoising)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                unpacked_latent = model.generate_image(
                    past_key_values=past_key_values,
                    num_timesteps=num_timesteps,
                    cfg_text_scale=cfg_scale,
                    cfg_interval=[0, 1.0],
                    cfg_renorm_min=0.0,
                    timestep_shift=timestep_shift,
                    cfg_text_past_key_values=cfg_past_key_values,
                    cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                    cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                    cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                    cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                    **generation_input,
                )

                # 5. Decode latent -> PIL image
                latent = unpacked_latent[0]
                h, w = resolution // 16, resolution // 16
                latent = latent.reshape(1, h, w, 2, 2, 16)
                latent = torch.einsum("nhwpqc->nchpwq", latent)
                latent = latent.reshape(1, 16, h * 2, w * 2)
                image = vae_model.decode(latent.to(device))
            image = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
            pil_img = PILImage.fromarray(image)
            pil_img.save(f"val_gen_{i}.png")

            images.append(wandb.Image(pil_img, caption=prompt[:100]))
            logger.info(f"  Generated image {i+1}/{len(prompts)}")

        except Exception as e:
            logger.warning(f"Generation validation error at prompt {i}: {e}")
            continue

    model.train()

    metrics = {}
    metrics["val_generated_images"] = images
    return metrics


def calculate_metrics(predictions, references, loaded_metrics=None):
    """Calculate OCR metrics: WER, CER, METEOR, BLEU, edit_distance, word P/R/F1."""
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


def _extract_position_label(text):
    """Return one quadrant label from text, or None when parsing is ambiguous."""
    matches = re.findall(r"(top-left|top-right|bottom-left|bottom-right)", str(text).lower())
    if len(matches) != 1:
        return None
    return matches[0]


@torch.no_grad()
def validate_understanding(
    model, tokenizer, new_token_ids, image_transform, val_ds, task, device, logger,
):
    """Run validation on a small dataset and return metrics.

    Uses Bagel.chat() for autoregressive inference, then computes:
    - counting: accuracy (exact match) and MAD (mean absolute deviation)
    - ocr / ocr_synthetic: exact-match accuracy and simple character error rate
    - rel_position: parsed quadrant accuracy over top-left / top-right / bottom-left / bottom-right
    """
    from data.data_utils import pil_img2rgb

    model = model.module if hasattr(model, 'module') else model
    model.eval()
    correct = 0
    total = 0
    mad_sum = 0.0
    pos_parse_fail = 0
    predictions = []
    references = []

    n_samples = min(100, len(val_ds))  # Limit to 20 for speed
    indices = list(range(n_samples))

    for idx in tqdm(indices, total=n_samples, desc="Validation(Und)", disable=dist.get_rank() != 0):
        try:
            item = val_ds[idx]

            if task == "counting":
                image = pil_img2rgb(item["image"])
                question = item.get("question_count") or item.get("question")
                answer_gt = str(item.get("answer_count") or item.get("answer"))
            elif task == "ocr":
                image = pil_img2rgb(item["image"])
                question = item.get("question") or "Extract all text from the image."
                answer_gt = item.get("answer", item.get("text", item.get("ground_truth", "")))
            elif task == "ocr_synthetic":
                from data.ocr_render import generate_image as generate_ocr_image
                raw_text = item["text"]
                answer_gt = raw_text.replace("\n", " ").strip()[:120]
                question = "Extract all text from the image."
                image = generate_ocr_image(
                    "# " + answer_gt, template="clean_light",
                    width=512, height=512, quality=100,
                )
                image = pil_img2rgb(image)
            elif task == "celeb":
                image = pil_img2rgb(item["image"])
                question = item.get("question", "")
                answer_gt = str(item.get("answer", ""))
            elif task == "rel_position":
                image = pil_img2rgb(item["image"])
                question = item.get("question", "")
                answer_gt = str(item.get("answer", ""))
            else:
                continue

            pred = model.chat(
                tokenizer, new_token_ids, image_transform,
                images=[image], prompt=question, max_length=128,
            )
            pred = pred.strip()

            if task == "counting":
                # Try to extract number from prediction
                pred_num = int("".join(c for c in pred if c.isdigit()) or "-1")
                gt_num = int("".join(c for c in answer_gt if c.isdigit()) or "-1")
                if pred_num == gt_num:
                    correct += 1
                mad_sum += abs(pred_num - gt_num)
            elif task == "celeb":
                if pred.strip().lower() == answer_gt.strip().lower():
                    correct += 1
            elif task == "rel_position":
                pred_pos = _extract_position_label(pred)
                gt_pos = _extract_position_label(answer_gt)
                if pred_pos is None or gt_pos is None:
                    pos_parse_fail += 1
                if pred_pos is not None and gt_pos is not None and pred_pos == gt_pos:
                    correct += 1
            else:
                # OCR: exact match + collect for nltk metrics
                if pred.strip().lower() == answer_gt.strip().lower():
                    correct += 1
                predictions.append(pred.strip())
                references.append(answer_gt.strip())
            print(f"="*50)
            print(f"Question: \n{question}")
            print(f"Prediction: \n{pred}")
            print(f"Ground Truth: \n{answer_gt}")

            total += 1

        except Exception as e:
            logger.warning(f"Validation error at idx {idx}: {e}")
            continue

    model.train()

    metrics = {}
    if total > 0:
        metrics["val_accuracy"] = correct / total
        if task == "counting":
            metrics["val_mad"] = mad_sum / total
        elif task == "rel_position":
            metrics["val/pos_acc"] = metrics["val_accuracy"]
            metrics["val/pos_parse_fail_rate"] = pos_parse_fail / total
        elif task in ("ocr", "ocr_synthetic") and predictions:
            loaded = {
                "wer": evaluate.load("wer"),
                "cer": evaluate.load("cer"),
                "meteor": evaluate.load("meteor"),
            }
            ocr_metrics = calculate_metrics(predictions, references, loaded_metrics=loaded)
            for k, v in ocr_metrics.items():
                metrics[f"val/{k}"] = v
    else:
        metrics["val_accuracy"] = 0.0
        if task == "rel_position":
            metrics["val/pos_acc"] = 0.0
            metrics["val/pos_parse_fail_rate"] = 0.0

    return metrics


def _save_lora_ckpt_no_fsdp(ckpt_dir, train_steps, model, optimizer, scheduler, data_status, logger, save_name=None, global_cumulative_samples=0):
    """Save LoRA checkpoint without FSDP (for LoRA training, with or without DDP)."""
    # Unwrap DDP if needed
    raw_model = model.module if hasattr(model, 'module') else model
    if save_name is None:
        save_name = f"step{train_steps}"
    save_path = os.path.join(ckpt_dir, save_name)
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Saving LoRA checkpoint to {save_path}")
    # Save LoRA adapter via PeftModel.save_pretrained (correct key format without .default.)
    raw_model.save_pretrained(save_path)
    logger.info(f"Saved LoRA adapter to {save_path}")
    # Save optimizer, scheduler, data_status
    torch.save(optimizer.state_dict(), os.path.join(save_path, "optimizer.00000-of-00001.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
    if data_status is not None:
        torch.save(data_status, os.path.join(save_path, "data_status.pt"))
    torch.save(global_cumulative_samples, os.path.join(save_path, "global_cumulative_samples.pt"))


def _do_checkpoint_save(
    training_args, skip_fsdp, fsdp_model, optimizer, scheduler,
    data_status, fsdp_config, curr_step, logger, save_name=None,
    global_cumulative_samples=0,
):
    """Gather data_status across ranks, save checkpoint, and cleanup CUDA cache."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        gather_list = [None] * dist.get_world_size()
    else:
        gather_list = None
    try:
        dist.gather_object(data_status, gather_list, dst=0)
    except RuntimeError as e:
        logger.error(f"Error during gather_object at step {curr_step}: {e}")
        gather_list = None if dist.get_rank() != 0 else [data_status] * dist.get_world_size()

    if skip_fsdp:
        if dist.get_rank() == 0:
            _save_lora_ckpt_no_fsdp(
                training_args.checkpoint_dir, curr_step, fsdp_model,
                optimizer, scheduler, gather_list, logger, save_name=save_name,
                global_cumulative_samples=global_cumulative_samples,
            )
    else:
        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir=training_args.checkpoint_dir, train_steps=curr_step,
            model=fsdp_model, ema_model=None, optimizer=optimizer,
            scheduler=scheduler, logger=logger, fsdp_config=fsdp_config,
            data_status=gather_list, use_lora=training_args.use_lora, save_name=save_name,
            global_cumulative_samples=global_cumulative_samples,
        )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.peak_device_tflops <= 0:
        auto_tflops = detect_peak_tflops(training_args.peak_device_tflops)
        if auto_tflops > 0:
            training_args.peak_device_tflops = auto_tflops

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())

        print(
        '''
        # ---------------------------------------------------------------------------- #
        #                                 WANDB STATUS                                 #
        # ---------------------------------------------------------------------------- #
        '''
        )
        print(f"W&B Project: {training_args.wandb_project}")
        print(f"W&B Run Name: {training_args.wandb_name}")
        print(f"W&B Run ID: {training_args.wandb_runid}")
        print(f"W&B Resume: {training_args.wandb_resume}")
        print(f"W&B Offline: {training_args.wandb_offline}")
        wandb.init(
            project=training_args.wandb_project, 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            id=training_args.wandb_runid if training_args.wandb_resume in ["must", "allow"] else None, # Only set ID if resuming, to avoid accidentally resuming when you meant to start fresh
            mode="offline" if training_args.wandb_offline else "online",
            settings=wandb.Settings(init_timeout=120)
        )
        wandb.config.update(training_args, allow_val_change = True)
        wandb.config.update(model_args, allow_val_change = True)
        wandb.config.update(data_args, allow_val_change = True)
        if training_args.peak_device_tflops > 0:
            logger.info(f"Using peak_device_tflops={training_args.peak_device_tflops:.2f} TFLOPs (per GPU).")
        else:
            logger.warning("Peak device TFLOPs not set or auto-detected; MFU will report 0.")
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    with init_empty_weights():
        if training_args.finetune_from_hf:
            llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
        else:
            llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
        llm_config.layer_module = model_args.layer_module
        llm_config.qk_norm = model_args.llm_qk_norm
        llm_config.tie_word_embeddings = model_args.tie_word_embeddings
        llm_config.freeze_und = training_args.freeze_und
        if training_args.finetune_from_hf:
            language_model = Qwen2ForCausalLM(llm_config)
        else:
            language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
        if training_args.copy_init_moe:
            language_model.init_moe()

        if training_args.visual_und:  
            if training_args.finetune_from_hf:
                vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
            else:
                vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
            vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
            vit_config.rope = model_args.vit_rope
            if training_args.finetune_from_hf:
                vit_model = SiglipVisionModel(vit_config)
            else:
                vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors") 
            if training_args.finetune_from_hf else model_args.vae_path
        )
        vae_model = vae_model.to(torch.bfloat16).to(device)

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        config
    )
    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    
    if training_args.finetune_from_hf:
        print("=" * 50)
        print(f"Loading PRETRAINED model weights from {model_args.model_path} (HuggingFace format)")
        print("=" * 50)
        # safetensors에서 GPU로 직접 로드
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(model_args.model_path, "ema.safetensors"), device=device)
        # assign=True가 핵심 — meta tensor를 실제 tensor로 "교체
        model.load_state_dict(state_dict, strict=False, assign=True)


    total_param_count = count_parameters(model)
    lm_param_count = count_parameters(model.language_model)
    logger.info(f"Model parameter count: {total_param_count / 1e9:.2f}B (LM-only: {lm_param_count / 1e9:.2f}B)")

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # --- LoRA Injection (before FSDP wrap) ---
    if training_args.use_lora:
        if training_args.freeze_llm:
            raise ValueError("Cannot use --use_lora with --freeze_llm; LoRA freezes base params itself.")

        from peft import LoraConfig, get_peft_model, PeftModel

        # Freeze all base parameters; LoRA will add trainable adapters
        for param in model.parameters():
            param.requires_grad = False

        target_modules = [m.strip() for m in training_args.lora_target_modules.split(",")]

        if training_args.lora_ckpt_path is not None:
            # Resume from a saved LoRA adapter
            print("=" * 50)
            print(f"Loading LoRA adapter from {training_args.lora_ckpt_path}")
            print("=" * 50)

            # Fix legacy key format: strip ".default." from adapter keys
            from safetensors.torch import load_file, save_file as sf_save
            adapter_path = os.path.join(training_args.lora_ckpt_path, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                sd = load_file(adapter_path)
                if any(".default." in k for k in sd.keys()):
                    sd = {k.replace(".default.", "."): v for k, v in sd.items()}
                    sf_save(sd, adapter_path)
                    logger.info("Fixed legacy .default. keys in adapter checkpoint")

            logger.info(f"Loading LoRA adapter from {training_args.lora_ckpt_path}")
            model = PeftModel.from_pretrained(
                model, training_args.lora_ckpt_path,
                is_trainable=True, torch_device="cpu",
            )
            # move adapter to GPU if not already there
            model = model.to(device)
        else:
            logger.info(f"Applying LoRA: rank={training_args.lora_rank}, alpha={training_args.lora_alpha}, "
                        f"dropout={training_args.lora_dropout}, targets={target_modules}")
            # Exclude gen branch (moe_gen) from LoRA when training understanding only
            lora_exclude = ".*moe_gen.*" if data_args.mode == "und" else None

            lora_config = LoraConfig(
                r=training_args.lora_rank,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules,
                exclude_modules=lora_exclude,
                lora_dropout=training_args.lora_dropout,
                bias="none",
            )
            model = get_peft_model(model, lora_config)

        # model.print_trainable_parameters()

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    # Skip base-model checkpoint load when resuming from LoRA adapter
    # (PeftModel.from_pretrained already loaded base + adapter)
    if not (training_args.use_lora and training_args.lora_ckpt_path is not None):
        model, _ = FSDPCheckpoint.try_load_ckpt(
            resume_from, logger, model, ema_model=None, resume_from_ema=finetune_from_ema
        )
    model = model.to(torch.bfloat16)

    # 1-GPU LoRA: skip FSDP to avoid full-model gradient buffers
    skip_fsdp = training_args.use_lora 

    if skip_fsdp:
        model = model.to(device)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=grad_checkpoint_check_fn
        )
        if dist.get_world_size() > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            fsdp_model = DDP(model, device_ids=[device], find_unused_parameters=True)
            logger.info(f"Wrapped LoRA model in DDP for {dist.get_world_size()}-GPU gradient sync")
        else:
            fsdp_model = model
    else:
        fsdp_model = fsdp_wrapper(model, fsdp_config, use_lora=training_args.use_lora)
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=grad_checkpoint_check_fn
        )

    # if dist.get_rank() == 0:
    #     print(fsdp_model)
    #     for name, param in model.named_parameters():
    #         print(name, param.requires_grad)

    # Setup optimizer and scheduler
    # LoRA + 8-bit AdamW: override dangerously small eps (1e-15) to prevent
    # numerical instability from quantized second moments hitting zero.
    opt_eps = training_args.eps
    if training_args.use_lora and opt_eps < 1e-8:
        logger.warning(f"LoRA + 8-bit AdamW: overriding eps={opt_eps} → 1e-8 for numerical stability")
        opt_eps = 1e-8

    # bitsandbytes 8-bit AdamW requires all tensors on GPU;
    # fall back to standard AdamW when FSDP cpu_offload is enabled.
    if training_args.cpu_offload:
        optimizer = torch.optim.AdamW(
            fsdp_model.parameters(),
            lr=training_args.lr,
            betas=(training_args.beta1, training_args.beta2),
            eps=opt_eps,
            weight_decay=0
        )
    else:
        optimizer = bnb.optim.AdamW8bit(
            fsdp_model.parameters(),
            lr=training_args.lr,
            betas=(training_args.beta1, training_args.beta2),
            eps=opt_eps,
            weight_decay=0
        )
    # LoRA: auto-add warmup to avoid full-lr cold start instability
    # if training_args.use_lora and training_args.warmup_steps == 0:
    #     training_args.warmup_steps = 200
    #     logger.warning(f"LoRA training: auto-setting warmup_steps=200 for stability")

    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
        restored_global_cumulative_samples = 0
        restored_last_saved_epoch = -1
    else:
        optimizer, scheduler, train_step, data_status, restored_global_cumulative_samples, restored_last_saved_epoch = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config,
        )
        

    # Setup packed dataloader
    if data_args.task is not None:
        # Dynamic config from --task / --mode
        dataset_meta = build_task_dataset_meta(
            data_args.task, data_args.mode, data_args.hf_dataset_path
        )
        logger.info(f"Built dataset config for task={data_args.task}, mode={data_args.mode}: "
                     f"{list(dataset_meta.keys())}")
    else:
        with open(data_args.dataset_config_file, "r") as stream:
            dataset_meta = yaml.safe_load(stream)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )

    total_dataset_samples = train_dataset.total_samples
    if total_dataset_samples and dist.get_rank() == 0:
        logger.info(f"Total dataset samples (1 epoch): {total_dataset_samples}")

    # Load validation dataset (only on rank 0 since validation runs there)
    val_ds = None
    val_image_transform = None
    if (data_args.task is not None
        and (data_args.validation_interval > 0 or data_args.eval_before_training)
        and (data_args.mode in ("und", "both") or data_args.eval_everything)
        and dist.get_rank() == 0):
        from data.transforms import ImageTransform
        val_ds = _load_validation_dataset(
            data_args.task, 'und', data_args.hf_dataset_path, data_args.validation_samples
        )
        val_image_transform = ImageTransform(
            max_image_size=980, min_image_size=378, image_stride=14, max_pixels=2_007_040
        )
        logger.info(f"Loaded validation dataset: {len(val_ds)} samples for task={data_args.task}")

    # Load generation validation prompts (only on rank 0)
    val_gen_prompts = None
    if (data_args.task is not None
        and (data_args.validation_interval > 0 or data_args.eval_before_training)
        and (data_args.mode in ("gen", "both") or data_args.eval_everything)
        and training_args.visual_gen
        and dist.get_rank() == 0):
        # Use the same val_ds if already loaded, otherwise load it
        _gen_val_ds = _load_validation_dataset(
            data_args.task, 'gen', data_args.hf_dataset_path, data_args.validation_samples
        )
        if _gen_val_ds is not None:
            val_gen_prompts = _setup_val_gen_prompts(
                _gen_val_ds, data_args.task, num_prompts=data_args.val_gen_num_images
            )
            logger.info(f"Loaded {len(val_gen_prompts)} generation validation prompts")
            for i, p in enumerate(val_gen_prompts):
                logger.info(f"  {i+1}. {p}")

    # Prepare models for training:
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()

    # eval before training
    if val_ds is not None and data_args.eval_before_training:
        if dist.get_rank() == 0:
            logger.info("Running validation before training (step 0)...")
            val_metrics = validate_understanding(
                model=fsdp_model, tokenizer=tokenizer,
                new_token_ids=new_token_ids, image_transform=val_image_transform,
                val_ds=val_ds, task=data_args.task, device=device, logger=logger,
            )
            val_message = "(step=0000000) Validation [before training]: "
            for k, v in val_metrics.items():
                val_message += f"{k}={v:.4f} "
            logger.info(val_message)
            print(val_message, flush=True)
            wandb.log(val_metrics)
            fsdp_model.train()
    dist.barrier()

    # eval before training - generation
    if val_gen_prompts is not None and data_args.eval_before_training:
        if dist.get_rank() == 0:
            logger.info("Running generation validation before training (step 0)...")
            gen_metrics = validate_generation(
                model=fsdp_model, vae_model=vae_model,
                tokenizer=tokenizer, new_token_ids=new_token_ids,
                prompts=val_gen_prompts, resolution=data_args.val_gen_resolution,
                device=device, logger=logger,
            )
            wandb.log(gen_metrics)
            fsdp_model.train()
    dist.barrier()

    # train loop
    start_time = time()
    cumulative_samples = int(restored_global_cumulative_samples) // max(dist.get_world_size(), 1)
    last_saved_epoch = restored_last_saved_epoch
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    optimizer.zero_grad()
    total_norm = torch.tensor(0.0, device=device)
    token_window = 0.0
    seqlen_square_window = 0.0
    accum_loss_dict = {}
    accum_ce_tokens = 0
    accum_mse_tokens = 0
    accum_count = 0
    dense_token_factor, attn_factor = qwen2_flop_coefficients(model.language_model.config)
    for micro_step, data in enumerate(train_loader):
        curr_step = train_step + micro_step // training_args.gradient_accumulation_steps
        if curr_step >= training_args.total_steps:
            logger.info(f"Reached total_steps={training_args.total_steps}, stopping training.")
            break
        data = data.cuda(device).to_dict()
        cumulative_samples += len(data['sample_lens'])
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)       
        tokens_tensor = torch.tensor(float(data['sequence_length']), device=device)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        token_window += tokens_tensor.item()
        if data['sample_lens']:
            sample_lens_tensor = torch.tensor(data['sample_lens'], dtype=torch.float32, device=device)
            sample_square = torch.dot(sample_lens_tensor, sample_lens_tensor)
            dist.all_reduce(sample_square, op=dist.ReduceOp.SUM)
            seqlen_square_window += sample_square.item()

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen and 'padded_images' in data:
                with torch.no_grad():
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'))
            try:
                loss_dict = fsdp_model(**data)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"CUDA OOM at step {curr_step}: {e}")
                    torch.cuda.empty_cache()
                raise e
        
        loss = 0
        ce = loss_dict["ce"]
        if ce is not None: # 
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.visual_gen and loss_dict["mse"] is not None:
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)

        for key, value in loss_dict.items():
            if key not in accum_loss_dict:
                accum_loss_dict[key] = 0.0
            accum_loss_dict[key] += value.item()
        accum_ce_tokens += total_ce_tokens.item()
        accum_mse_tokens += total_mse_tokens.item()
        accum_count += 1

        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()

        if (micro_step + 1) % training_args.gradient_accumulation_steps == 0:
            if skip_fsdp:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in fsdp_model.parameters() if p.requires_grad],
                    training_args.max_grad_norm,
                )
            else:
                total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Log loss values:
        if (curr_step % training_args.log_every == 0
            and (micro_step + 1) % training_args.gradient_accumulation_steps == 0):
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            elapsed = max(end_time - start_time, 1e-6)
            steps_per_sec = training_args.log_every / elapsed
            tokens_per_sec = token_window / elapsed
            tokens_per_step = token_window / training_args.log_every
            flops_all_token = dense_token_factor * token_window + attn_factor * seqlen_square_window
            actual_tflops = flops_all_token / elapsed / 1e12
            peak_total_tflops = training_args.peak_device_tflops * dist.get_world_size()
            mfu_value = actual_tflops / peak_total_tflops if peak_total_tflops > 0 else 0.0
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key in accum_loss_dict:
                avg_loss = torch.tensor(accum_loss_dict[key] / accum_count, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Global Steps/Sec: {steps_per_sec:.2f}, Tokens/Sec: {tokens_per_sec/1000:.2f}k, MFU: {mfu_value*100:.1f}%, "
            if total_dataset_samples:
                current_epoch = cumulative_samples * dist.get_world_size() / total_dataset_samples
                message += f"Epoch: {current_epoch:.3f}, Global processed Samples: {cumulative_samples * dist.get_world_size()}, Total Samples: {total_dataset_samples}, "
            logger.info(message)
            if dist.get_rank() == 0:
                print(message, flush=True)

            wandb_log['train_lr'] = optimizer.param_groups[0]['lr']
            wandb_log['train_total_mse_tokens'] = accum_mse_tokens
            wandb_log['train_total_ce_tokens'] = accum_ce_tokens
            wandb_log['train_total_norm'] = total_norm.item()
            wandb_log['train_total_samples'] = total_samples.item()
            wandb_log['train_tokens_per_sec'] = tokens_per_sec
            wandb_log['train_tokens_per_step'] = tokens_per_step
            wandb_log['train_actual_tflops'] = actual_tflops
            wandb_log['train_mfu'] = mfu_value

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['train_mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['train_mem_cache'] = mem_cache
            if total_dataset_samples:
                wandb_log['train_epoch'] = current_epoch

            if dist.get_rank() == 0:
                wandb.log(wandb_log)
            start_time = time()
            token_window = 0.0
            seqlen_square_window = 0.0
            accum_loss_dict = {}
            accum_ce_tokens = 0
            accum_mse_tokens = 0
            accum_count = 0

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        # Inline validation
        is_val_step = (data_args.validation_interval > 0
            and curr_step > 0
            and curr_step % data_args.validation_interval == 0
            and (micro_step + 1) % training_args.gradient_accumulation_steps == 0)

        if is_val_step and val_ds is not None:
            if dist.get_rank() == 0:
                logger.info(f"Running validation at step {curr_step}...")
                val_metrics = validate_understanding(
                    model=fsdp_model,
                    tokenizer=tokenizer,
                    new_token_ids=new_token_ids,
                    image_transform=val_image_transform,
                    val_ds=val_ds,
                    task=data_args.task,
                    device=device,
                    logger=logger,
                )
                val_message = f"(step={curr_step:07d}) Validation: "
                for k, v in val_metrics.items():
                    val_message += f"{k}={v:.4f} "
                logger.info(val_message)
                print(val_message, flush=True)
                wandb.log(val_metrics)
                fsdp_model.train()
        dist.barrier()

        # Inline generation validation
        if is_val_step and val_gen_prompts is not None:
            if dist.get_rank() == 0:
                logger.info(f"Running generation validation at step {curr_step}...")
                gen_metrics = validate_generation(
                    model=fsdp_model, vae_model=vae_model,
                    tokenizer=tokenizer, new_token_ids=new_token_ids,
                    prompts=val_gen_prompts, resolution=data_args.val_gen_resolution,
                    device=device, logger=logger,
                )
                if gen_metrics:
                    wandb.log(gen_metrics)
                fsdp_model.train()
        dist.barrier()

        # Epoch-based checkpoint save
        if (total_dataset_samples
            and (micro_step + 1) % training_args.gradient_accumulation_steps == 0):
            # Synchronize cumulative_samples across ranks to avoid deadlock
            # (each rank may count slightly different samples due to packing)
            global_cumulative = torch.tensor(float(cumulative_samples), device=device)
            dist.all_reduce(global_cumulative, op=dist.ReduceOp.SUM)
            current_epoch_for_save = global_cumulative.item() / total_dataset_samples
            completed_epoch = int(current_epoch_for_save)
            if completed_epoch > last_saved_epoch and completed_epoch > 0: # 바뀌는 시점에 저장
                last_saved_epoch = completed_epoch
                logger.info(f"Epoch {completed_epoch} completed at step {curr_step}, saving checkpoint...")
                _do_checkpoint_save(
                    training_args, skip_fsdp, fsdp_model, optimizer, scheduler,
                    data_status, fsdp_config, curr_step, logger,
                    save_name=f"epoch{completed_epoch}",
                    global_cumulative_samples=global_cumulative.item(),
                )

        # Step-based checkpoint save
        if (curr_step > 0
            and curr_step % training_args.save_every == 0
            and (micro_step + 1) % training_args.gradient_accumulation_steps == 0):
            global_cumulative = torch.tensor(float(cumulative_samples), device=device)
            dist.all_reduce(global_cumulative, op=dist.ReduceOp.SUM)
            epoch_int = int(global_cumulative.item() / total_dataset_samples) if total_dataset_samples else 0
            save_name = f"epoch{epoch_int}-step{curr_step}"
            _do_checkpoint_save(
                training_args, skip_fsdp, fsdp_model, optimizer, scheduler,
                data_status, fsdp_config, curr_step, logger,
                save_name=save_name,
                global_cumulative_samples=global_cumulative.item(),
            )

    # Save final checkpoint if not already saved
    if curr_step > 0:
        logger.info(f"Saving final checkpoint at step {curr_step}...")
        global_cumulative = torch.tensor(float(cumulative_samples), device=device)
        dist.all_reduce(global_cumulative, op=dist.ReduceOp.SUM)
        epoch_int = int(global_cumulative.item() / total_dataset_samples) if total_dataset_samples else 0
        save_name = f"epoch{epoch_int}-step{curr_step}"
        _do_checkpoint_save(
            training_args, skip_fsdp, fsdp_model, optimizer, scheduler,
            data_status, fsdp_config, curr_step, logger,
            save_name=save_name,
            global_cumulative_samples=global_cumulative.item(),
        )
        logger.info(f"Final checkpoint saved at step {curr_step}")
    
    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
