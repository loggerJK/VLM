# Copyright 2025 MMaDA Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# T2I counting generation fine-tuning script for MMaDA.
# Based on train_mmada_counting.py; flips the learning objective from MMU
# (image -> count text) to T2I (descriptions -> image with given count),
# mirroring Lumina-DiMOO `task=counting, mode=gen`.
#
# Key design notes (see plans/iridescent-roaming-mccarthy.md for full rationale):
#   - Uses new CountingGenDataset (descriptions != '' subset of pixmo-point-count-gen-und).
#   - T2I-only training: batch_size_lm = batch_size_mmu = 0.
#   - T2ITrainStepWrapper computes T2I CE loss inline (device-safe), avoiding both
#     `forward_process` (missing batch_size_mmu==0 guard) and `forward_t2i`
#     (untested attention_bias device bug).
#   - Validation: val_gen split only; wandb image table logging (no auto count metric).
#   - `t2i_generate(..., resolution=max_seq_length)` is REQUIRED — default 512 only
#     happens to match demo/stage3-cot configs; mismatched resolution corrupts CFG.

import io
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, InitProcessGroupKwargs, set_seed

from training.utils import get_config, flatten_omega_conf
from parquet import CountingGenDataset

from models import MAGVITv2, get_mask_schedule, MMadaModelLM, MMadaConfig
from training.prompting_utils import UniversalPrompting
from peft import LoraConfig, get_peft_model, PeftModel
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from tqdm import tqdm

from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter

try:
    import apex
    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


class T2ITrainStepWrapper(torch.nn.Module):
    """T2I-only train-step wrapper.

    Routes train step through `forward()` so DDP / DeepSpeed reducer bookkeeping
    (`_post_forward` -> `prepare_for_backward`) fires per step. Computes T2I CE
    loss inline, replicating `MMadaModelLM.forward_process` L228-244 — but with
    `device=input_ids.device` explicitly so the CUDA model never receives a CPU
    `attention_bias`.

    Why not delegate to `base.forward_t2i(...)`? Because that method (modeling_mmada.py:355)
    creates `attention_bias` without `device=`, which breaks on CUDA — it is dead
    code in this repo (no callers). Why not `base.forward_process(...)`? Because
    L246-249, L266-270 unconditionally dereference `p_mask_mmu` / `answer_lengths`
    and slice `[-batch_size_mmu:]` (which equals `[0:]` when batch_size_mmu==0),
    so T2I-only calls crash with `NoneType.to(...)` or compute MMU loss over the
    entire batch.

    Other accesses (`mmu_generate`, `t2i_generate`, `config`, `save_pretrained`)
    fall back to base via __getattr__ so `accelerator.unwrap_model(model).<method>`
    call sites work unchanged.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, input_ids, labels, batch_size_t2i, max_seq_length, t2i_masks):
        # device-safe attention_bias (forward_process L228-234 pattern)
        attention_bias = torch.ones(
            input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1],
            device=input_ids.device,
        )
        if batch_size_t2i > 0 and t2i_masks is not None:
            attention_bias_t2i = (t2i_masks[:, :, None] & t2i_masks[:, None, :]).bool().unsqueeze(1)
            attention_bias[:batch_size_t2i] = attention_bias_t2i.to(attention_bias.device)
        logits = self.base(input_ids, attention_bias=attention_bias).logits

        # T2I CE slicing (forward_process L241-244 verbatim)
        loss_t2i = F.cross_entropy(
            logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, logits.shape[-1]),
            labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1),
            ignore_index=-100,
        )
        return loss_t2i

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = get_config()

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
        kwargs_handlers=[ddp_kwargs],
    )

    total_batch_size_per_gpu = (
        config.training.batch_size_t2i
        + config.training.batch_size_lm
        + config.training.batch_size_mmu
    )
    total_batch_size = (
        total_batch_size_per_gpu
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = total_batch_size_per_gpu

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")
        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    accelerator.wait_for_everyone()

    if config.training.seed is not None:
        set_seed(config.training.seed)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.mmada.tokenizer_path, padding_side="left"
    )

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
            "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True,
    )

    print("special tokens:\n", uni_prompting.sptids_dict)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = vq_model().to(accelerator.device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    model = MMadaModelLM.from_pretrained(
        config.model.mmada.pretrained_model_path, torch_dtype=torch.bfloat16
    ).to(accelerator.device)

    mask_id = model.config.mask_token_id

    lora_cfg_node = config.get("lora", None)
    use_lora = bool(lora_cfg_node and lora_cfg_node.get("enable", False))
    if use_lora:
        lora_cfg = LoraConfig(
            r=lora_cfg_node.rank,
            lora_alpha=lora_cfg_node.alpha,
            lora_dropout=lora_cfg_node.dropout,
            target_modules=list(lora_cfg_node.target_modules),
            bias=lora_cfg_node.bias,
            task_type=None,
        )
        model = get_peft_model(model, lora_cfg)
        if accelerator.is_main_process:
            model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    optimizer_config = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )

    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **args)
    else:
        mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    # ------------------------------------------------------------------
    # Dataloaders (T2I-only via CountingGenDataset)
    # ------------------------------------------------------------------
    logger.info("Creating dataloaders")

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    assert config.training.batch_size_t2i > 0, (
        "T2I-only counting_gen training requires batch_size_t2i > 0"
    )
    assert config.training.batch_size_lm == 0, "batch_size_lm must be 0 in counting_gen"
    assert config.training.batch_size_mmu == 0, "batch_size_mmu must be 0 in counting_gen"

    dataset_counting_gen = CountingGenDataset(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        resolution=preproc_config.resolution,
        num_workers=dataset_config.num_workers,
        count_lower_limit=dataset_config.get("count_lower_limit", 0),
        count_upper_limit=dataset_config.get("count_upper_limit", 20),
        shuffle=True,
        repeat=True,
        buffer_size=dataset_config.get("shuffle_buffer_size", 100),
    )
    train_dataloader_t2i = DataLoader(
        dataset_counting_gen,
        batch_size=config.training.batch_size_t2i,
        sampler=None,
        collate_fn=dataset_counting_gen.collate_fn,
        num_workers=dataset_config.num_workers,
    )
    num_gen_samples = dataset_counting_gen.num_samples
    num_update_steps_per_epoch = max(1, num_gen_samples // total_batch_size)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
    logger.info(
        f"gen-subset size: {num_gen_samples}, steps/epoch: {num_update_steps_per_epoch}, "
        f"total_epochs: {num_train_epochs}"
    )

    iterables = {"t2i_flow": train_dataloader_t2i}
    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

    # ------------------------------------------------------------------
    # Pre-load val_gen data (once, every rank holds full list, stride sharded inside validate_*)
    # ------------------------------------------------------------------
    val_gen_items = []
    logger.info("Pre-loading counting-gen validation data (val_gen split)...")
    try:
        from datasets import load_dataset as hf_load_dataset
        val_ds = hf_load_dataset("heez/pixmo-point-count-gen-und", split="val_gen")
        max_val = config.experiment.get("max_val_counting_gen_samples", 16)
        n = min(max_val, len(val_ds))
        val_gen_items = [val_ds[i] for i in range(n)]
        logger.info(f"Loaded {len(val_gen_items)} val_gen samples.")
    except Exception as e:
        logger.warning(f"Could not load val_gen split: {e}")

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir) if os.path.exists(config.experiment.output_dir) else []
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if dirs else None
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)
            logger.info(f"Resuming from checkpoint: {path}")
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            adapter_dir = f'{path}/unwrapped_model'
            has_adapter = os.path.exists(f'{adapter_dir}/adapter_config.json')
            if use_lora and has_adapter:
                from peft import set_peft_model_state_dict
                from peft.utils import load_peft_weights
                adapter_state = load_peft_weights(adapter_dir)
                set_peft_model_state_dict(model, adapter_state, adapter_name="default")
            elif os.path.exists(f'{path}/unwrapped_model/pytorch_model.bin'):
                state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
                model.load_state_dict(state_dict, strict=True)
                del state_dict
            elif os.path.exists(f'{path}/unwrapped_model/model.safetensors.index.json'):
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(model, f'{path}/unwrapped_model/')
            else:
                raise FileNotFoundError(f"Checkpoint not found under {path}/unwrapped_model/")

    # ------------------------------------------------------------------
    # Accelerate prepare
    # ------------------------------------------------------------------
    model = T2ITrainStepWrapper(model)

    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    vq_model.to(device=accelerator.device)

    # ------------------------------------------------------------------
    # Inner helper functions
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prepare_inputs_and_labels(
        pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
        texts,
        min_masking_rate: float = 0.0,
        is_train: bool = True,
    ):
        image_tokens = vq_model.get_code(pixel_values_or_image_ids)
        image_tokens = image_tokens + len(uni_prompting.text_tokenizer)
        input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
            image_tokens, mask_id, config, mask_schedule=mask_schedule, is_train=is_train,
        )
        input_ids, masks, labels = uni_prompting((texts, input_ids, labels), 't2i')
        return input_ids, labels, mask_prob, image_tokens, masks

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if config.experiment.get("validate_before_train", False):
        validate_counting_gen(
            model, vq_model, uni_prompting, accelerator, config,
            global_step, val_gen_items, mask_schedule,
        )
        accelerator.wait_for_everyone()

    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Total training epochs = {num_train_epochs}")
    logger.info(f"  Steps per epoch = {num_update_steps_per_epoch}")

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        num_update_step = 0
        accum_loss_t2i = 0.0
        accum_masking_rate = 0.0
        accum_log_steps = 0

        for batch, batch_idx, dataloader_idx in combined_dataloader:
            data_time_m.update(time.time() - end)

            # T2I (only flow)
            assert "t2i_flow" in batch, "t2i_flow missing — counting_gen requires it"
            batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
            pixel_values = batch["t2i_flow"]["images"].to(accelerator.device, non_blocking=True)
            texts = batch["t2i_flow"]["input_ids"]
            (
                input_ids_t2i, labels_t2i, mask_prob, image_tokens_ori, t2i_masks
            ) = prepare_inputs_and_labels(pixel_values, texts, config.training.min_masking_rate)

            # Step-0 sanity (pure image-region masked count, not text/soi/eoi)
            if global_step == 0 and accelerator.is_main_process:
                msl = config.dataset.preprocessing.max_seq_length
                # sequence: [pad...<|t2i|><bos>text<eos>] (msl+1) | <|soi|>(1) | image(1024) | <|eoi|>(1)
                img_labels = labels_t2i[:, msl + 2:-1]
                n_masked_img = (img_labels != -100).sum().item()
                print("=== T2I COUNTING-GEN SANITY (sample 0) ===")
                print(f"caption[0]: {texts[0]!r}")
                print(f"images.shape: {batch['t2i_flow']['images'].shape}")
                print(
                    f"input_ids_t2i.shape: {input_ids_t2i.shape}, "
                    f"masked image tokens (img region only): {n_masked_img}, "
                    f"t2i_masks.shape: {t2i_masks.shape}"
                )

            input_ids = input_ids_t2i.to(accelerator.device)
            labels = labels_t2i.to(accelerator.device)

            with accelerator.accumulate(model):
                loss_t2i = model(
                    input_ids=input_ids,
                    labels=labels,
                    batch_size_t2i=batch_size_t2i,
                    max_seq_length=config.dataset.preprocessing.max_seq_length,
                    t2i_masks=t2i_masks,
                )

                avg_loss_t2i = accelerator.gather(
                    loss_t2i.repeat(config.training.batch_size_t2i)
                ).mean()
                avg_masking_rate = accelerator.gather(
                    mask_prob.repeat(config.training.batch_size_t2i)
                ).mean()
                accum_loss_t2i += avg_loss_t2i.item()
                accum_masking_rate += avg_masking_rate.item()
                accum_log_steps += 1

                loss = config.training.t2i_coeff * loss_t2i

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()
                num_update_step += 1
                log_denom = max(accum_log_steps, 1)
                log_loss_t2i = accum_loss_t2i / log_denom
                log_masking_rate = accum_masking_rate / log_denom

                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps
                        * total_batch_size_per_gpu
                        / batch_time_m.val
                    )
                    epoch_progress = epoch + (num_update_step - 1) / num_update_steps_per_epoch
                    logs = {
                        "step_loss_t2i": log_loss_t2i,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": log_masking_rate,
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                        "epoch": epoch_progress,
                    }
                    accelerator.log(logs, step=global_step + 1)
                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Epoch: {epoch_progress:.2f} "
                        f"Loss_t2i: {log_loss_t2i:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f} "
                        f"Mask: {log_masking_rate:0.4f} "
                        f"Samples/sec/gpu: {samples_per_second_per_gpu:0.2f} "
                        f"Data: {data_time_m.val:0.3f}s "
                        f"Batch: {batch_time_m.val:0.3f}s"
                    )
                    batch_time_m.reset()
                    data_time_m.reset()
                accum_loss_t2i = 0.0
                accum_masking_rate = 0.0
                accum_log_steps = 0

                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1, uni_prompting)

                need_eval = (global_step + 1) % config.experiment.eval_every == 0
                if need_eval:
                    validate_counting_gen(
                        model, vq_model, uni_prompting, accelerator, config,
                        global_step + 1, val_gen_items, mask_schedule,
                    )

                accelerator.wait_for_everyone()

                global_step += 1

            if global_step >= config.training.max_train_steps:
                break

    accelerator.wait_for_everyone()
    save_checkpoint(model, config, accelerator, global_step, uni_prompting)

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        if isinstance(unwrapped, T2ITrainStepWrapper):
            unwrapped = unwrapped.base
        unwrapped.save_pretrained(config.experiment.output_dir, safe_serialization=True)
        if use_lora:
            merged_dir = os.path.join(config.experiment.output_dir, "final_merged")
            unwrapped.merge_and_unload().save_pretrained(merged_dir, safe_serialization=True)
            uni_prompting.text_tokenizer.save_pretrained(merged_dir)

    accelerator.end_training()


# ---------------------------------------------------------------------------
# Counting-Gen validation (val_gen split, image generation -> wandb table)
# ---------------------------------------------------------------------------

def _pil_to_png_bytes(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


@torch.no_grad()
def validate_counting_gen(
    model,
    vq_model,
    uni_prompting,
    accelerator,
    config,
    global_step,
    val_items,
    mask_schedule,
):
    """Generate images for val_gen captions and log a wandb image table.

    Mirrors Lumina-DiMOO `log_validation_images` (no auto count-accuracy metric).
    All ranks run generation (stride sharded), gather PNG bytes via
    `dist.all_gather_object` (more robust than PIL pickle), rank 0 builds the
    wandb table.
    """
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    is_main = accelerator.is_main_process

    if is_main:
        logger.info("Running counting-gen validation...")
    model.eval()

    if not val_items:
        if is_main:
            logger.warning("validate_counting_gen: empty val_items, skipping.")
        model.train()
        return

    device = accelerator.device
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    mask_token_id = accelerator.unwrap_model(model).config.mask_token_id
    num_vq_tokens = config.model.mmada.num_vq_tokens
    codebook_size = config.model.mmada.codebook_size
    msl = config.dataset.preprocessing.max_seq_length

    local_results = []
    local_items = [
        (sample_idx, item)
        for sample_idx, item in enumerate(val_items)
        if sample_idx % world_size == rank
    ]

    if is_main:
        print("=" * 70)
        print(
            f"Counting-Gen validation @ step {global_step}  "
            f"({len(val_items)} samples, {world_size} ranks)"
        )
        print("=" * 70)

    iterator = tqdm(
        local_items,
        total=len(local_items),
        desc=f"Counting-Gen val rank {rank}",
        position=rank,
        leave=True,
        dynamic_ncols=True,
    )

    for sample_idx, item in iterator:
        try:
            caption = item.get('descriptions', None)
            if caption is None:
                continue
            if isinstance(caption, list):
                caption = caption[0] if caption else ''
            if not isinstance(caption, str):
                continue
            caption = caption.strip()
            if not caption:
                continue

            gt_count = int(item.get('count', -1))
            gt_image = item.get('image', None)
            if not isinstance(gt_image, Image.Image) and gt_image is not None:
                gt_image = Image.fromarray(gt_image).convert('RGB')
            elif isinstance(gt_image, Image.Image):
                gt_image = gt_image.convert('RGB')

            # Build T2I input: all-mask image tokens + caption.
            image_tokens = torch.ones(
                (1, num_vq_tokens), dtype=torch.long, device=device
            ) * mask_token_id
            input_ids, attention_mask = uni_prompting(([caption], image_tokens), 't2i_gen')

            cfg_scale = config.training.guidance_scale
            if cfg_scale > 0:
                uncond_input_ids, uncond_attention_mask = uni_prompting(
                    ([''], image_tokens), 't2i_gen'
                )
            else:
                uncond_input_ids = None
                uncond_attention_mask = None

            with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
                gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    uncond_attention_mask=uncond_attention_mask,
                    guidance_scale=cfg_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=config.training.get("noise_type", "mask"),
                    predict_all_tokens=config.training.get("predict_all_tokens", False),
                    seq_len=num_vq_tokens,
                    # CRITICAL: 'resolution' here is actually the CFG text-prefix length;
                    # default 512 corresponds to demo/stage3-cot max_seq_length=512. With
                    # our msl=256, the default would slice into image tokens and corrupt CFG.
                    resolution=msl,
                    uni_prompting=uni_prompting,
                    config=config,
                )

            gen_token_ids = torch.clamp(gen_token_ids, max=codebook_size - 1, min=0)
            gen_images = vq_model.decode_code(gen_token_ids)
            gen_images = torch.clamp((gen_images + 1.0) / 2.0, min=0.0, max=1.0)
            gen_images = (gen_images * 255.0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            gen_pil = Image.fromarray(gen_images[0])

            gen_bytes = _pil_to_png_bytes(gen_pil)
            gt_bytes = _pil_to_png_bytes(gt_image) if gt_image is not None else b''

            local_results.append({
                "sample_idx": sample_idx,
                "caption": caption,
                "gt_count": gt_count,
                "gt_image_bytes": gt_bytes,
                "gen_image_bytes": gen_bytes,
            })

            print(
                f"[rank {rank}] sample={sample_idx:>3} gt_count={gt_count:>2} "
                f"caption={caption[:80]!r}{'...' if len(caption) > 80 else ''}"
            )
        except Exception as e:
            logger.warning(f"Counting-gen val error rank {rank} idx {sample_idx}: {e}")
            continue

    # Gather across ranks.
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        all_results = [None for _ in range(world_size)]
        dist.barrier()
        dist.all_gather_object(all_results, local_results)
        results = [item for shard in all_results if shard for item in shard]
    else:
        results = local_results

    # Non-main ranks return after restoring train mode (see plan F.10).
    if not is_main:
        model.train()
        return

    results = sorted(results, key=lambda x: x["sample_idx"])
    if not results:
        logger.warning("validate_counting_gen: no results gathered.")
        model.train()
        return

    pred_table = wandb.Table(columns=["sample_idx", "caption", "gt_count", "gt_image", "gen_image"])
    for r in results:
        gt_img_obj = (
            wandb.Image(Image.open(io.BytesIO(r["gt_image_bytes"])))
            if r["gt_image_bytes"] else None
        )
        gen_img_obj = wandb.Image(Image.open(io.BytesIO(r["gen_image_bytes"])))
        pred_table.add_data(r["sample_idx"], r["caption"], r["gt_count"], gt_img_obj, gen_img_obj)

    wandb.log({
        "counting_gen/val_images": pred_table,
    }, step=global_step)

    print("-" * 70)
    print(f"Logged {len(results)} counting-gen validation samples to wandb @ step {global_step}")
    print("=" * 70)

    logger.info(
        f"Counting-gen validation step {global_step}: logged {len(results)} samples"
    )

    model.train()


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(model, config, accelerator, global_step, uni_prompting):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint")],
            key=lambda x: int(x.split("-")[1]),
        )
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            for removing_checkpoint in checkpoints[:num_to_remove]:
                shutil.rmtree(os.path.join(output_dir, removing_checkpoint))

    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        if isinstance(unwrapped_model, T2ITrainStepWrapper):
            unwrapped_model = unwrapped_model.base
            state_dict = {
                k[len("base."):]: v
                for k, v in state_dict.items()
                if k.startswith("base.")
            }
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        uni_prompting.text_tokenizer.save_pretrained(save_path / "unwrapped_model")
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
