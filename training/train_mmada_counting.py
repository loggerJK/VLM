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
# Counting fine-tuning script for MMaDA.
# Based on train_mmada_stage2.py; replaces MMU data with PixMo counting dataset.

import io
import os
import re
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
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer, AutoConfig
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, InitProcessGroupKwargs, set_seed

from training.data import Text2ImageDataset
from training.utils import get_config, flatten_omega_conf, image_transform
from training.imagenet_dataset import ImageNetDataset
from parquet import RefinedWebDataset, CountingDataset

from models import MAGVITv2, get_mask_schedule, MMadaModelLM, MMadaConfig
from training.prompting_utils import UniversalPrompting
from peft import LoraConfig, get_peft_model, PeftModel
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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

def extract_number_fixed(text: str) -> int:
    """Extract a count from model output (Lumina/BAGEL compatible).

    Priority: **N** > English word (zero-ten) > plain digit.
    Returns -1 on failure.
    """
    text = text.lower()
    match = re.search(r"\*\*(\d+)\*\*", text)
    if match:
        return int(match.group(1))
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    }
    for word, num in word_to_num.items():
        if re.search(r"\b" + word + r"\b", text):
            return num
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return -1


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


class TrainStepWrapper(torch.nn.Module):
    """Routes train-step `forward_process` through `forward()` so DDP / DeepSpeed
    reducer bookkeeping (`_post_forward` → `prepare_for_backward`) fires per step
    and gradient all-reduce is scheduled. Calling `inner.forward_process(...)`
    directly bypasses DDP.forward and silently breaks rank gradient sync.

    Other accesses (e.g., `mmu_generate`, `t2i_generate`, `config`,
    `save_pretrained`) fall back to the base via __getattr__, so existing
    `accelerator.unwrap_model(model).<method>(...)` call sites work unchanged.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, **fp_kwargs):
        return self.base.forward_process(**fp_kwargs)

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
    # Dataloaders
    # ------------------------------------------------------------------
    logger.info("Creating dataloaders")

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    total_batch_size_t2i_without_accum = config.training.batch_size_t2i * accelerator.num_processes
    total_batch_size_t2i = (
        config.training.batch_size_t2i
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )

    # T2I data (regularization) — only build the loader when batch_size_t2i > 0
    if config.training.batch_size_t2i > 0:
        if config.dataset.gen_type == "t2i":
            dataset = Text2ImageDataset(
                train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
                tokenizer=None,
                max_seq_length=preproc_config.max_seq_length,
                num_train_examples=config.experiment.max_train_examples_t2i,
                per_gpu_batch_size=config.training.batch_size_t2i,
                global_batch_size=total_batch_size_t2i_without_accum,
                num_workers=dataset_config.num_workers,
                resolution=preproc_config.resolution,
                shuffle_buffer_size=dataset_config.shuffle_buffer_size,
                pin_memory=dataset_config.pin_memory,
                persistent_workers=dataset_config.persistent_workers,
                external_caption_path=dataset_config.external_caption_path,
                external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
                external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
                external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
            )
            train_dataloader_t2i = dataset.train_dataloader
            num_update_steps_per_epoch = math.ceil(
                train_dataloader_t2i.num_batches / config.training.gradient_accumulation_steps
            )
            num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
        elif config.dataset.gen_type == "imagenet1k":
            dataset_imagenet = ImageNetDataset(
                dataset_config.train_t2i_shards_path_or_url,
                image_size=preproc_config.resolution,
            )
            if accelerator.num_processes > 1:
                sampler = DistributedSampler(
                    dataset_imagenet,
                    num_replicas=accelerator.num_processes,
                    rank=accelerator.process_index,
                    shuffle=True,
                )
                shuffle = False
            else:
                sampler = None
                shuffle = True
            train_dataloader_t2i = DataLoader(
                dataset_imagenet,
                batch_size=config.training.batch_size_t2i,
                sampler=sampler,
                collate_fn=dataset_imagenet.collate_fn,
                shuffle=shuffle,
                num_workers=dataset_config.num_workers,
            )
            num_update_steps_per_epoch = math.ceil(len(dataset_imagenet) / total_batch_size_t2i)
            num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)
        else:
            raise ValueError(f"Unsupported gen_type: {config.dataset.gen_type}")
    else:
        train_dataloader_t2i = None
        # CountingDataset is an IterableDataset(repeat=True); collapse epochs to 1.
        num_update_steps_per_epoch = config.training.max_train_steps
        num_train_epochs = 1

    # LM data (regularization) — only build when batch_size_lm > 0
    if config.training.batch_size_lm > 0:
        dataset_lm = RefinedWebDataset(
            data_path=dataset_config.train_lm_shards_path_or_url,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            num_workers=dataset_config.num_workers,
        )
        train_dataloader_lm = DataLoader(
            dataset_lm,
            batch_size=config.training.batch_size_lm,
            sampler=None,
            collate_fn=dataset_lm.collate_fn,
            num_workers=dataset_config.num_workers,
        )
    else:
        train_dataloader_lm = None

    # Counting data (main MMU task)
    assert config.dataset.und_type == "counting", (
        f"This script expects und_type='counting', got '{config.dataset.und_type}'"
    )
    dataset_counting = CountingDataset(
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        tokenizer=uni_prompting.text_tokenizer,
        resolution=preproc_config.resolution,
        count_lower_limit=dataset_config.get("count_lower_limit", 0),
        count_upper_limit=dataset_config.get("count_upper_limit", 20),
        num_workers=dataset_config.num_workers,
    )
    train_dataloader_mmu = DataLoader(
        dataset_counting,
        batch_size=config.training.batch_size_mmu,
        sampler=None,
        collate_fn=dataset_counting.collate_fn,
        num_workers=dataset_config.num_workers,
    )

    iterables = {"mmu_flow": train_dataloader_mmu}
    if train_dataloader_t2i is not None:
        iterables["t2i_flow"] = train_dataloader_t2i
    if train_dataloader_lm is not None:
        iterables["lm_flow"] = train_dataloader_lm
    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

    # ------------------------------------------------------------------
    # Pre-load counting validation data (once)
    # ------------------------------------------------------------------
    val_counting_items = []
    if accelerator.is_main_process:
        logger.info("Pre-loading counting validation data (val_und split)...")
        try:
            from datasets import load_dataset as hf_load_dataset
            val_ds = hf_load_dataset("heez/pixmo-point-count-gen-und", split="val_und")
            max_val = config.experiment.get("max_val_counting_samples", 100)
            n = min(max_val, len(val_ds))
            val_counting_items = [val_ds[i] for i in range(n)]
            logger.info(f"Loaded {len(val_counting_items)} counting validation samples.")
        except Exception as e:
            logger.warning(f"Could not load counting validation data: {e}")

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
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
    model = TrainStepWrapper(model)

    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    vq_model.to(device=accelerator.device)

    # ------------------------------------------------------------------
    # Inner helper functions
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prepare_inputs_and_labels(
        pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
        texts: Union[str, str],
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

    @torch.no_grad()
    def prepare_inputs_and_labels_for_text(texts, max_seq_len, eps=1e-3):
        input_ids_lm, prompt_mask, labels_lm = uni_prompting((texts, max_seq_len), 'lm')
        b, l = input_ids_lm.shape
        t = torch.rand(b, device=input_ids_lm.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)
        masked_indices = torch.rand((b, l), device=input_ids_lm.device) < p_mask
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_lm)
        return noisy_batch, labels_lm, p_mask

    @torch.no_grad()
    def prepare_inputs_and_labels_for_mmu(input_ids_mmu, prompt_masks, labels_mmu, eps=1e-3):
        b, l = input_ids_mmu.shape
        t = torch.rand(b, device=input_ids_mmu.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)
        masked_indices = torch.rand((b, l), device=input_ids_mmu.device) < p_mask
        noisy_batch = torch.where(masked_indices, mask_id, input_ids_mmu)
        noisy_batch[prompt_masks.bool()] = input_ids_mmu[prompt_masks.bool()]
        masked_indices = noisy_batch == mask_id
        prompt_masks = prompt_masks.to(torch.int64)
        answer_lengths = torch.sum((1 - prompt_masks), dim=-1, keepdim=True)
        answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])
        return noisy_batch, labels_mmu, p_mask, answer_lengths

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {config.training.gradient_accumulation_steps}")

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch, batch_idx, dataloader_idx in combined_dataloader:
            data_time_m.update(time.time() - end)

            batch_size_t2i = 0
            batch_size_lm = 0
            t2i_masks = None
            p_mask_lm = None
            mask_prob = None
            input_ids_list = []
            labels_list = []

            # T2I (only when its loader is present)
            if "t2i_flow" in batch:
                batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
                pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["input_ids"]
                pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
                (
                    input_ids_t2i, labels_t2i, mask_prob, image_tokens_ori, t2i_masks
                ) = prepare_inputs_and_labels(pixel_values, texts, config.training.min_masking_rate)
                input_ids_list.append(input_ids_t2i)
                labels_list.append(labels_t2i)
                max_seq_len = input_ids_t2i.shape[-1]
            else:
                max_seq_len = config.dataset.preprocessing.max_seq_length

            # LM (only when its loader is present)
            if "lm_flow" in batch:
                batch_size_lm = len(batch["lm_flow"]["input_ids"])
                texts_lm = batch["lm_flow"]["input_ids"]
                input_ids_lm, labels_lm, p_mask_lm = prepare_inputs_and_labels_for_text(
                    texts_lm, max_seq_len
                )
                input_ids_list.append(input_ids_lm)
                labels_list.append(labels_lm)

            # Counting (MMU) — always present
            pixel_values_mmu = batch["mmu_flow"]["images"].to(accelerator.device, non_blocking=True)
            texts_mmu = batch["mmu_flow"]["input_ids"]
            batch_size_mmu = pixel_values_mmu.shape[0]
            image_tokens_mmu = vq_model.get_code(pixel_values_mmu) + len(uni_prompting.text_tokenizer)
            input_ids_mmu, prompt_masks, labels_mmu = uni_prompting(
                (image_tokens_mmu, texts_mmu), 'mmu'
            )
            (
                input_ids_mmu, labels_mmu, p_mask_mmu, answer_lengths
            ) = prepare_inputs_and_labels_for_mmu(input_ids_mmu, prompt_masks, labels_mmu)
            input_ids_list.append(input_ids_mmu)
            labels_list.append(labels_mmu)

            input_ids = torch.cat([t.to(accelerator.device) for t in input_ids_list], dim=0)
            labels = torch.cat([t.to(accelerator.device) for t in labels_list], dim=0)

            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))

            with accelerator.accumulate(model):
                logits, loss_t2i, loss_lm, loss_mmu = model(
                    input_ids=input_ids,
                    labels=labels,
                    batch_size_t2i=batch_size_t2i,
                    batch_size_lm=batch_size_lm,
                    batch_size_mmu=batch_size_mmu,
                    max_seq_length=config.dataset.preprocessing.max_seq_length,
                    p_mask_lm=p_mask_lm,
                    p_mask_mmu=p_mask_mmu,
                    answer_lengths=answer_lengths,
                    t2i_masks=t2i_masks,
                )

                if batch_size_t2i > 0:
                    avg_loss_t2i = accelerator.gather(
                        loss_t2i.repeat(config.training.batch_size_t2i)
                    ).mean()
                    avg_masking_rate = accelerator.gather(
                        mask_prob.repeat(config.training.batch_size_t2i)
                    ).mean()
                else:
                    avg_loss_t2i = loss_t2i.detach()
                    avg_masking_rate = torch.tensor(0.0, device=accelerator.device)

                if batch_size_lm > 0:
                    avg_loss_lm = accelerator.gather(
                        loss_lm.repeat(config.training.batch_size_lm)
                    ).mean()
                else:
                    avg_loss_lm = loss_lm.detach()

                avg_loss_mmu = accelerator.gather(
                    loss_mmu.repeat(config.training.batch_size_mmu)
                ).mean()

                loss = (
                    config.training.t2i_coeff * loss_t2i
                    + config.training.lm_coeff * loss_lm
                    + config.training.mmu_coeff * loss_mmu
                )

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

                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps
                        * total_batch_size_per_gpu
                        / batch_time_m.val
                    )
                    logs = {
                        "step_loss_t2i": avg_loss_t2i.item(),
                        "step_loss_mmu_counting": avg_loss_mmu.item(),
                        "step_loss_lm": avg_loss_lm.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)
                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
                        f"Loss_counting: {avg_loss_mmu.item():0.4f} "
                        f"Loss_lm: {avg_loss_lm.item():0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f} "
                        f"Mask: {avg_masking_rate.item():0.4f} "
                        f"Samples/sec/gpu: {samples_per_second_per_gpu:0.2f} "
                        f"Data: {data_time_m.val:0.3f}s "
                        f"Batch: {batch_time_m.val:0.3f}s"
                    )
                    batch_time_m.reset()
                    data_time_m.reset()

                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1, uni_prompting)

                need_eval = (global_step + 1) % config.experiment.eval_every == 0
                need_gen = (
                    config.training.batch_size_t2i > 0
                    and (
                        (global_step + 1) % config.experiment.generate_every == 0
                        or global_step == 0
                    )
                )

                if accelerator.is_main_process:
                    if need_eval and val_counting_items:
                        validate_counting(
                            model,
                            vq_model,
                            uni_prompting,
                            accelerator,
                            config,
                            global_step + 1,
                            val_counting_items,
                        )
                    if need_gen:
                        generate_images(
                            model, vq_model, uni_prompting, accelerator, config,
                            global_step + 1, mask_schedule=mask_schedule, force_no_cfg=False,
                        )
                        generate_images(
                            model, vq_model, uni_prompting, accelerator, config,
                            global_step + 1, mask_schedule=mask_schedule, force_no_cfg=True,
                        )

                accelerator.wait_for_everyone()

                global_step += 1

            if global_step >= config.training.max_train_steps:
                break

    accelerator.wait_for_everyone()
    save_checkpoint(model, config, accelerator, global_step, uni_prompting)

    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=True)
        if use_lora:
            merged_dir = os.path.join(config.experiment.output_dir, "final_merged")
            model.merge_and_unload().save_pretrained(merged_dir, safe_serialization=True)
            uni_prompting.text_tokenizer.save_pretrained(merged_dir)

    accelerator.end_training()


# ---------------------------------------------------------------------------
# Counting validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_counting(
    model,
    vq_model,
    uni_prompting,
    accelerator,
    config,
    global_step,
    val_items,
):
    """Run counting validation and log accuracy / MAD / sample predictions to WandB."""
    from parquet.my_dataset import image_transform_squash

    logger.info("Running counting validation...")
    model.eval()

    resolution = config.dataset.preprocessing.resolution
    device = accelerator.device

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    gt_counts, pred_counts, pred_answers, questions, sample_images = [], [], [], [], []

    total_samples = len(val_items)
    print("=" * 70)
    print(f"Counting validation @ step {global_step}  ({total_samples} samples)")
    print("=" * 70)

    for item in val_items:
        try:
            image = item['image']
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert('RGB')
            else:
                image = image.convert('RGB')

            question = item.get('question', item.get('question_count', ''))
            gt_count = int(item.get('count', -1))
            if gt_count < 0:
                continue

            pil_image = image.copy()

            image_tensor = image_transform_squash(
                {'images': image}, resolution=resolution
            )['images'].unsqueeze(0).to(device)

            image_tokens = vq_model.get_code(image_tensor) + len(uni_prompting.text_tokenizer)

            # Match inference_mmu.py: apply_chat_template directly after <|eoi|>
            text_token_ids = uni_prompting.text_tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            input_ids = torch.cat([
                torch.full((1, 1), int(uni_prompting.sptids_dict['<|mmu|>']), dtype=torch.long, device=device),
                torch.full((1, 1), int(uni_prompting.sptids_dict['<|soi|>']), dtype=torch.long, device=device),
                image_tokens,
                torch.full((1, 1), int(uni_prompting.sptids_dict['<|eoi|>']), dtype=torch.long, device=device),
                text_token_ids,
            ], dim=1).long()

            prompt_len = input_ids.shape[1]

            with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
                output_ids = accelerator.unwrap_model(model).mmu_generate(
                    input_ids,
                    max_new_tokens=64,
                    steps=64,
                    block_length=64,
                    temperature=0.0,
                    remasking='low_confidence',
                )

            new_tok_ids = output_ids[0, prompt_len:]
            generated_text = uni_prompting.text_tokenizer.decode(
                new_tok_ids, skip_special_tokens=True
            )
            generated_text_raw = uni_prompting.text_tokenizer.decode(
                new_tok_ids, skip_special_tokens=False
            )

            pred_count = extract_number_fixed(generated_text)

            gt_counts.append(gt_count)
            pred_counts.append(pred_count)
            pred_answers.append(generated_text)
            questions.append(question)
            sample_images.append(pil_image)

            if len(gt_counts) == 1:
                prompt_preview = uni_prompting.text_tokenizer.decode(
                    input_ids[0], skip_special_tokens=False
                )
                print("--- Prompt preview (first sample, last 400 chars) ---")
                print(prompt_preview[-400:])
                print("--- end preview ---")

            mark = "✓" if pred_count == gt_count else ("?" if pred_count < 0 else "✗")
            text_preview = generated_text.strip()
            if len(text_preview) > 60:
                text_preview = text_preview[:60] + "..."
            idx = len(gt_counts)
            print(
                f"[{idx:3d}/{total_samples}] gt={gt_count:>2}  "
                f"pred={pred_count:>2}  {mark}  {text_preview!r}"
            )
            n_new = int(new_tok_ids.numel())
            eos_id = uni_prompting.text_tokenizer.eos_token_id
            n_eos = int((new_tok_ids == eos_id).sum().item()) if eos_id is not None else 0
            raw_preview = generated_text_raw if len(generated_text_raw) <= 120 \
                else generated_text_raw[:120] + "..."
            print(f"           raw[{n_new}t, {n_eos} EOS]: {raw_preview!r}")

        except Exception as e:
            logger.warning(f"Counting validation error: {e}")
            continue

    if not gt_counts:
        model.train()
        return

    total = len(gt_counts)
    correct = sum(1 for g, p in zip(gt_counts, pred_counts) if g == p and p >= 0)
    accuracy = correct / total

    # Lumina parity: include parse failures (-1) in MAD so misses are penalised
    deviations = [abs(g - p) for g, p in zip(gt_counts, pred_counts)]
    mad = sum(deviations) / len(deviations) if deviations else 0.0

    # Confusion matrix heatmap (0–20 fixed range)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    fixed_labels = list(range(0, 21))
    cm = sk_confusion_matrix(gt_counts, pred_counts, labels=fixed_labels)
    fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=fixed_labels, yticklabels=fixed_labels,
                cmap='viridis', ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Ground Truth')
    ax_cm.set_title(f'Counting CM (step {global_step})  Acc: {accuracy:.4f}  MAD: {mad:.4f}')
    cm_image = wandb.Image(fig_cm)
    plt.close(fig_cm)

    # Prediction table — prefer mismatches first, cap at 10 samples (image included)
    paired = list(zip(questions, gt_counts, pred_counts, pred_answers, sample_images))
    mismatches = [p for p in paired if p[1] != p[2]]
    matches = [p for p in paired if p[1] == p[2]]
    table_rows = (mismatches + matches)[:10]

    pred_table = wandb.Table(columns=["image", "question", "gt_count", "pred_count", "pred_answer"])
    for q, g, p, a, img in table_rows:
        pred_table.add_data(wandb.Image(img), q, g, p, a)

    wandb.log({
        "counting/val_accuracy": accuracy,
        "counting/val_mad": mad,
        "counting/confusion_matrix": cm_image,
        "counting/predictions": pred_table,
    }, step=global_step)

    parse_fails = sum(1 for p in pred_counts if p < 0)
    mismatches_n = total - correct
    print("-" * 70)
    print(
        f"Acc={accuracy:.4f} ({correct}/{total})  "
        f"MAD={mad:.4f}  Mismatches={mismatches_n}  ParseFails={parse_fails}"
    )
    print("=" * 70)

    logger.info(
        f"Counting validation step {global_step}: "
        f"Accuracy={accuracy:.4f} ({correct}/{total}), MAD={mad:.4f}"
    )

    model.train()


# ---------------------------------------------------------------------------
# T2I image generation (unchanged from stage2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_images(
    model, vq_model, uni_prompting, accelerator, config, global_step,
    mask_schedule, force_no_cfg=False,
):
    logger.info("Generating images...")
    model.eval()

    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()

    mask_token_id = accelerator.unwrap_model(model).config.mask_token_id
    image_tokens = torch.ones(
        (len(validation_prompts), config.model.mmada.num_vq_tokens),
        dtype=torch.long, device=accelerator.device,
    ) * mask_token_id
    input_ids, attention_mask = uni_prompting((validation_prompts, image_tokens), 't2i_gen')

    if not force_no_cfg and config.training.guidance_scale > 0:
        uncond_input_ids, uncond_attention_mask = uni_prompting(
            ([''] * len(validation_prompts), image_tokens), 't2i_gen'
        )
        cfg_scale = config.training.guidance_scale
    else:
        uncond_input_ids = None
        uncond_attention_mask = None
        cfg_scale = 0

    weight_dtype = (
        torch.float16 if accelerator.mixed_precision == "fp16"
        else torch.bfloat16 if accelerator.mixed_precision == "bf16"
        else torch.float32
    )

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
            seq_len=config.model.mmada.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )

    gen_token_ids = torch.clamp(
        gen_token_ids,
        max=accelerator.unwrap_model(model).config.codebook_size - 1,
        min=0,
    )
    images = vq_model.decode_code(gen_token_ids)
    model.train()

    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(img) for img in images]

    wandb_images = [
        wandb.Image(img, caption=validation_prompts[i]) for i, img in enumerate(pil_images)
    ]
    wandb.log({f"Generated images with cfg {cfg_scale}": wandb_images}, step=global_step)


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
        if isinstance(unwrapped_model, TrainStepWrapper):
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
