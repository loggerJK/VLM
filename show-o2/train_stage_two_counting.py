# coding=utf-8
# Copyright 2025 NUS Show Lab, HuggingFace.
#
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

import os
import json
import logging
import math
import shutil
import time
import re
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from tqdm import tqdm
from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.lr_schedulers import get_scheduler
from models.my_logging import set_verbosity_info, set_verbosity_error
from models.misc import prepare_gen_input, get_text_tokenizer, get_weight_type
from torch.nn.attention.flex_attention import flex_attention
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers.trainer_pt_utils import get_model_param_count

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)

from datasets_ import CountingDataset
from utils import get_config, flatten_omega_conf, AverageMeter, denorm, denorm_vid, get_hyper_params, \
    path_to_llm_name, _freeze_params

from transport import Sampler, create_transport

logger = get_logger(__name__, log_level="INFO")

def extract_number(text):
    """모델 출력에서 **숫자** 패턴 추출"""
    match = re.search(r"\*\*(\d+)\*\*", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return -1

@torch.no_grad()
def evaluate(model, dataset, vae_model, accelerator, weight_type, config, text_tokenizer):
    logger.info("Running validation (Accuracy using mmu_generate with sharding)...")
    model.eval()
    
    # Validation limit
    eval_limit = config.experiment.get("eval_limit", 100)
    total_samples = min(len(dataset), eval_limit)
    
    # Shard indices across processes for parallel validation
    all_indices = list(range(total_samples))
    local_indices = all_indices[accelerator.process_index::accelerator.num_processes]
    
    correct = 0
    total = 0

    unwrapped_model = accelerator.unwrap_model(model)
    device = accelerator.device

    for idx in tqdm(local_indices, disable=not accelerator.is_main_process, desc="Validating"):
        item = dataset[idx]
        
        # Prepare inputs (add batch dim)
        text_tokens = item['text_tokens'].unsqueeze(0).to(device)
        pixel_values = item['images'].unsqueeze(0).to(device).to(weight_type)
        modality_positions = item['modality_positions'].unsqueeze(0).to(device)
        text_labels = item['text_labels'].unsqueeze(0).to(device)

        # Encode images to latents
        if config.model.vae_model.type == 'wan21':
            if len(pixel_values.shape) == 4:
                pixel_values = pixel_values.unsqueeze(2)
            image_latents = vae_model.sample(pixel_values)
            if pixel_values.shape[2] == 1:
                image_latents = image_latents.squeeze(2)
        
        # Generation expects BS=1
        labels = text_labels[0]
        input_ids = text_tokens[0]
        
        # Find start of answer
        answer_start_indices = (labels != -100).nonzero(as_tuple=True)[0]
        if len(answer_start_indices) == 0:
            continue
            
        prompt_end_idx = answer_start_indices[0].item()
        prompt_ids = input_ids[:prompt_end_idx] # (L,)
        
        # Ground truth
        gt_ids = labels[prompt_end_idx:]
        gt_ids = gt_ids[gt_ids != -100]
        gt_text = text_tokenizer.decode(gt_ids, skip_special_tokens=True)
        gt_count = extract_number(gt_text)
        
        if gt_count == -1: continue

        # 1. Image embeds
        img_lat = image_latents # (1, C, H, W)
        img_und = unwrapped_model.image_embedder_und(img_lat.to(weight_type))
        img_gen = unwrapped_model.image_embedder_gen(img_lat.to(weight_type))
        img_und = img_und + unwrapped_model.position_embedding(unwrapped_model.image_position_ids)
        img_und = unwrapped_model.und_trans(img_und)['last_hidden_state']
        image_embeds = unwrapped_model.fusion_proj(torch.cat([img_und, img_gen], dim=-1)) # (1, L, D)

        # 2. Time embeds
        t_val = torch.ones(1, device=device, dtype=weight_type)
        time_embeds = unwrapped_model.time_embed(t_val, weight_type)
        if hasattr(unwrapped_model, 'time_embed_proj'):
            time_embeds = unwrapped_model.time_embed_proj(time_embeds)
        
        # 3. Construct combined embeds
        init_input_embeds = unwrapped_model.showo.model.embed_tokens(prompt_ids.unsqueeze(0))
        
        # Use modality_positions to place image/time embeds
        offset, length = modality_positions[0, 0]
        if unwrapped_model.config.add_time_embeds:
            init_input_embeds[0, offset] = time_embeds[0]
            init_input_embeds[0, offset+1 : offset+1+length-1] = image_embeds[0, :max(length-1, 0)]
        else:
            init_input_embeds[0, offset : offset+length] = image_embeds[0, :length]

        # 4. Prepare initial attention mask
        attention_mask = omni_attn_mask_naive(
            B=1,
            LEN=init_input_embeds.size(1),
            modalities=modality_positions[0:1],
            device=device, inverted=True
        ).to(weight_type)

        # 5. Use model.mmu_generate
        output_tokens = unwrapped_model.mmu_generate(
            input_embeds=init_input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=20, # Enough for a number
            temperature=0.01,
            top_k=1,
            eos_token=text_tokenizer.eos_token_id
        )
        
        # output_tokens is a list of scalars
        full_answer = text_tokenizer.decode(output_tokens, skip_special_tokens=True)
        pred_count = extract_number(full_answer)
        
        print(f"""=======================
              [DEBUG] Full answer: {full_answer}
              pred count: {pred_count}
              GT count: {gt_count}
              =======================""")    
        if pred_count == gt_count:
            correct += 1
        total += 1

    # Aggregate results from all processes
    metrics = torch.tensor([correct, total], device=device, dtype=torch.float32)
    metrics = accelerator.reduce(metrics, reduction="sum")
    final_correct, final_total = metrics[0].item(), metrics[1].item()

    acc = final_correct / final_total if final_total > 0 else 0.0
    model.train()
    return {"val/accuracy": acc}


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()
    
    run_name = os.getenv("WANDB_NAME", config.experiment.name)
    config.experiment.name = run_name
    config.experiment.output_dir = os.path.join('./checkpoints', run_name)

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    bs_mmu = config.training.batch_size_mmu
    total_batch_size_per_gpu = bs_mmu 
    total_batch_size = total_batch_size_per_gpu * accelerator.num_processes * config.training.gradient_accumulation_steps

    # if "concat" in config.dataset.mixed_loader_mode:
    #     assert config.dataset.accumulation == 1, "No need to enable accumulation in mixed-dataloader!"
    #     # total_batch_size_per_gpu = bs_t2i + bs_mmu 
    #     total_batch_size_per_gpu = bs_mmu
    #     total_batch_size_without_accum = total_batch_size_per_gpu * accelerator.num_processes
    #     total_batch_size = total_batch_size_without_accum * config.training.gradient_accumulation_steps
    # else:
    #     assert bs_t2i == bs_mmu, "We should ensure batch size is consistent at each iteration if we use FlexAttention!"
    #     total_batch_size_per_gpu = bs_t2i * config.dataset.accumulation
    #     total_batch_size_without_accum = total_batch_size_per_gpu * accelerator.num_processes
    #     total_batch_size = total_batch_size_without_accum * config.training.gradient_accumulation_steps

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
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
            name=os.getenv("WANDB_NAME", config.experiment.name),
            id=os.getenv("WANDB_RUN_ID", run_id),
            resume=resume_wandb_run,
            entity=os.getenv("WANDB_ENTITY", config.wandb.get("entity", None)),
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
        OmegaConf.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    weight_type = get_weight_type(config)
    print(f"[INFO] Using weight type: {weight_type}")

    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type,
                           device=accelerator.device)
    else:
        raise NotImplementedError

    text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path, add_showo_tokens=True,
                                                         return_showo_token_ids=True,
                                                         llm_name=path_to_llm_name[config.model.showo.llm_model_path])
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(config.model.showo.pretrained_model_path, use_safetensors=False).to(accelerator.device).to(weight_type)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(accelerator.device).to(weight_type)

    _freeze_params(model, config.model.showo.frozen_params)

    if config.model.get("gradient_checkpointing", False):
        print("[INFO] Enable gradient checkpointing")
        model.enable_gradient_checkpointing()

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # for time embedding
    if config.model.showo.add_time_embeds:
        # we prepend the time embedding to vision tokens
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_hq_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
        config.dataset.preprocessing.num_mixed_modal_tokens += 1

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params
    # optimizer_grouped_parameters = [
    #     {"params": [p for n, p in model.named_parameters() if (('und_trans' in n or 'image_embedder' in n or 'position_embedding' in n) and p.requires_grad)], "lr": optimizer_config.learning_rate_ve},
    #     {"params": [p for n, p in model.named_parameters() if ('fusion_proj' in n and p.requires_grad)], "lr": optimizer_config.learning_rate_proj},
    #     {"params": [p for n, p in model.named_parameters() if (('showo' in n or 'diffusion' in n or 'diff_proj' in n or 'time_embed_proj' in n) and p.requires_grad)], "lr": optimizer_config.learning_rate_showo},
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, betas=(optimizer_config.beta1, optimizer_config.beta2), weight_decay=optimizer_config.weight_decay, eps=optimizer_config.epsilon)
    import bitsandbytes as bnb
    optimizer = bnb.optim.Adam8bit( filter(lambda p: p.requires_grad, model.parameters()), betas=(optimizer_config.beta1, optimizer_config.beta2), weight_decay=optimizer_config.weight_decay, eps=optimizer_config.epsilon, lr=optimizer_config.learning_rate_overall)
    
    
    # config.experiment.output_dir에 optimize되는 파라미터 이름들 저장 & wandb에 기록
    if accelerator.is_main_process:
        optimized_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        with open(os.path.join(config.experiment.output_dir, "optimized_param_names.txt"), "w") as f:
            for name in optimized_param_names:
                f.write(f"{name}\n")
        wandb.config.update({"optimized_param_names": optimized_param_names})    

    ##################################
    #         DATALOADER             #
    #################################
    def create_dataloader(dataset, batch_size, collate_fn, shuffle=True):
        sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=shuffle, drop_last=True) if accelerator.num_processes > 1 else None
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, shuffle=(sampler is None and shuffle), num_workers=dataset_config.num_workers, drop_last=True)

    dataset_mmu = CountingDataset(
        dataset_config.train_mmu_shards_path_or_url,
        text_tokenizer=text_tokenizer,
        image_size=preproc_config.resolution,
        max_seq_len=preproc_config.max_seq_length,
        num_image_tokens=preproc_config.num_mmu_image_tokens,
        latent_width=preproc_config.latent_width,
        latent_height=preproc_config.latent_height,
        cond_dropout_prob=config.training.cond_dropout_prob,
        stage=config.training.stage,
        showo_token_ids=showo_token_ids,
        split='train'
    )
    train_dataloader_mmu = create_dataloader(dataset_mmu, config.training.batch_size_mmu, dataset_mmu.collate_fn, shuffle=True)

    # Validation
    dataset_val = CountingDataset(
        dataset_config.train_mmu_shards_path_or_url,
        text_tokenizer=text_tokenizer,
        image_size=preproc_config.resolution,
        max_seq_len=preproc_config.max_seq_length,
        num_image_tokens=preproc_config.num_mmu_image_tokens,
        latent_width=preproc_config.latent_width,
        latent_height=preproc_config.latent_height,
        cond_dropout_prob=0.0,
        stage=config.training.stage,
        showo_token_ids=showo_token_ids,
        split='validation'
    )
    val_dataloader = create_dataloader(dataset_val, config.training.batch_size_mmu, dataset_val.collate_fn, shuffle=False)

    num_update_steps_per_epoch = math.ceil(len(dataset_mmu) / total_batch_size)
    if config.training.get("num_epochs", None) is not None:
        config.training.max_train_steps = config.training.num_epochs * num_update_steps_per_epoch
        num_train_epochs = config.training.num_epochs
    else:
        config.training.max_train_steps = len(dataset_mmu) // (config.training.batch_size_mmu * accelerator.num_processes) * config.training.gradient_accumulation_steps
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(config.lr_scheduler.scheduler, optimizer=optimizer, num_training_steps=config.training.max_train_steps, num_warmup_steps=int(config.training.max_train_steps * config.lr_scheduler.params.warmup_ratio))
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    global_step = 0
    global_grad_step = 0
    first_epoch = 0
    if config.experiment.resume_from_checkpoint:
        dirs = [d for d in os.listdir(config.experiment.output_dir) if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = os.path.join(config.experiment.output_dir, dirs[-1]) if dirs else None
        if path:
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            model.load_state_dict(torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu"))

    logger.info("***** Running training *****")
    logger.info(f"  Num epochs = {num_train_epochs}")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Number of model parameters = {get_model_param_count(model)}")
    logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True)}")
    logger.info(f"  Percentage of trainable parameters = {get_model_param_count(model, trainable_only=True) / get_model_param_count(model) * 100:.2f}%")

    logger.info(f"***** Running training: Epochs={num_train_epochs}, Steps={config.training.max_train_steps} *****")

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        loss_meter = AverageMeter()

        if epoch == first_epoch and global_step > 0:
            progress_bar.update(global_step % num_update_steps_per_epoch)

        for batch in train_dataloader_mmu:
            with accelerator.accumulate(model):
                text_tokens = batch['text_tokens'].to(accelerator.device)
                text_labels = batch['text_labels'].to(accelerator.device)
                pixel_values = batch['images'].to(accelerator.device).to(weight_type)
                modality_positions = batch['modality_positions'].to(accelerator.device)
                
                if config.model.vae_model.type == 'wan21':
                    if len(pixel_values.shape) == 4: pixel_values = pixel_values.unsqueeze(2)
                    image_latents = vae_model.sample(pixel_values)
                    if pixel_values.shape[2] == 1: image_latents = image_latents.squeeze(2)

                block_mask = omni_attn_mask_naive(text_tokens.size(0), text_tokens.size(1), modality_positions, accelerator.device).to(weight_type)
                t = torch.ones(image_latents.shape[0], device=accelerator.device, dtype=weight_type)

                logits, loss_ntp = model.forward_und_only(text_tokens=text_tokens, image_latents=image_latents, t=t, attention_mask=block_mask, text_masks=batch['text_masks'].to(accelerator.device), image_masks=batch['image_masks'].to(accelerator.device), text_labels=text_labels, modality_positions=modality_positions, device=accelerator.device)

                accelerator.backward(loss_ntp)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Accumulate loss for logging (across micro-batches)
            loss_meter.update(loss_ntp.item(), text_tokens.size(0))

            if accelerator.sync_gradients:
                
                if global_step % config.experiment.log_every == 0:
                    # Use the average loss accumulated over the gradient accumulation steps
                    avg_loss = loss_meter.avg
                    accelerator.log({"train_loss": avg_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    loss_meter.reset()

                if global_step % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step, epoch)

                if (global_step) % config.experiment.eval_every == 0:
                    val_metrics = evaluate(model, dataset_val, vae_model, accelerator, weight_type, config, text_tokenizer)
                    accelerator.log(val_metrics, step=global_step)
                    logger.info(f"Step: {global_step}, Val Acc: {val_metrics['val/accuracy']:.4f}")
                
                progress_bar.update(1)
                global_step += 1

            # if global_step >= config.training.max_train_steps: 
            #     break
        
        progress_bar.close()

    save_checkpoint(model, config, accelerator, "final")


    save_checkpoint(model, config, accelerator, "final")
    if accelerator.is_main_process: accelerator.unwrap_model(model).save_pretrained(config.experiment.output_dir, safe_serialization=False)
    accelerator.end_training()

def save_checkpoint(model, config, accelerator, global_step, epoch=None):
    save_path = Path(config.experiment.output_dir) / f"checkpoint-{global_step}-epoch{epoch if epoch is not None else 'NA'}"
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_path / "unwrapped_model", save_function=accelerator.save, state_dict=accelerator.get_state_dict(model), safe_serialization=False)

if __name__ == "__main__":
    main()
