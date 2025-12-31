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
    # logger.info("Running validation (Accuracy using mmu_generate with sharding)...")
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

if __name__ == "__main__":
    
    accelerator = Accelerator()

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
    
    #########################
    # MODELS and OPTIMIZER  #
    #########################
    weight_type = torch.bfloat16
    print(f"[INFO] Using weight type: {weight_type}")

    VAE_PATH = 'Wan2.1_VAE.pth'
    from models import WanVAE
    vae_model = WanVAE(vae_pth=VAE_PATH, dtype=weight_type,
                        device=accelerator.device)
    
    LLM_MODEL_PATH = 'Qwen/Qwen2.5-7B-Instruct'
    text_tokenizer, showo_token_ids = get_text_tokenizer(LLM_MODEL_PATH, add_showo_tokens=True,
                                                         return_showo_token_ids=True,
                                                         llm_name=path_to_llm_name[LLM_MODEL_PATH])
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    SHOWO_MODEL_PATH = 'showlab/show-o2-7B'
    model = Showo2Qwen2_5.from_pretrained(SHOWO_MODEL_PATH, use_safetensors=False).to(accelerator.device).to(weight_type)
    model.eval()
    model = accelerator.prepare(model)
    
    val_metrics = evaluate(model, dataset_val, vae_model, accelerator, weight_type, config, text_tokenizer)
    
    print(f"Validation Metrics for {SHOWO_MODEL_PATH} :  {val_metrics}")
    
    # val_metrics 파일에 기록
    
    with open('evaluation_count_results.txt', 'a') as f:
        f.write(f"Model: {SHOWO_MODEL_PATH}, Validation Metrics: {val_metrics}\n")