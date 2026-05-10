import os

# Must be set before torch import for deterministic cuBLAS kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import json
import argparse
import math
import random

import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from models import MAGVITv2, get_mask_schedule, MMadaModelLM
from training.prompting_utils import UniversalPrompting


def configure_deterministic_cuda():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    raise ValueError(f"model_type {model_type} not supported.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt_files", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--generation_timesteps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_path", default=None)
    return parser.parse_args()


def disable_peft_tp_sharding():
    import peft.utils.save_and_load as peft_save_and_load

    # torchrun/init_process_group 상태에서 peft 0.19.1이
    # transformers.integrations.tensor_parallel 을 import하다가 죽는 경로 차단
    peft_save_and_load._maybe_shard_state_dict_for_tp = lambda *args, **kwargs: None

def load_model(args, config, device):
    base_path = config.model.mmada.pretrained_model_path
    model = MMadaModelLM.from_pretrained(
        base_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    local_files_only=True,
    )
    model=model.to(device)

    if args.checkpoint_path:
        from peft import PeftModel

        lora_ckpt_path = os.path.join(args.checkpoint_path, "unwrapped_model")
        print(f"Loading adapter from checkpoint: {lora_ckpt_path}")

        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_ckpt_path,
        #     is_trainable=False,
        #     local_files_only=True,
        # )
        disable_peft_tp_sharding()
        model.load_adapter(lora_ckpt_path,  device_map={"": "cpu"},)
        model = model.to(device).eval()

        # model = model.merge_and_unload()

    model = model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    configure_deterministic_cuda()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    config = OmegaConf.load(args.config)

    if args.guidance_scale is not None:
        config.training.guidance_scale = args.guidance_scale
    if args.generation_timesteps is not None:
        config.training.generation_timesteps = args.generation_timesteps

    tokenizer = AutoTokenizer.from_pretrained(config.model.mmada.pretrained_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
        use_reserved_token=True,
    )

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = load_model(args, config, device)
    mask_token_id = model.config.mask_token_id

    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        sched_params = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_schedule(schedule, **sched_params)
    else:
        mask_schedule = get_mask_schedule(config.training.get("mask_schedule", "cosine"))

    with open(args.prompt_files) as f:
        all_prompts = [json.loads(line) for line in f if line.strip()]

    my_prompt_items = [(idx, item) for idx, item in enumerate(all_prompts) if idx % world_size == rank]
    total_samples = len(my_prompt_items) * args.num_samples

    with tqdm(
        total=total_samples,
        desc=f"Rank {rank}/{world_size}",
        position=rank,
        dynamic_ncols=True,
        leave=True,
        unit="sample",
    ) as progress:
        for prompt_idx, metadata in my_prompt_items:
            prompt = metadata.get("prompt", metadata.get("user_prompt", ""))
            prompt_dir = os.path.join(args.output_dir, f"{prompt_idx:05d}")
            samples_dir = os.path.join(prompt_dir, "samples")

            for sample_idx in range(args.num_samples):
                sample_path = os.path.join(samples_dir, f"{sample_idx:04d}.png")
                progress.set_postfix_str(f"prompt={prompt_idx:05d} sample={sample_idx:04d}", refresh=False)
                if os.path.exists(sample_path):
                    progress.update(1)
                    continue

                current_seed = args.seed + prompt_idx * args.num_samples + sample_idx
                random.seed(current_seed)
                np.random.seed(current_seed)
                torch.manual_seed(current_seed)
                torch.cuda.manual_seed_all(current_seed)
                generator = torch.Generator(device=device).manual_seed(current_seed)

                image_tokens = torch.ones((1, config.model.mmada.num_vq_tokens), dtype=torch.long, device=device) * mask_token_id
                input_ids, attention_mask = uni_prompting(([prompt], image_tokens), 't2i_gen')

                if config.training.guidance_scale > 0:
                    uncond_input_ids, uncond_attention_mask = uni_prompting(([''], image_tokens), 't2i_gen')
                else:
                    uncond_input_ids = uncond_attention_mask = None

                with torch.no_grad():
                    gen_token_ids = model.t2i_generate(
                        input_ids=input_ids,
                        uncond_input_ids=uncond_input_ids,
                        attention_mask=attention_mask,
                        uncond_attention_mask=uncond_attention_mask,
                        guidance_scale=config.training.guidance_scale,
                        temperature=args.temperature,
                        timesteps=config.training.generation_timesteps,
                        noise_schedule=mask_schedule,
                        noise_type=config.training.get("noise_type", "mask"),
                        seq_len=config.model.mmada.num_vq_tokens,
                        # CRITICAL: t2i_generate's `resolution` arg is actually the CFG
                        # text-prefix length; default 512 corrupts CFG when our config
                        # uses max_seq_length=128 (rel_position).
                        resolution=config.dataset.preprocessing.max_seq_length,
                        uni_prompting=uni_prompting,
                        config=config,
                        generator=generator,
                    )

                gen_token_ids = torch.clamp(gen_token_ids, max=config.model.mmada.codebook_size - 1, min=0)
                images = vq_model.decode_code(gen_token_ids)
                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images *= 255.0
                images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                pil_image = Image.fromarray(images[0])

                os.makedirs(samples_dir, exist_ok=True)
                pil_image.save(sample_path)
                progress.update(1)

            meta_path = os.path.join(prompt_dir, "metadata.jsonl")
            if not os.path.exists(meta_path):
                os.makedirs(prompt_dir, exist_ok=True)
                with open(meta_path, "w") as f:
                    f.write(json.dumps(metadata) + "\n")

            existing_samples = [
                os.path.join(samples_dir, f"{si:04d}.png")
                for si in range(args.num_samples)
                if os.path.exists(os.path.join(samples_dir, f"{si:04d}.png"))
            ]
            if existing_samples:
                imgs = [Image.open(p) for p in existing_samples]
                w, h = imgs[0].size
                ncols = math.ceil(math.sqrt(len(imgs)))
                nrows = math.ceil(len(imgs) / ncols)
                grid = Image.new("RGB", (ncols * w, nrows * h))
                for i, img in enumerate(imgs):
                    grid.paste(img, ((i % ncols) * w, (i // ncols) * h))
                grid.save(os.path.join(prompt_dir, "grid.png"))

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
