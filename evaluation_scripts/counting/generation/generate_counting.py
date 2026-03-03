import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from transformers import CLIPImageProcessor
from blip3o.constants import *
from blip3o.conversation import conv_templates
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
import random
import warnings
from peft import PeftConfig
from safetensors.torch import load_file

# Add project root to path for pipeline_llava_gen import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from pipeline_llava_gen import EmuVisualGenerationPipeline


def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_template(prompt):
    conv = conv_templates['qwen'].copy()
    conv.append_message(conv.roles[0], prompt[0])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return [prompt]


torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--base_model", type=str, default=None, help="Base model path (required for LoRA checkpoints)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated images")
    parser.add_argument("--metadata_file", type=str, required=True, help="Path to evaluation_metadata_count.jsonl")
    parser.add_argument("--prompt_template", type=str, default="qwen", help="Template format")
    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples per prompt")
    parser.add_argument("--steps", type=int, default=50, help="Number of ddim sampling steps")
    parser.add_argument("--scale", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--skip_grid", action="store_true", help="Skip saving grid")
    parser.add_argument("--lora_mode", type=str, default="und", help="LoRA mode (e.g., 'und' or 'gen')")
    opt = parser.parse_args()
    return opt


def main(opt):
    # Distributed setup
    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(days=1))
        device = f"cuda:{local_rank}"
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = "cuda"

    model_name = opt.model
    is_lora = os.path.exists(os.path.join(model_name, "adapter_config.json"))

    os.makedirs(opt.output_dir, exist_ok=True)
    prompt_template = opt.prompt_template
    disable_torch_init()

    if is_lora:
        assert opt.base_model is not None, "--base_model is required for LoRA checkpoints"
        print(f"[Rank {rank}] LoRA checkpoint detected. Loading base model from {opt.base_model}")
        tokenizer, multi_model, context_len = load_pretrained_model(opt.base_model, device=device)
        if opt.lora_mode == "und":
            peft_model_id = model_name
            adapter_weight_path = os.path.join(peft_model_id, 'adapter_model.safetensors')
            peft_config = PeftConfig.from_pretrained(
              peft_model_id,
            )   
            ckpt = load_file(adapter_weight_path)   
            new_ckpt = {}
            for k in ckpt.keys():
                new_k = k.replace('base_model.model.model.language_model', 'model')
                new_ckpt[new_k] = ckpt[k]
            print("="*50)
            print(f"Loading adapter weights from {adapter_weight_path}")
            print("="*50)
            
            multi_model.load_adapter(
                peft_config=peft_config,
                adapter_state_dict=new_ckpt,
            )
        else:
            print("="*50)
            print(f"Loading adapter weights from {adapter_weight_path}")
            print("="*50)
            multi_model.load_adapter(model_name)
        # Restore latent_queries from gen_components.pt
        gen_ckpt = os.path.join(model_name, "gen_components.pt")
        if os.path.exists(gen_ckpt):
            gen_state = torch.load(gen_ckpt, map_location="cpu")
            inner = multi_model.get_model()
            if "latent_queries" in gen_state:
                inner.latent_queries.data.copy_(gen_state["latent_queries"])
                print(f"[Rank {rank}] Restored latent_queries from {gen_ckpt}")
        diffusion_path = opt.base_model + "/diffusion-decoder"
    else:
        tokenizer, multi_model, context_len = load_pretrained_model(model_name, device=device)
        diffusion_path = model_name + "/diffusion-decoder"

    multi_model.to(device)

    scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_path, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(diffusion_path, subfolder="unet", torch_dtype=torch.bfloat16, variant="bf16")
    vae = AutoencoderKL.from_pretrained(diffusion_path, subfolder="vae", torch_dtype=torch.bfloat16, variant="bf16")
    feature_extractor = CLIPImageProcessor.from_pretrained(diffusion_path, subfolder="feature_extractor")

    pipe = EmuVisualGenerationPipeline(
        tokenizer=tokenizer,
        multimodal_encoder=multi_model,
        scheduler=scheduler,
        unet=unet,
        vae=vae,
        feature_extractor=feature_extractor,
        safety_checker=None,
    )

    pipe.vae.to(device)
    pipe.unet.to(device)

    # Load all prompts
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    # Assign global indices before sharding
    for i, m in enumerate(metadatas):
        m['_global_index'] = i

    # Shard data across ranks (interleaved)
    metadatas = metadatas[rank::world_size]
    print(f"[Rank {rank}] Processing {len(metadatas)} samples (world_size={world_size})")

    skipped = 0
    for local_idx, metadata in tqdm(enumerate(metadatas), total=len(metadatas), desc=f"Rank {rank}"):
        set_global_seed(seed=opt.seed)
        global_index = metadata['_global_index']
        outpath = os.path.join(opt.output_dir, f"{global_index:05d}")

        # Resume logic: skip if enough samples already exist
        sample_path = os.path.join(outpath, "samples")
        existing = [f for f in os.listdir(sample_path) if f.endswith('.png')] if os.path.isdir(sample_path) else []
        if len(existing) >= opt.n_samples:
            print(f"[Rank {rank}] Skipping ({local_idx:>3}/{len(metadatas)}): {outpath} already has {len(existing)} samples")
            skipped += 1
            continue

        os.makedirs(outpath, exist_ok=True)
        os.makedirs(sample_path, exist_ok=True)
        prompt = metadata['prompt']

        prompt = [f"Please generate image based on the following caption: {prompt}"]
        if "qwen" in prompt_template:
            prompt = add_template(prompt)
        print(f"[Rank {rank}] Prompt ({local_idx:>3}/{len(metadatas)}): '{prompt}'")

        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0
        batch_size = opt.batch_size
        n_rows = opt.batch_size
        with torch.no_grad():
            all_samples = list()
            for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
                gen_img = pipe(prompt, guidance_scale=opt.scale).image

                samples = [gen_img]
                for sample in samples:
                    sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1
                if not opt.skip_grid:
                    all_samples.append(torch.stack([ToTensor()(sample) for sample in samples], 0))

            if not opt.skip_grid:
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, f'grid.png'))
                del grid
        del all_samples

    print(f"[Rank {rank}] Done. Skipped {skipped}/{len(metadatas)} prompts.")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
