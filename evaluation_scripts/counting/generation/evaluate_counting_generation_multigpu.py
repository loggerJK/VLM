"""
Counting Generation Evaluation Script (Multi-GPU) — Janus

Generates images from GenEval metadata prompts for counting evaluation.
Adapted from geneval_janus_evalulate_multigpu.py with conventions aligned
to evaluation_scripts/ocr/generation/.

Flow: GenEval metadata JSONL -> Janus generates images -> saved per prompt folder
Output structure: {output_dir}/{index:05d}/samples/{i:05d}.png
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from transformers import AutoModelForCausalLM, set_seed

from janus.models import MultiModalityCausalLM, VLChatProcessor


torch.set_grad_enabled(False)


def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    # Resolve the base transformer model for forward passes.
    # Without PeftModel: language_model.model -> base transformer (has last_hidden_state)
    # With PeftModel:    language_model.model -> CausalLM (no last_hidden_state)
    #                    language_model.model.model -> base transformer (has last_hidden_state)
    # LoRA-adapted linear layers live inside the base transformer, so bypassing
    # the CausalLM wrapper does NOT skip LoRA.
    try:
        from peft import PeftModel
        if isinstance(mmgpt.language_model, PeftModel):
            inner_model = mmgpt.language_model.model.model
        else:
            inner_model = mmgpt.language_model.model
    except ImportError:
        inner_model = mmgpt.language_model.model

    device = inner_model.device
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    # Generate a single sample at a time to keep peak memory bounded.
    tokens = torch.zeros((2, len(input_ids)), dtype=torch.int).to(device)
    tokens[0, :] = input_ids
    tokens[1, :] = input_ids
    tokens[1, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((1, image_token_num_per_image), dtype=torch.int).to(device)

    outputs = None
    for i in range(image_token_num_per_image):
        outputs = inner_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[0, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)


    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[1, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec[0]

    return Image.fromarray(visual_img)


def parse_args():
    parser = argparse.ArgumentParser(description="Counting Generation Evaluation (Janus, Multi-GPU)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="deepseek-ai/Janus-Pro-7B",
        help="Huggingface model path or local checkpoint directory"
    )
    parser.add_argument(
        "--lora_ckpt_path",
        type=str,
        default=None,
        help="Path to LoRA checkpoint directory (optional)"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="/mnt/data1/jiwon/geneval/prompts/evaluation_metadata.jsonl",
        help="JSONL file containing lines of metadata for each prompt"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to write results to",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of samples per prompt",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="Image size in pixel space",
    )
    parser.add_argument(
        "--cfg_weight",
        type=float,
        default=5.0,
        help="Unconditional guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="Skip saving grid image",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup distributed environment
    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(days=1))
        device = torch.device("cuda", local_rank)
        print(f"[Rank {rank}] Distributed initialized. World Size: {world_size}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("[INFO] Running in single process mode.")

    set_all_seeds(args.seed)

    # Load prompts
    with open(args.metadata_file) as fp:
        all_metadatas = [json.loads(line) for line in fp]

    # Create indexed list to preserve original indices for folder naming
    indexed_metadatas = list(enumerate(all_metadatas))

    # Shard data
    my_batch = indexed_metadatas[rank::world_size]
    print(f"Rank {rank} processing {len(my_batch)} prompts out of {len(all_metadatas)} total.")
    print(f"[Rank {rank}] args.n_samples: {args.n_samples}")
    

    # Load model
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = model.to(torch.bfloat16).to(device).eval()

    # Load LoRA weights if provided
    if args.lora_ckpt_path is not None:
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(
            model.language_model, args.lora_ckpt_path, torch_device='cpu',
        )
        model = model.to(torch.bfloat16).to(device).eval()
        if rank == 0:
            print(f"Loaded LoRA weights from {args.lora_ckpt_path}")

    for index, metadata in tqdm(my_batch, total=len(my_batch), desc=f"Rank {rank} Processing"):
        outpath = os.path.join(args.output_dir, f"{index:0>5}")

        # --- Resume: skip if already completed ---
        if not args.skip_grid:
            already_done = os.path.exists(os.path.join(outpath, "grid.png"))
        else:
            sample_path = os.path.join(outpath, "samples")
            done_list = [os.path.exists(os.path.join(sample_path, f"{i:05}.png")) for i in range(args.n_samples)]
            already_done = all(done_list)

        if already_done:
            print(f"[Rank {rank}] Skipping ({index: >3}/{len(all_metadatas)}): already completed.")
            continue
        # ---

        os.makedirs(outpath, exist_ok=True)

        prompt_text = metadata['prompt']
        print(f"[Rank {rank}] Prompt ({index: >3}/{len(all_metadatas)}): '{prompt_text}'")

        # Format prompt for Janus
        conversation = [
            {"role": "<|User|>", "content": prompt_text},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        with torch.no_grad():
            all_samples_tensors = []
            for i in range(args.n_samples):
                sample_seed = args.seed + (index * args.n_samples) + i
                set_all_seeds(sample_seed)
                sample = generate(
                    model,
                    vl_chat_processor,
                    prompt,
                    cfg_weight=args.cfg_weight,
                    img_size=args.img_size,
                )
                sample.save(os.path.join(sample_path, f"{i:05}.png"))
                if not args.skip_grid:
                    all_samples_tensors.append(ToTensor()(sample))
                del sample

            if not args.skip_grid and all_samples_tensors:
                # additionally, save as grid
                grid = make_grid(torch.stack(all_samples_tensors, 0), nrow=args.n_samples)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, 'grid.png'))
                del grid
        if not args.skip_grid:
            del all_samples_tensors


    print(f"Rank {rank} Done.")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
