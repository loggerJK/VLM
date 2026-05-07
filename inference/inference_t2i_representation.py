# -*- coding: utf-8 -*-
"""
Text-to-image inference script with transformer block representation capture.
"""
import os
import json
import argparse
import time
import math
import random
import re
from contextlib import contextmanager

import torch
from PIL import Image
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed, cosine_schedule, gumbel_max_sample, mask_by_random_topk
from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line, encode_img_with_paint
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates


def sanitize_prompt_dirname(prompt_text, max_length=120):
    sanitized = re.sub(r"\s+", "_", prompt_text.strip())
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", sanitized)
    sanitized = sanitized.strip("._")
    if not sanitized:
        sanitized = "prompt"
    return sanitized[:max_length]


def find_subsequence(sequence, subsequence):
    if not subsequence or len(subsequence) > len(sequence):
        return None
    for start_idx in range(len(sequence) - len(subsequence) + 1):
        if sequence[start_idx : start_idx + len(subsequence)] == subsequence:
            return start_idx
    return None


def decode_single_token(tokenizer, token_id):
    try:
        return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode([token_id])


def build_user_prompt_token_metadata(tokenizer, input_prompt, prompt_text, full_token_ids):
    user_tag = "<user>"
    end_tag = "</user>"
    user_start = input_prompt.index(user_tag) + len(user_tag)
    user_end = input_prompt.index(end_tag, user_start)

    token_indices = []
    try:
        encoded = tokenizer(input_prompt, return_offsets_mapping=True)
        encoded_ids = encoded["input_ids"]
        token_offset = 0
        if len(encoded_ids) != len(full_token_ids):
            token_offset = find_subsequence(full_token_ids, encoded_ids)
            if token_offset is None:
                raise ValueError("Failed to align offset-mapped tokens with the full prompt tokenization.")
        offsets = encoded["offset_mapping"]
        for token_idx, (char_start, char_end) in enumerate(offsets):
            if char_end <= char_start:
                continue
            if char_end <= user_start or char_start >= user_end:
                continue
            token_indices.append(token_offset + token_idx)
    except Exception:
        token_indices = []

    if not token_indices:
        user_token_ids = tokenizer(prompt_text)["input_ids"]
        start_idx = find_subsequence(full_token_ids, user_token_ids)
        if start_idx is None:
            user_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            start_idx = find_subsequence(full_token_ids, user_token_ids)
        if start_idx is None:
            raise ValueError("Failed to locate user prompt token span inside the conditional prompt.")
        token_indices = list(range(start_idx, start_idx + len(user_token_ids)))

    token_ids = [full_token_ids[idx] for idx in token_indices]
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
    else:
        token_pieces = [decode_single_token(tokenizer, token_id) for token_id in token_ids]

    return {
        "query_indices": token_indices,
        "token_ids": token_ids,
        "token_pieces": token_pieces,
        "token_decoded": [decode_single_token(tokenizer, token_id) for token_id in token_ids],
        "char_span": [user_start, user_end],
        "prompt_text": prompt_text,
    }


class BlockRepresentationCapture:
    def __init__(self, model):
        self.model = model.module if hasattr(model, "module") else model
        self.blocks = list(self.model.model.transformer.blocks)
        self.enabled = False
        self.capture_config = None
        self.handles = []
        for layer_idx, block in enumerate(self.blocks):
            self.handles.append(block.register_forward_hook(self._make_hook(layer_idx)))

    def _make_hook(self, layer_idx):
        def hook(_module, _inputs, output):
            if not self.enabled or self.capture_config is None:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden[0].detach().to(dtype=torch.float32).cpu()
            image_indices = self.capture_config["image_indices"]
            image_representation = hidden[image_indices]
            payload = {
                "image_representation": image_representation,
                "step": int(self.capture_config["step"]),
                "layer": int(layer_idx),
                "prompt_text": self.capture_config["prompt_text"],
                "sample_idx": int(self.capture_config["sample_idx"]),
                "image_token_indices": list(image_indices),
                "grid_size": list(self.capture_config["grid_size"]),
            }
            save_path = os.path.join(
                self.capture_config["output_dir"],
                f"step_{int(self.capture_config['step']):03d}_layer_{layer_idx:02d}.pt",
            )
            torch.save(payload, save_path)
            print(
                f"[REP SAVE] prompt={self.capture_config['prompt_text']!r} "
                f"step={int(self.capture_config['step'])} layer={layer_idx} "
                f"shape={tuple(image_representation.shape)} path={save_path}",
                flush=True,
            )
        return hook

    @contextmanager
    def activate(self, capture_config):
        self.capture_config = capture_config
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = False
            self.capture_config = None


@torch.no_grad()
def generate_image_with_representations(
    model,
    prompt: torch.LongTensor,
    *,
    seq_len: int,
    timesteps: int,
    mask_token_id: int,
    newline_id: int,
    temperature: float,
    cfg_scale: float,
    uncon_ids: torch.LongTensor,
    code_start: int,
    codebook_size: int,
    generator: torch.Generator,
    representation_capture,
    capture_step_interval: int,
    capture_output_dir: str,
    capture_prompt_text: str,
    capture_sample_idx: int,
    image_indices,
    grid_size,
    use_cache=False,
    cache_ratio=0.9,
    refresh_interval=5,
    warmup_ratio=0.3,
):
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, _ = prompt.shape
    assert B == 1, "batch>1 not supported"

    x = prompt
    vq_mask = x == mask_token_id
    unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = unknown_cnt

    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "caching"):
        m.caching(use_cache)

    warmup_step = int(timesteps * warmup_ratio)
    refresh_steps = torch.zeros(timesteps, dtype=torch.bool)
    for step in range(timesteps):
        if not use_cache or step <= warmup_step or (step - warmup_step) % refresh_interval == 0:
            refresh_steps[step] = True
    compute_ratio = 1 - cache_ratio

    vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
    text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size
    cond_to_compute_mask = None
    uncond_to_compute_mask = None

    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            break

        if step < timesteps - 1:
            frac = cosine_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)

        if use_cache and step and refresh_steps[step] and hasattr(m, "empty_cache"):
            m.empty_cache()

        should_capture = representation_capture is not None and step % capture_step_interval == 0
        capture_config = {
            "output_dir": capture_output_dir,
            "step": step,
            "prompt_text": capture_prompt_text,
            "sample_idx": capture_sample_idx,
            "image_indices": image_indices,
            "grid_size": grid_size,
        }
        cond_compute_mask = None if should_capture else cond_to_compute_mask if (use_cache and not refresh_steps[step] and step > 0) else None

        if cfg_scale > 0:
            uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start - 2:]), axis=1)
            uncond_vq_mask = torch.cat(
                (torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool, device=x.device), vq_mask[:, code_start - 2:]),
                axis=1,
            )
            with representation_capture.activate(capture_config) if should_capture else null_context():
                cond_logits = model(
                    x,
                    infer=True,
                    cat="cond",
                    use_cache=use_cache,
                    to_compute_mask=cond_compute_mask,
                ).logits[..., vocab_offset : vocab_offset + codebook_size]
            cond_mask_logits = cond_logits[vq_mask].view(B, -1, codebook_size)
            uncond_logits = model(
                uncond,
                infer=True,
                cat="uncond",
                use_cache=use_cache,
                to_compute_mask=uncond_to_compute_mask if (use_cache and not refresh_steps[step] and step > 0) else None,
            ).logits[..., vocab_offset : vocab_offset + codebook_size]
            uncond_mask_logits = uncond_logits[uncond_vq_mask].view(B, -1, codebook_size)
            logits = (1 + cfg_scale) * cond_mask_logits - cfg_scale * uncond_mask_logits
        else:
            with representation_capture.activate(capture_config) if should_capture else null_context():
                logits = model(
                    x,
                    infer=True,
                    use_cache=use_cache,
                    to_compute_mask=cond_compute_mask,
                ).logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]

        sampled = gumbel_max_sample(logits, temperature, generator=generator)
        sampled_full = sampled + vocab_offset
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        x.view(-1)[flat_idx] = sampled_full.view(-1)

        mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
        x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

        if use_cache and step < timesteps - 1 and not refresh_steps[step + 1]:
            cond_conf = cond_logits.max(dim=-1)[0]
            cond_conf_threshold = torch.quantile(cond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
            cond_to_compute_mask = cond_conf <= cond_conf_threshold

            uncond_conf = uncond_logits.max(dim=-1)[0]
            uncond_conf_threshold = torch.quantile(uncond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
            uncond_to_compute_mask = uncond_conf <= uncond_conf_threshold

    vq_ids = x[0, code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    return vq_ids


@contextmanager
def null_context():
    yield


def main():
    parser = argparse.ArgumentParser(description="Text-to-image inference with representation capture")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_files", type=str, default=None)
    parser.add_argument("--painting_mode", type=str, default=None)
    parser.add_argument("--painting_image", type=str, default=None)
    parser.add_argument("--mask_h_ratio", type=float, default=1)
    parser.add_argument("--mask_w_ratio", type=float, default=0.2)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vae_ckpt", type=str, default="./vae_ckpt")
    parser.add_argument("--output_dir", type=str, default="results_text_to_image_representation")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--cache_ratio", type=float, default=0.9)
    parser.add_argument("--warmup_ratio", type=float, default=0.3)
    parser.add_argument("--refresh_interval", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--capture-step-interval", type=int, default=16)
    parser.add_argument("--save-representations", dest="save_representations", action="store_true", help="Save transformer block representations during inference")
    parser.add_argument("--no-save-representations", dest="save_representations", action="store_false", help="Disable transformer block representation saving during inference")
    parser.add_argument("--lora_ckpt_path", type=str, default=None)
    parser.set_defaults(save_representations=True)
    args = parser.parse_args()

    if args.prompt is None and args.prompt_files is None:
        parser.error("At least one of --prompt or --prompt_files must be provided.")

    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]
    EOA = SPECIAL_TOKENS["answer_end"]
    BOI = SPECIAL_TOKENS["boi"]
    EOI = SPECIAL_TOKENS["eoi"]

    if args.seed != 0:
        setup_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.lora_ckpt_path:
        print(f"[INFO] Loading LoRA from {args.lora_ckpt_path}")
        model.load_adapter(args.lora_ckpt_path)
    model.eval()

    if args.painting_mode:
        img = Image.open(args.painting_image)
        width, height = img.size
    else:
        height = args.height
        width = args.width

    from diffusers import VQModel
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)
    vqvae.eval()
    vqvae.requires_grad_(False)

    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(height, width)
    templates = create_prompt_templates()
    rep_capture = BlockRepresentationCapture(model) if args.save_representations else None

    prompt_data_list = []
    if args.prompt:
        prompt_data_list.append({"prompt": args.prompt, "metadata": {"tag": "text-to-image", "prompt": args.prompt, "include": [], "exclude": []}})
    if args.prompt_files:
        with open(args.prompt_files, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                prompt_str = data.get("prompt") or data.get("user_prompt")
                if prompt_str:
                    prompt_data_list.append({"prompt": prompt_str, "metadata": data})

    if args.painting_mode:
        img_mask_token, img_vis = encode_img_with_paint(
            img,
            vqvae=vqvae,
            mask_h_ratio=args.mask_h_ratio,
            mask_w_ratio=args.mask_w_ratio,
            mask_mode=args.painting_mode,
        )
    else:
        img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number=NEW_LINE)
    img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]

    for i, prompt_data in enumerate(prompt_data_list):
        prompt_text = prompt_data["prompt"]
        prompt_dir = os.path.join(args.output_dir, sanitize_prompt_dirname(prompt_text))
        samples_dir = os.path.join(prompt_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        with open(os.path.join(prompt_dir, "metadata.jsonl"), "w") as mf:
            json.dump(prompt_data["metadata"], mf, indent=2)

        input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt_text, templates)
        con_prompt_token = tokenizer(input_prompt)["input_ids"]
        uncon_prompt_token = tokenizer(uncon_prompt)["input_ids"]
        user_prompt_meta = build_user_prompt_token_metadata(tokenizer, input_prompt, prompt_text, con_prompt_token)

        prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=device).unsqueeze(0)
        uncon_ids = torch.tensor(uncon_prompt_token, device=device).unsqueeze(0)
        code_start = len(con_prompt_token) + 2
        image_indices = [
            code_start + offset
            for offset, token_id in enumerate(img_mask_token)
            if token_id != NEW_LINE
        ]

        if args.save_representations:
            with open(os.path.join(prompt_dir, "user_prompt_tokens.json"), "w") as tf:
                json.dump(
                    {
                        "prompt_text": prompt_text,
                        "query_indices": user_prompt_meta["query_indices"],
                        "token_ids": user_prompt_meta["token_ids"],
                        "token_pieces": user_prompt_meta["token_pieces"],
                        "token_decoded": user_prompt_meta["token_decoded"],
                        "char_span": user_prompt_meta["char_span"],
                        "grid_size": [token_grid_height, token_grid_width],
                    },
                    tf,
                    ensure_ascii=False,
                    indent=2,
                )

        for sample_idx in range(args.num_samples):
            current_seed = args.seed + sample_idx if args.seed != 0 else random.randint(1, 2**32 - 1)
            setup_seed(current_seed)
            generator = torch.Generator(device=device)
            generator.manual_seed(current_seed)
            sample_rep_dir = None
            if args.save_representations:
                sample_rep_dir = os.path.join(prompt_dir, "representations", f"{sample_idx:04d}")
                os.makedirs(sample_rep_dir, exist_ok=True)

            start_time = time.time()
            vq_tokens = generate_image_with_representations(
                model,
                prompt_ids.clone(),
                seq_len=seq_len,
                timesteps=args.timesteps,
                mask_token_id=MASK,
                newline_id=NEW_LINE,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                uncon_ids=uncon_ids,
                code_start=code_start,
                codebook_size=8192,
                generator=generator,
                representation_capture=rep_capture,
                capture_step_interval=args.capture_step_interval,
                capture_output_dir=sample_rep_dir,
                capture_prompt_text=prompt_text,
                capture_sample_idx=sample_idx,
                image_indices=image_indices,
                grid_size=[token_grid_height, token_grid_width],
                use_cache=args.use_cache,
                cache_ratio=args.cache_ratio,
                refresh_interval=args.refresh_interval,
                warmup_ratio=args.warmup_ratio,
            )

            filename = f"{sample_idx:04d}.png"
            save_path = os.path.join(samples_dir, filename)
            out_img = decode_vq_to_image(
                vq_tokens,
                save_path,
                vae_ckpt=args.vae_ckpt,
                image_height=height,
                image_width=width,
                vqvae=vqvae,
            )
            out_img.save(save_path)
            print(f"[REP] saved image {save_path} in {time.time() - start_time:.2f}s", flush=True)


if __name__ == "__main__":
    main()
