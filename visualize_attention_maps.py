#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SKIP_TOKEN_LABELS = {"a", "photo", "of"}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize saved attention maps as 4x8 layer canvases.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root output directory containing prompt folders")
    parser.add_argument("--sample-id", type=str, default=None, help="Sample subdirectory to visualize, e.g. 0000. Default: all samples")
    parser.add_argument("--save-dirname", type=str, default="attention_viz", help="Directory name under each prompt for visualization outputs")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument("--figscale", type=float, default=1.0, help="Scale factor for figure size")
    parser.add_argument("--head-reduction", choices=["mean", "max"], default="mean", help="How to reduce the head axis")
    parser.add_argument("--direction", choices=["t2i", "i2t", "both"], default="both", help="Which saved attention direction(s) to visualize")
    return parser.parse_args()


def reduce_heads(attention_map, reduction):
    if reduction == "mean":
        return attention_map.mean(axis=0)
    if reduction == "max":
        return attention_map.max(axis=0)
    raise ValueError(f"Unsupported head reduction: {reduction}")


def collect_step_layers(sample_dir):
    direction_to_step_layer = {}
    for pt_path in sorted(sample_dir.glob("step_*_layer_*.pt")):
        payload = torch.load(pt_path, map_location="cpu")
        step = int(payload["step"])
        layer = int(payload["layer"])
        direction = payload.get("direction")
        if direction is None:
            direction = "t2i" if pt_path.stem.endswith("_t2i") else "i2t" if pt_path.stem.endswith("_i2t") else "t2i"
        direction_to_step_layer.setdefault(direction, {}).setdefault(step, {})[layer] = payload
    return direction_to_step_layer


def clean_token_label(token_text):
    label = token_text.replace("\n", "\\n")
    if len(label) > 30:
        label = label[:27] + "..."
    return label


def should_skip_token(token_text):
    normalized = token_text.strip().lower()
    return normalized in SKIP_TOKEN_LABELS


def render_canvas(prompt_dir, sample_dir, save_dirname, direction, step, token_idx, token_label, layer_payloads, dpi, figscale, head_reduction):
    first_payload = next(iter(layer_payloads.values()))
    grid_h, grid_w = first_payload["grid_size"]
    layer_ids = sorted(layer_payloads.keys())

    reduced_maps = {}
    for layer_id in layer_ids:
        attention_map = layer_payloads[layer_id]["attention_map"].numpy()
        reduced_attention = reduce_heads(attention_map, head_reduction)
        if direction == "t2i":
            token_map = reduced_attention[token_idx].reshape(grid_h, grid_w)
        elif direction == "i2t":
            token_map = reduced_attention[:, token_idx].reshape(grid_h, grid_w)
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        reduced_maps[layer_id] = token_map

    vmin = min(float(arr.min()) for arr in reduced_maps.values())
    vmax = max(float(arr.max()) for arr in reduced_maps.values())
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-8

    fig, axes = plt.subplots(4, 8, figsize=(32 * figscale, 16 * figscale), dpi=dpi, constrained_layout=True)
    fig.suptitle(
        f"Prompt: {prompt_dir.name} | Sample: {sample_dir.name} | Direction: {direction} | Step: {step:03d} | Token {token_idx}: {token_label}",
        fontsize=18,
    )

    image_artist = None
    flat_axes = axes.flatten()
    for ax_idx, ax in enumerate(flat_axes):
        if ax_idx >= len(layer_ids):
            ax.axis("off")
            continue
        layer_id = layer_ids[ax_idx]
        image_artist = ax.imshow(reduced_maps[layer_id], cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Layer {layer_id}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    if image_artist is not None:
        fig.colorbar(image_artist, ax=flat_axes.tolist(), shrink=0.92, pad=0.01)

    save_root = prompt_dir / save_dirname / direction / sample_dir.name / f"step_{step:03d}"
    save_root.mkdir(parents=True, exist_ok=True)
    save_path = save_root / f"token_{token_idx:03d}.png"
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(
        f"[ATTN VIZ] prompt={prompt_dir.name} sample={sample_dir.name} direction={direction} "
        f"step={step} token={token_idx} label={token_label!r} path={save_path}",
        flush=True,
    )


def main():
    args = parse_args()
    prompt_dirs = sorted(path for path in args.output_root.iterdir() if path.is_dir())
    for prompt_dir in prompt_dirs:
        attention_root = prompt_dir / "attentions"
        if not attention_root.is_dir():
            continue

        sample_dirs = [attention_root / args.sample_id] if args.sample_id else sorted(path for path in attention_root.iterdir() if path.is_dir())
        for sample_dir in sample_dirs:
            if not sample_dir.is_dir():
                continue
            direction_to_step_layer = collect_step_layers(sample_dir)
            if not direction_to_step_layer:
                continue

            directions = ["t2i", "i2t"] if args.direction == "both" else [args.direction]
            for direction in directions:
                step_to_layer = direction_to_step_layer.get(direction, {})
                if not step_to_layer:
                    continue

                first_step = min(step_to_layer)
                first_payload = next(iter(step_to_layer[first_step].values()))
                token_labels = first_payload["query_token_decoded"]

                for step, layer_payloads in sorted(step_to_layer.items()):
                    for token_idx, token_label in enumerate(token_labels):
                        if should_skip_token(token_label):
                            continue
                        render_canvas(
                            prompt_dir=prompt_dir,
                            sample_dir=sample_dir,
                            save_dirname=args.save_dirname,
                            direction=direction,
                            step=step,
                            token_idx=token_idx,
                            token_label=clean_token_label(token_label),
                            layer_payloads=layer_payloads,
                            dpi=args.dpi,
                            figscale=args.figscale,
                            head_reduction=args.head_reduction,
                        )


if __name__ == "__main__":
    main()
