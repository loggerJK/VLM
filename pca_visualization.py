#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize saved image representations with RGB PCA.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root directory containing prompt folders")
    parser.add_argument("--sample-id", type=str, default=None, help="Sample subdirectory to process, e.g. 0000")
    parser.add_argument("--save-dirname", type=str, default="pca_visualization", help="Directory name under each prompt for PCA RGB outputs")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    return parser.parse_args()


def pca_to_rgb(features: torch.Tensor) -> np.ndarray:
    features = features.to(torch.float32)
    features = features - features.mean(dim=0, keepdim=True)
    q = min(3, features.shape[0], features.shape[1])
    if q == 0:
        raise ValueError("Empty feature tensor")

    # tokens x dim -> tokens x 3 principal coordinates
    u, s, _ = torch.pca_lowrank(features, q=q, center=False)
    coords = u[:, :q] * s[:q]

    if q < 3:
        pad = torch.zeros(coords.shape[0], 3 - q, dtype=coords.dtype)
        coords = torch.cat([coords, pad], dim=1)

    coords = coords.numpy()
    rgb = np.zeros_like(coords, dtype=np.float32)
    for channel_idx in range(3):
        channel = coords[:, channel_idx]
        channel_min = float(channel.min())
        channel_max = float(channel.max())
        if channel_max > channel_min:
            rgb[:, channel_idx] = (channel - channel_min) / (channel_max - channel_min)
        else:
            rgb[:, channel_idx] = 0.5
    return rgb


def render_representation(pt_path: Path, save_root: Path, dpi: int):
    payload = torch.load(pt_path, map_location="cpu")
    image_representation = payload["image_representation"]
    grid_h, grid_w = payload["grid_size"]

    rgb = pca_to_rgb(image_representation).reshape(grid_h, grid_w, 3)

    save_root.mkdir(parents=True, exist_ok=True)
    save_path = save_root / f"{pt_path.stem}.png"

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(f"step={payload['step']:03d} layer={payload['layer']:02d}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(
        f"[PCA VIZ] step={payload['step']} layer={payload['layer']} "
        f"shape={tuple(image_representation.shape)} path={save_path}",
        flush=True,
    )


def main():
    args = parse_args()

    prompt_dirs = sorted(path for path in args.output_root.iterdir() if path.is_dir())
    for prompt_dir in prompt_dirs:
        rep_root = prompt_dir / "representations"
        if not rep_root.is_dir():
            continue

        sample_dirs = [rep_root / args.sample_id] if args.sample_id else sorted(path for path in rep_root.iterdir() if path.is_dir())
        for sample_dir in sample_dirs:
            if not sample_dir.is_dir():
                continue

            save_root = prompt_dir / args.save_dirname / sample_dir.name
            for pt_path in sorted(sample_dir.glob("step_*_layer_*.pt")):
                render_representation(pt_path, save_root, args.dpi)


if __name__ == "__main__":
    main()
