#!/usr/bin/env python3
"""Fix legacy `.default.` keys in LoRA adapter checkpoints.

Some early LoRA checkpoints contain keys like `...lora_A.default.weight`.
Newer PEFT versions expect keys without `.default.` (e.g., `...lora_A.weight`).
This script finds and fixes all such adapter_model.safetensors files in-place.

Usage:
    python scripts/fix_adapter_default_keys.py --checkpoint_dir ./checkpoints
"""

import argparse
import glob
import os

from safetensors.torch import load_file, save_file


def fix_adapter_file(adapter_path: str) -> bool:
    """Fix `.default.` keys in a single adapter file. Returns True if modified."""
    sd = load_file(adapter_path)

    if not any(".default." in k for k in sd.keys()):
        return False

    fixed_sd = {k.replace(".default.", "."): v for k, v in sd.items()}
    save_file(fixed_sd, adapter_path)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix legacy .default. keys in LoRA adapter checkpoints"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Root directory to search for adapter_model.safetensors files",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.checkpoint_dir, "**", "adapter_model.safetensors")
    adapter_files = sorted(glob.glob(pattern, recursive=True))

    if not adapter_files:
        print(f"No adapter_model.safetensors files found under {args.checkpoint_dir}")
        return

    print(f"Found {len(adapter_files)} adapter file(s) under {args.checkpoint_dir}\n")

    fixed_count = 0
    skipped_count = 0

    for path in adapter_files:
        if fix_adapter_file(path):
            fixed_count += 1
            print(f"  FIXED: {path}")
        else:
            skipped_count += 1
            print(f"  OK:    {path}")

    print(f"\nSummary: {fixed_count} fixed, {skipped_count} already OK, {len(adapter_files)} total")


if __name__ == "__main__":
    main()
