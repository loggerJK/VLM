"""
VQ encode/decode round-trip test for Janus VQ-16 model.
Verifies that:
  - indices.shape == [576] (24x24)
  - indices values in [0, 16383] (codebook_size=16384)
  - Reconstructed image is visually similar to original
  - Works in both fp32 and bf16 without NaN
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image

from janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoConfig


def encode_image_to_vq_tokens(vq_model, image, img_size=384):
    """PIL Image -> 576 VQ token IDs (LongTensor)"""
    image = image.resize((img_size, img_size))
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor * 2 - 1  # [0,1] -> [-1,1]
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype

    with torch.no_grad():
        _, _, (_, _, indices) = vq_model.encode(image_tensor.to(device=device, dtype=dtype))

    return indices.view(-1)  # LongTensor [576]


def decode_vq_tokens_to_image(vq_model, indices, img_size=384):
    """VQ token IDs -> PIL Image"""
    # decode_code expects shape = [batch, codebook_embed_dim, h, w]
    # spatial_size = img_size / 16 (VQ-16 downsamples by 16x)
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    spatial_size = img_size // 16

    with torch.no_grad():
        decoded = vq_model.decode_code(
            indices.to(device),
            shape=[1, 8, spatial_size, spatial_size],
            channel_first=True,
        )

    # [-1, 1] -> [0, 255]
    decoded = ((decoded + 1) / 2 * 255).clamp(0, 255)
    decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().to(torch.uint8).numpy()
    return Image.fromarray(decoded)


def compute_psnr(img1, img2):
    """Compute PSNR between two PIL images."""
    a1 = np.array(img1).astype(np.float64)
    a2 = np.array(img2).astype(np.float64)
    mse = np.mean((a1 - a2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B")
    parser.add_argument("--image_path", type=str, required=True, help="Path to test image")
    parser.add_argument("--output_dir", type=str, default="output/debug_vq_janus")
    parser.add_argument("--img_size", type=int, default=384)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model_path}...")
    config = AutoConfig.from_pretrained(args.model_path)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'

    for test_dtype, dtype_name in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        print(f"\n{'='*50}")
        print(f"Testing with {dtype_name}")
        print(f"{'='*50}")

        model = MultiModalityCausalLM.from_pretrained(
            args.model_path,
            language_config=language_config,
            trust_remote_code=True,
            torch_dtype=test_dtype,
        )
        model = model.to(device)
        model.eval()

        vq_model = model.gen_vision_model

        # Load and resize original image
        original = Image.open(args.image_path).convert("RGB")
        original_resized = original.resize((args.img_size, args.img_size))

        # Encode
        indices = encode_image_to_vq_tokens(vq_model, original, img_size=args.img_size)
        print(f"indices.shape: {indices.shape}")
        print(f"indices dtype: {indices.dtype}")
        print(f"indices min: {indices.min().item()}, max: {indices.max().item()}")
        print(f"unique tokens: {indices.unique().shape[0]}")

        # Verification checks
        expected_tokens = (args.img_size // 16) ** 2
        assert indices.shape == (expected_tokens,), f"Expected shape ({expected_tokens},), got {indices.shape}"
        assert indices.min() >= 0, f"Min index {indices.min()} < 0"
        assert indices.max() < 16384, f"Max index {indices.max()} >= 16384"
        print("Shape and range checks PASSED")

        # Decode
        reconstructed = decode_vq_tokens_to_image(vq_model, indices, img_size=args.img_size)

        # Check for NaN
        recon_np = np.array(reconstructed).astype(np.float64)
        assert not np.isnan(recon_np).any(), f"NaN detected in {dtype_name} reconstruction!"
        print(f"No NaN in reconstruction: PASSED")

        # Compute PSNR
        psnr = compute_psnr(original_resized, reconstructed)
        print(f"PSNR: {psnr:.2f} dB")

        # Save images side by side
        w, h = original_resized.size
        comparison = Image.new("RGB", (w * 2, h))
        comparison.paste(original_resized, (0, 0))
        comparison.paste(reconstructed, (w, 0))
        comparison.save(os.path.join(args.output_dir, f"comparison_{dtype_name}.png"))
        reconstructed.save(os.path.join(args.output_dir, f"recon_{dtype_name}.png"))
        print(f"Saved to {args.output_dir}/comparison_{dtype_name}.png")

        # Also test prepare_gen_img_embeds
        gen_embeds = model.prepare_gen_img_embeds(indices.to(device))
        print(f"gen_embeds shape: {gen_embeds.shape}")
        assert not torch.isnan(gen_embeds).any(), "NaN in gen_embeds!"
        print("prepare_gen_img_embeds check: PASSED")

        del model
        torch.cuda.empty_cache()

    print("\nAll tests PASSED!")


if __name__ == "__main__":
    main()
