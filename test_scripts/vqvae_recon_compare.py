import os
import glob
import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import VQModel
from diffusers.image_processor import VaeImageProcessor
def compute_psnr(gt_np, recon_np, data_range=255.0):
    mse = np.mean((gt_np - recon_np) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(data_range ** 2 / mse)


def make_comparison_image(gt_img, recon_img):
    """Create a side-by-side comparison: GT | Recon | Diff (amplified)."""
    gt_np = np.array(gt_img).astype(np.float32)
    recon_np = np.array(recon_img).astype(np.float32)

    diff = np.abs(gt_np - recon_np)
    diff_amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)

    w, h = gt_img.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(gt_img, (0, 0))
    canvas.paste(recon_img, (w, 0))
    canvas.paste(Image.fromarray(diff_amplified), (w * 2, 0))
    return canvas


def compute_metrics(gt_img, recon_img):
    """Compute PSNR between GT and reconstruction."""
    gt_np = np.array(gt_img).astype(np.float32)
    recon_np = np.array(recon_img).astype(np.float32)
    p = compute_psnr(gt_np, recon_np)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, help="Resize images to this size (square)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = "cuda"
    dtype = torch.bfloat16
    model_id = "Alpha-VLLM/Lumina-DiMOO"
    input_dir = "/mnt/data1/jiwon/ocr_dataset_construction/output"
    output_dir = f"/mnt/data1/jiwon/Lumina-DiMOO/test_scripts/output/vqvae_recon_compare_{args.image_size}"

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading VQ-VAE from {model_id} (bf16)...")
    vqvae = VQModel.from_pretrained(model_id, subfolder="vqvae", torch_dtype=dtype).to(device)
    vqvae.eval()

    scale_factor = 2 ** (len(vqvae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=scale_factor, do_normalize=False)

    # Gather images
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    print(f"Found {len(image_paths)} images in {input_dir}\n")

    results = []

    for img_path in image_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing {name}...")

        # Load and prepare image
        original = Image.open(img_path).convert("RGB").resize((args.image_size, args.image_size), Image.LANCZOS)
        w, h = original.size
        new_w = (w // scale_factor) * scale_factor
        new_h = (h // scale_factor) * scale_factor
        gt_img = original.resize((new_w, new_h), Image.LANCZOS)

        x = image_processor.preprocess(gt_img).to(device=device, dtype=dtype)

        with torch.no_grad():
            # Encode → Quantize → Decode
            enc_out = vqvae.encode(x).latents
            indices = vqvae.quantize(enc_out)[2][2]

            latent_h, latent_w = new_h // scale_factor, new_w // scale_factor
            indices_grid = indices.view(1, latent_h, latent_w)

            target_shape = (1, latent_h, latent_w, vqvae.config.latent_channels)
            dec_out = vqvae.decode(indices_grid, force_not_quantize=True, shape=target_shape).sample

            recon_img = image_processor.postprocess(
                dec_out.detach().float().clip(0, 1), output_type="pil"
            )[0]

        # Compute metrics
        p = compute_metrics(gt_img, recon_img)
        results.append({"name": name, "psnr": p})
        print(f"  PSNR: {p:.2f} dB")

        # Save individual images
        gt_img.save(os.path.join(output_dir, f"{name}_gt.png"))
        recon_img.save(os.path.join(output_dir, f"{name}_recon.png"))

        # Save comparison (GT | Recon | Diff x5)
        comp = make_comparison_image(gt_img, recon_img)
        comp.save(os.path.join(output_dir, f"{name}_compare.png"))

    # Summary
    print("\n" + "=" * 50)
    print(f"{'Image':<12} {'PSNR (dB)':>10}")
    print("-" * 50)
    psnr_vals = []
    for r in results:
        print(f"{r['name']:<12} {r['psnr']:>10.2f}")
        psnr_vals.append(r["psnr"])
    print("-" * 50)
    print(f"{'Average':<12} {np.mean(psnr_vals):>10.2f}")
    print("=" * 50)

    # Save summary to text file
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("VQ-VAE Reconstruction Quality (bf16)\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Image size: {args.image_size}x{args.image_size}\n")
        f.write(f"Input: {input_dir}\n\n")
        f.write(f"{'Image':<12} {'PSNR (dB)':>10}\n")
        f.write("-" * 25 + "\n")
        for r in results:
            f.write(f"{r['name']:<12} {r['psnr']:>10.2f}\n")
        f.write("-" * 25 + "\n")
        f.write(f"{'Average':<12} {np.mean(psnr_vals):>10.2f}\n")
        f.write(f"\nComparison images: GT | Recon | Diff (x5 amplified)\n")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
