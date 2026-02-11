import os
import torch
from PIL import Image
from diffusers import VQModel
from diffusers.image_processor import VaeImageProcessor

def test_vqvae_clean_api():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Alpha-VLLM/Lumina-DiMOO"
    image_path = "examples/example_1.png"
    output_dir = "output/debug_vqvae_clean"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing VQ-VAE with Clean API from {model_id} on {device}...")
    
    # Load Image
    original_image = Image.open(image_path).convert("RGB").resize((512, 512))
    w, h = original_image.size
    
    # # Load Model (FP32)
    # vqvae = VQModel.from_pretrained(model_id, subfolder="vqvae", torch_dtype=torch.float32).to(device)
    # vqvae.eval()
    
    # scale_factor = 2 ** (len(vqvae.config.block_out_channels) - 1)
    # new_w, new_h = (w // scale_factor) * scale_factor, (h // scale_factor) * scale_factor
    # original_image = original_image.resize((new_w, new_h), Image.LANCZOS)
    
    
    dtypes = [
        torch.float32, 
        # torch.float16, 
        # torch.bfloat16
    ]
    dtype_names = [
        "fp32", 
        # "fp16", 
        # "bf16"
    ]

    for dtype, dtype_name in zip(dtypes, dtype_names):
        print(f"\n--- Testing {dtype_name} ---")
        try:
            # Reload model in target dtype
            vqvae = VQModel.from_pretrained(model_id, subfolder="vqvae", torch_dtype=dtype).to(device)
            vqvae.eval()
            
            scale_factor = 2 ** (len(vqvae.config.block_out_channels) - 1)
            new_w, new_h = (w // scale_factor) * scale_factor, (h // scale_factor) * scale_factor
            original_image = original_image.resize((new_w, new_h), Image.LANCZOS)
            
            image_processor = VaeImageProcessor(vae_scale_factor=scale_factor, do_normalize=False)
            
            x = image_processor.preprocess(original_image).to(device=device, dtype=dtype)

            with torch.no_grad():
                # 1. Get Indices
                enc_out = vqvae.encode(x).latents  # Encoder output, (B, C, H, W)
                print(f"enc_out.shape: {enc_out.shape}, dtype: {enc_out.dtype}")
                indices = vqvae.quantize(enc_out)[2][2]  # Quantized indices, (B, H, W)
                print(f"indices.shape: {indices.shape}, dtype: {indices.dtype}")
                
                latent_h, latent_w = new_h // scale_factor, new_w // scale_factor
                indices_grid = indices.view(1, latent_h, latent_w)
                
                # 2. Decode using official API
                target_shape = (1, latent_h, latent_w, vqvae.config.latent_channels)
                
                dec_out = vqvae.decode(indices_grid, force_not_quantize=True, shape=target_shape).sample
                
                # 3. Direct reference
                direct_out = vqvae(x).sample
                
                # Postprocess & Save
                img_api = image_processor.postprocess(dec_out.detach().float().clip(0, 1), output_type="pil")[0]
                img_direct = image_processor.postprocess(direct_out.detach().float().clip(0, 1), output_type="pil")[0]
                
                img_api.save(os.path.join(output_dir, f"recon_api_{dtype_name}.png"))
                img_direct.save(os.path.join(output_dir, f"recon_direct_{dtype_name}.png"))
                
                print(f"[{dtype_name}] API Output Stats: min={dec_out.min().item():.4f}, max={dec_out.max().item():.4f}")
                print(f"[{dtype_name}] Direct Output Stats: min={direct_out.min().item():.4f}, max={direct_out.max().item():.4f}")
                
                if torch.isnan(dec_out).any():
                    print(f"!!! {dtype_name} output contains NaNs !!!")

        except Exception as e:
            print(f"Failed with {dtype_name}: {e}")
            import traceback
            traceback.print_exc()
            
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_vqvae_clean_api()
