
export CUDA_VISIBLE_DEVICES=3

python inference/inference_mmu.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --prompt "Please describe this image." \
    --image_path examples/example_6.jpg \
    --steps 128 \
    --gen_length 128 \
    --block_length 32 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output/outputs_text_understanding


# python inference/inference_mmu.py \
#     --checkpoint Alpha-VLLM/Lumina-DiMOO \
#     --prompt "Please describe this image." \
#     --image_path examples/example_6.jpg \
#     --steps 1 \
#     --gen_length 1024 \
#     --block_length 1024 \
#     --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#     --output_dir output/outputs_text_understanding