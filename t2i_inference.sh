python inference/inference_t2i_batch.py\
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --height 512 \
    --width 512 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --seed 65513 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output/generation/baseline_512 \
    --prompt_file /mnt/data1/heeji/VLM/output/Lumina-Generation-20260115-133152/validation_prompts.txt 