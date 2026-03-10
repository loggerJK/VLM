python inference/inference_t2i_batch.py\
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --height 1024 \
    --width 1024 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --seed 65513 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output/job_prompts_test \
    --prompt_file job_prompts.txt  