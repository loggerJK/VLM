# python inference/inference_t2i_batch.py\
#     --checkpoint Alpha-VLLM/Lumina-DiMOO \
#     --height 1024 \
#     --width 1024 \
#     --timesteps 64 \
#     --cfg_scale 4.0 \
#     --seed 65513 \
#     --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#     --output_dir output/job_prompts_test \
#     --prompt_file job_prompts.txt  
python inference/inference_t2i_seeds.py\
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --height 1024 \
    --width 1024 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --seed 65513 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output2/count_test_baseline \
    --prompt_files ./count_subset.jsonl \
    --attention-direction both 

python inference/inference_t2i_seeds.py\
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --height 1024 \
    --width 1024 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --seed 65513 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output2/count_test_und \
    --prompt_files ./count_subset.jsonl \
    --lora_ckpt_path /data/mm-llm-backbone_890/personal/sirius/audio_ablation/lumina_checkpoints/lumina_lora128_counting_wohead_epoch12 \
    --attention-direction both 

python inference/inference_t2i_seeds.py\
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --height 1024 \
    --width 1024 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --seed 65513 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output2/count_test_gen \
    --prompt_files ./count_subset.jsonl \
    --lora_ckpt_path /data/mm-llm-backbone_890/personal/sirius/audio_ablation/lumina_checkpoints/lumina_lumina_train_generation_wohead_1024_epoch12 \
    --attention-direction both 
