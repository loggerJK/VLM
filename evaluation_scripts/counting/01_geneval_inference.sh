# export CUDA_VISIBLE_DEVICES=7

# python inference/inference_t2i.py\
#     --checkpoint Alpha-VLLM/Lumina-DiMOO \
#     --height 1024 \
#     --width 1024 \
#     --timesteps 64 \
#     --cfg_scale 4.0 \
#     --seed 65513 \
#     --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#     --output_dir /mnt/data1/heeji/geneval/lumina \
#     --prompt_files /home/cvlab12/project/heeji/Lumina-DiMOO/evaluation_metadata_count.jsonl
#     # --prompt "A striking photograph of a glass of orange juice on a wooden kitchen table, capturing a playful moment. The orange juice splashes out of the glass and forms the word \"Smile\" in a whimsical, swirling script just above the glass. The background is softly blurred, revealing a cozy, homely kitchen with warm lighting and a sense of comfort." \

# python inference/inference_t2i.py\
#     --checkpoint /mnt/data1/jiwon/Lumina-DiMOO/output/Lumina-DiMOO-counting-full/epoch3-iter38911-step5900 \
#     --height 1024 \
#     --width 1024 \
#     --timesteps 64 \
#     --cfg_scale 4.0 \
#     --seed 65513 \
#     --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#     --output_dir /mnt/data1/heeji/geneval/lumina_full_counting_epoch3-iter38911-step5900 \
#     --prompt_files /mnt/data1/jiwon/geneval/prompts/evaluation_metadata_count.jsonl

# CUDA_VISIBLE_DEVICES=0 python inference/inference_t2i.py\
#     --checkpoint /mnt/data1/jiwon/Lumina-DiMOO/output/Lumina-DiMOO-counting-full/epoch3-iter38911-step5900 \
#     --height 512 \
#     --width 512 \
#     --timesteps 64 \
#     --cfg_scale 4.0 \
#     --seed 65513 \
#     --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#     --output_dir /mnt/data1/heeji/geneval/lumina_full_counting_epoch3-iter38911-step5900_res512 \
#     --prompt_files /mnt/data1/jiwon/geneval/prompts/evaluation_metadata_count.jsonl


# CUDA_VISIBLE_DEVICES=1 python inference/inference_t2i.py\
#     --checkpoint Alpha-VLLM/Lumina-DiMOO \
#     --height 512 \
#     --width 512 \
#     --timesteps 64 \
#     --cfg_scale 4.0 \
#     --seed 65513 \
#     --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#     --lora_ckpt_path /mnt/data1/jiwon/Lumina-DiMOO/output/Lumina-DiMOO-counting-lora128/epoch0-iter63999-step1000 \
#     --output_dir /mnt/data1/heeji/geneval/lumina_lora128_counting_res512 \
#     --prompt_files /mnt/data1/jiwon/geneval/prompts/evaluation_metadata_count.jsonl

# CUDA_VISIBLE_DEVICES=2 python inference/inference_t2i.py\
#     --checkpoint Alpha-VLLM/Lumina-DiMOO \
#     --height 512 \
#     --width 512 \
#     --timesteps 64 \
#     --cfg_scale 4.0 \
#     --seed 65513 \
#     --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#     --lora_ckpt_path /mnt/data1/jiwon/Lumina-DiMOO/output/lora128_counting_wohead/epoch0-iter31999-step1000 \
#     --output_dir /mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_res512 \
#     --prompt_files /mnt/data1/jiwon/geneval/prompts/evaluation_metadata_count.jsonl

# CUDA_VISIBLE_DEVICES=3 python inference/inference_t2i.py\
#     --checkpoint Alpha-VLLM/Lumina-DiMOO \
#     --height 1024 \
#     --width 1024 \
#     --timesteps 64 \
#     --cfg_scale 4.0 \
#     --seed 65513 \
#     --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#     --lora_ckpt_path /mnt/data1/jiwon/Lumina-DiMOO/output/lora128_counting_wohead/epoch6-iter45311-step12000 \
#     --output_dir /mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_epoch6-iter45311-step12000 \
#     --prompt_files /mnt/data1/jiwon/geneval/prompts/evaluation_metadata_count.jsonl




#############################################
# evaluate position
python inference/inference_t2i.py\
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --height 1024 \
    --width 1024 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --seed 65513 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir /mnt/data1/heeji/geneval/position/baseline \
    --prompt_files /mnt/data1/jiwon/geneval/prompts/prompts/evaluation_metadata_position.jsonl
    # --lora_ckpt_path /mnt/cvlab22_data1/heeji/dVLM_outputs_from_ext_srv/lora128_spatial_relation_wohead_und-20260126-144357/epoch1-iter34431-step2500 \
#     --lora_ckpt_path /mnt/data1/jiwon/Lumina-DiMOO/output/lora128_counting_wohead/epoch10 \
#     --output_dir /mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_epoch10 \
#     --prompt_files /mnt/data1/jiwon/geneval/prompts/evaluation_metadata_count.jsonl

# CUDA_VISIBLE_DEVICES=3 python inference/inference_t2i.py\
#     --checkpoint Alpha-VLLM/Lumina-DiMOO \
#     --height 1024 \
#     --width 1024 \
#     --timesteps 64 \
#     --cfg_scale 4.0 \
#     --seed 65513 \
#     --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#     --lora_ckpt_path /mnt/data1/jiwon/Lumina-DiMOO/output/lora128_counting_wohead/epoch12 \
#     --output_dir /mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_epoch12 \
#     --prompt_files /mnt/data1/jiwon/geneval/prompts/evaluation_metadata_count.jsonl

