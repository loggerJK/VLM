CUDA_VISIBLE_DEVICES=0 python evaluate_blink_spatial.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir ./evaluation_results/lora_spatial_base \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 
    # --lora_ckpt_path /mnt/cvlab22_data1/heeji/dVLM_outputs_from_ext_srv/lora128_spatial_relation_wohead_und-20260126-144357/epoch1-iter34431-step2500 \

