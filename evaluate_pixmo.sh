CUDA_VISIBLE_DEVICES=2 python evaluate_pixmo.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --lora_ckpt_path /mnt/data1/jiwon/Lumina-DiMOO/output/Lumina-DiMOO-counting-lora128/epoch0-iter63999-step1000 \
    --output_dir ./evaluation_results/lora128_counting \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 

CUDA_VISIBLE_DEVICES=6 python evaluate_pixmo.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir ./evaluation_results/baseline \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 

CUDA_VISIBLE_DEVICES=7 python evaluate_pixmo.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --lora_ckpt_path /mnt/data1/jiwon/Lumina-DiMOO/output/lora128_counting_wohead/epoch0-iter31999-step1000 \
    --output_dir ./evaluation_results/lora128_counting_wohead \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 

