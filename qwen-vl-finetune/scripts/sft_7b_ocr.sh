#!/bin/bash

# OCR synthetic task LoRA fine-tuning for Qwen2.5-VL-7B
# Dataset: agentlans/high-quality-english-sentences (first 200K)
# No DeepSpeed — plain DDP with 8-bit AdamW

# GPU configuration
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4,5"}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# effective_batch = per_device_batch * grad_accum * NUM_GPUS = 128
EFFECTIVE_BATCH=128
PER_DEVICE_BATCH=1
GRAD_ACCUM=$((EFFECTIVE_BATCH / PER_DEVICE_BATCH / NUM_GPUS))

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

echo "GPUs: ${NUM_GPUS}, grad_accum: ${GRAD_ACCUM}, effective_batch: $((PER_DEVICE_BATCH * GRAD_ACCUM * NUM_GPUS))"

torchrun --nproc_per_node=${NUM_GPUS} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
    qwenvl/train/train_qwen_hf.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --task ocr \
    --data_flatten False \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --optim adamw_bnb_8bit \
    --bf16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate 2e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --max_grad_norm 1 \
    --max_pixels 50176 \
    --min_pixels 784 \
    --model_max_length 8192 \
    --save_steps 500 \
    --save_steps_callback 500 \
    --val_log_freq 500 \
    --save_strategy no \
    --eval_strategy no \
    --gradient_checkpointing True \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --output_dir ./output/ocr \
    --run_name qwen2vl-ocr-lora \
    --report_to wandb
