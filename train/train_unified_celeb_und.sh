#!/bin/bash
set -e

# 사용할 GPU 지정 (4,5,6,7)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 사용자 아이디에 따른 HF_HOME 설정
if echo $USER | grep -q "cvlab20"; then
    echo "Running on cvlab20"
    export HF_HOME='/mnt/dataset1/huggingface'
elif echo $USER | grep -q "cvlab22"; then
    export HF_HOME='/mnt/data1/huggingface'
fi
echo "HF_HOME is set to $HF_HOME"

source ./.env

# Settings
init_from="Alpha-VLLM/Lumina-DiMOO"
data_config="configs/data.yaml"
lr=1e-5
wd=0.1
epochs=999
batchsize_per_gpu=1
n_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
accum_iter=32
dropout=0.05
lora_rank=128
max_seq_len=5120
exp_name="lora128_celeb_und_wohead"
output_dir="output/$exp_name"
ckpt_max_keep=-1

# WandB 설정
source .env

mkdir -p "$output_dir"

echo "Starting Celeb Und training (unified)..."
echo "Output Directory: $output_dir"

# Torchrun 실행
python -m torch.distributed.run \
    --nproc_per_node=${n_gpus} \
    --master_port=29506 \
    train/train_unified.py \
    --task celeb \
    --mode und \
    --batch_size ${batchsize_per_gpu} \
    --accum_iter ${accum_iter} \
    --epochs ${epochs} \
    --warmup_epochs 0 \
    --lr ${lr} \
    --min_lr ${lr} \
    --wd ${wd} \
    --clip_grad 1.0 \
    --precision bf16 \
    --grad_precision bf16 \
    --und_image_size 512 \
    --gen_image_size 512 \
    --data_parallel none \
    --data_config $data_config \
    --num_workers 2 \
    --output_dir "$output_dir" \
    --save_iteration_interval 500 \
    --validation_interval 500 \
    --max_seq_len ${max_seq_len} \
    --dropout ${dropout} \
    --init_from ${init_from} \
    --disable_length_clustering \
    --use_wandb \
    --wandb_project "lumina-celeb" \
    --wandb_run_name $exp_name \
    --use_lora \
    --lora_rank ${lora_rank} \
    --ckpt_max_keep ${ckpt_max_keep} \
    --wo_lm_head \
    --eval_everything \
    2>&1 | tee "$output_dir/output.log"
