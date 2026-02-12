#!/bin/bash
set -e

# 사용할 GPU 지정 (4,5,6,7)
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONNOUSERSITE=1

# 사용자 아이디에 따른 HF_HOME 설정
if echo $USER | grep -q "cvlab20"; then
    echo "Running on cvlab20"
    export HF_HOME='/mnt/dataset1/huggingface'
elif echo $USER | grep -q "cvlab22"; then
    export HF_HOME='/mnt/data1/huggingface'
fi
echo "HF_HOME is set to $HF_HOME"

# Settings (train_pointing_lora.sh 디폴트값 반영)
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
max_seq_len=2048
exp_name="lora128_ocr_4_images_wohead"
output_dir="output/$exp_name"
ckpt_max_keep=-1

export WANDB_RUN_ID="6timxt2a"
# unset WANDB_RUN_ID

# WandB 설정
source .env

mkdir -p "$output_dir"

echo "Starting OCR training..."
echo "Output Directory: $output_dir"

# Torchrun 실행
/home/cvlab22/anaconda3/envs/lumina_dimoo/bin/python -m torch.distributed.run \
    --nproc_per_node=${n_gpus} \
    --master_port=29506 \
    train/train_unified.py \
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
    --data_parallel none \
    --data_config $data_config \
    --num_workers 4 \
    --output_dir "$output_dir" \
    --save_iteration_interval 500 \
    --validation_interval 500 \
    --max_seq_len ${max_seq_len} \
    --dropout ${dropout} \
    --init_from ${init_from} \
    --disable_length_clustering \
    --use_wandb \
    --wandb_project "lumina-ocr" \
    --wandb_run_name $exp_name \
    --use_lora \
    --lora_rank ${lora_rank} \
    --ckpt_max_keep ${ckpt_max_keep} \
    --wo_lm_head \
    --dataset_path "Jiwon-Kang/Llama-Nemotron-VLM-Dataset-v1-OCR4" \
    --wandb_run_id $WANDB_RUN_ID \
    --resume_path "/mnt/data1/jiwon/Lumina-DiMOO/output/lora128_ocr_4_images_wohead/epoch9-iter10079-step13500" \
    --task 'ocr' \
    --mode 'und' \
    2>&1 | tee "$output_dir/output.log"
