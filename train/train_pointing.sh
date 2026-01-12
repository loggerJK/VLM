#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=6

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null
conda activate lumina_dimoo

# Settings
init_from="Alpha-VLLM/Lumina-DiMOO"
data_config="configs/data.yaml" # Dummy config
lr=2e-5
wd=0.1
dropout=0.05
batchsize_per_gpu=1
max_seq_len=1024
exp_name="Lumina-DiMOO-Pointing-Test"
output_dir="output/$exp_name"

mkdir -p "$output_dir"

echo "Starting training verification..."
echo "Output Directory: $output_dir"

# Run with torchrun for 1 GPU
torchrun --nproc_per_node=1 --master_port=29504 train/train_pointing.py \
    --batch_size ${batchsize_per_gpu} \
    --accum_iter 4 \
    --epochs 1 \
    --warmup_epochs 0 \
    --lr ${lr} \
    --min_lr ${lr} \
    --wd ${wd} \
    --clip_grad 1.0 \
    --precision bf16 \
    --image_size 256 \
    --data_parallel fsdp \
    --checkpointing \
    --data_config $data_config \
    --num_workers 4 \
    --output_dir "$output_dir" \
    --save_iteration_interval 50 \
    --max_seq_len ${max_seq_len} \
    --dropout ${dropout} \
    --init_from ${init_from} \
    --disable_length_clustering \
    --use_wandb \
    --wandb_project "lumina-pointing-debug" \
    --wandb_run_name "single_gpu_run" 
    2>&1 | tee "$output_dir/output.log"
    # --cpu_offload 