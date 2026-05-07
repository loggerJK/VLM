#!/bin/bash
set -e

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

cd /data/mm-llm-backbone_890/personal/sirius/audio_ablation/audio_lumina
source ./env.sh

# Settings
init_from="Alpha-VLLM/Lumina-DiMOO"
data_config="configs/data.yaml"
task="counting+rel_position"
mode="und"
lr=1e-5
wd=0.1
epochs=999
batchsize_per_gpu=1
n_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
accum_iter=32
dropout=0.05
lora_rank=128
max_seq_len=2048
validation_samples=100
exp_name="lora128_counting_rel_position_und_wohead"
output_dir="output/$exp_name"
ckpt_max_keep=-1

mkdir -p "$output_dir"

echo "Starting Counting + Relative Position mixed understanding training..."
echo "Output Directory: $output_dir"

python -m torch.distributed.run \
    --nproc_per_node=${n_gpus} \
    --master_port=29506 \
    train/train_unified.py \
    --task ${task} \
    --mode ${mode} \
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
    --num_workers 2 \
    --output_dir "$output_dir" \
    --save_iteration_interval 500 \
    --validation_interval 500 \
    --validation_samples ${validation_samples} \
    --max_seq_len ${max_seq_len} \
    --dropout ${dropout} \
    --init_from ${init_from} \
    --disable_length_clustering \
    --use_wandb \
    --wandb_project "lumina-pointing" \
    --wandb_run_name $exp_name \
    --use_lora \
    --lora_rank ${lora_rank} \
    --ckpt_max_keep ${ckpt_max_keep} \
    --wo_lm_head \
    2>&1 | tee "$output_dir/output.log"
