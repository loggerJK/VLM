#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Activate conda environment
# unset PYTHONPATH
# source /home/work/.project/anaconda3/etc/profile.d/conda.sh
# conda activate lumina_dimoo
export PYTHONNOUSERSITE=1

# Check user_id == cvlab20
if echo $USER | grep -q "cvlab20"; then
    echo "Running on cvlab20"
    export HF_HOME='/mnt/dataset1/huggingface'
elif echo $USER | grep -q "cvlab22"; then
    export HF_HOME='/mnt/data1/huggingface'
fi
echo "HF_HOME is set to $HF_HOME"

# Settings
init_from="Alpha-VLLM/Lumina-DiMOO"
data_config="configs/data.yaml" # Dummy config
lr=5e-5
wd=0.1 # weight decay
epochs=999
batchsize_per_gpu=1
n_gpus=4
accum_iter=32
dropout=0.05
lora_rank=128
image_size=512
task="pointing"
max_seq_len=2048
exp_name="lora128_${task}_qaFixed_wohead"
output_dir="output/$exp_name"
ckpt_max_keep=-1

source .env


export LOCAL_TRAIN_DIR='/mnt/data1/jiwon/data/pixmo-point-count-concat_0-20-qaFixed-final'
export LOCAL_VAL_DIR='/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-count-filtered-imgContained'

mkdir -p "$output_dir"

echo "Starting training verification..."
echo "Output Directory: $output_dir"

# Run with torchrun for 1 GPU
python -m torch.distributed.run --nproc_per_node=${n_gpus} --master_port=29505 train/train_pointing.py \
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
    --image_size 512     \
    --data_parallel none \
    --data_config $data_config \
    --num_workers 4 \
    --output_dir "$output_dir" \
    --save_iteration_interval 100 \
    --validation_interval 100 \
    --max_seq_len ${max_seq_len} \
    --dropout ${dropout} \
    --init_from ${init_from} \
    --disable_length_clustering \
    --use_wandb \
    --wandb_project "lumina-pointing" \
    --wandb_run_name $exp_name \
    --use_lora \
    --lora_rank ${lora_rank} \
    --task ${task} \
    --ckpt_max_keep ${ckpt_max_keep} \
    --wo_lm_head \
    --validation_as_pointing_format  \
    2>&1 | tee "$output_dir/output.log"
