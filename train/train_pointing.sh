#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
unset PYTHONPATH
source /home/cvlab22/anaconda3/etc/profile.d/conda.sh
conda activate lumina_dimoo
export PYTHONNOUSERSITE=1

# Settings
init_from="Alpha-VLLM/Lumina-DiMOO"
data_config="configs/data.yaml" # Dummy config
lr=5e-5
wd=0.1 # weight decay
epochs=999
batchsize_per_gpu=1
n_gpus=1
accum_iter=64
dropout=0.05
lora_rank=128
image_size=512
task="pointing"
max_seq_len=2048
exp_name="lora128_counting_wohead"
output_dir="output/$exp_name"
ckpt_max_keep=-1

# Resume Settings (uncomment to resume from checkpoint)
resume_path="output/$exp_name/epoch0-iter47999-step1500"
wandb_run_id="ffeqqur0"  # WandB run ID to continue same run (find in WandB UI or first run log)

WANDB_API_KEY="f9831e23517e27f7ecac9b54bc2cdcabb3af8c33"


export LOCAL_TRAIN_DIR='/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-point-count-concat_0-20-qaFixed'
export LOCAL_VAL_DIR='/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-count-filtered-imgContained'

mkdir -p "$output_dir"

echo "Starting training verification..."
echo "Output Directory: $output_dir"

# Run with torchrun for 1 GPU
python -m torch.distributed.run --nproc_per_node=${n_gpus} --master_port=29504 train/train_pointing.py \
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
    --save_iteration_interval 500 \
    --validation_interval 100 \
    --max_seq_len ${max_seq_len} \
    --dropout ${dropout} \
    --init_from ${init_from} \
    --disable_length_clustering \
    --use_wandb \
    --wandb_project "lumina-pointing" \
    --wandb_run_name "full" \
    --lora_rank ${lora_rank} \
    --task ${task} \
    --ckpt_max_keep ${ckpt_max_keep} \
    --resume_path ${resume_path} \
    --wandb_run_id ${wandb_run_id} \
    --use_lora \
    2>&1 | tee "$output_dir/output.log"
    # --use_lora \
