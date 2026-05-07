#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1

# Activate conda environment
# unset PYTHONPATH
# source /home/work/.project/anaconda3/etc/profile.d/conda.sh
# conda activate lumina_dimoo

# Settings
init_from="Alpha-VLLM/Lumina-DiMOO"
data_config="configs/data.yaml" # Dummy config
lr=5e-5 
wd=0.1 # weight decay
epochs=13
batchsize_per_gpu=4
n_gpus=1
accum_iter=32
dropout=0.05
lora_rank=32
und_image_size=512
task="counting"
mode="und" # gen / und / both
max_seq_len=2048
# exp_name="lora128_${task}_wohead_resumeEpoch12_lr3e-6"
exp_name="lora${lora_rank}_${task}_wohead"
output_dir="output/$exp_name"
ckpt_max_keep=-1

# source .env
# export WANDB_RUN_ID="ffeqqur0"
unset WANDB_RUN_ID



# export LOCAL_TRAIN_DIR='/mnt/data1/jiwon/data/pixmo-point-count-concat_0-20-qaFixed-final'
# export LOCAL_VAL_DIR='/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-count-filtered-imgContained'

mkdir -p "$output_dir"

echo "Starting training verification..."
echo "Output Directory: $output_dir"

# Run with torchrun for 1 GPU
echo "Using python : $(which python)"
python -m torch.distributed.run --nproc_per_node=${n_gpus} --master_port=29510 train/train_pointing.py \
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
    --und_image_size ${und_image_size} \
    --data_parallel none \
    --data_config $data_config \
    --num_workers 16 \
    --mode ${mode} \
    --output_dir "$output_dir" \
    --save_iteration_interval 500 \
    --validation_interval 5 \
    --max_seq_len ${max_seq_len} \
    --dropout ${dropout} \
    --init_from ${init_from} \
    --disable_length_clustering \
    --use_wandb \
    --wandb_project "lumina-pointing" \
    --wandb_run_name "$exp_name" \
    --use_lora \
    --lora_rank ${lora_rank} \
    --task ${task} \
    --wo_lm_head \
    --ckpt_max_keep ${ckpt_max_keep} \
    --skip_initial_validation \
    2>&1 | tee "$output_dir/output.log"
    # --resume_path "/mnt/data1/jiwon/Lumina-DiMOO/output/lora128_counting_wohead/epoch12" \
    # --wandb_run_id ${WANDB_RUN_ID} \
