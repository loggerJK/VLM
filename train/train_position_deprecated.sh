#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Activate conda environment
# unset PYTHONPATH
# source /home/work/.project/anaconda3/etc/profile.d/conda.sh
# conda activate lumina_dimoo
export PYTHONNOUSERSITE=1
source .env
export WANDB_PROJECT=lumina-position-exp

# Settings
init_from="Alpha-VLLM/Lumina-DiMOO"
data_config="configs/data.yaml" # Dummy config
lr=2e-5
wd=0.1 # weight decay
epochs=999
batchsize_per_gpu=1
n_gpus=4
accum_iter=32
dropout=0.05
lora_rank=128
gen_image_size=1024
und_image_size=512
task="spatial_relation"
mode="und" # gen / und / both
max_seq_len=5120
exp_name="lora128_${task}_wohead_${mode}-$(date +%Y%m%d-%H%M%S)"
output_dir="/mnt/cvlab22_data1/heeji/dVLM_outputs_from_ext_srv/$exp_name"
ckpt_max_keep=-1

# export WANDB_RUN_ID="iznvrq84"


# export LOCAL_TRAIN_DIR='/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-point-count-concat_0-20-qaFixed-final'
# export LOCAL_VAL_DIR='/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-count-filtered-imgContained'

mkdir -p "$output_dir"

echo "Starting training verification..."
echo "Output Directory: $output_dir"

# Run with torchrun for 1 GPU
python -m torch.distributed.run --nproc_per_node=${n_gpus} --master_port=29504 train/train_position.py \
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
    --gen_image_size ${gen_image_size} \
    --und_image_size ${und_image_size} \
    --data_parallel none \
    --data_config $data_config \
    --num_workers 16 \
    --output_dir "$output_dir" \
    --save_iteration_interval 500 \
    --validation_interval 100 \
    --max_seq_len ${max_seq_len} \
    --dropout ${dropout} \
    --init_from ${init_from} \
    --disable_length_clustering \
    --use_wandb \
    --wandb_run_name $exp_name \
    --use_lora \
    --lora_rank ${lora_rank} \
    --task ${task} \
    --mode ${mode} \
    --wo_lm_head \
    --ckpt_max_keep ${ckpt_max_keep} \
    --wandb_project "$WANDB_PROJECT" \
    2>&1 | tee "$output_dir/output.log"
    # --resume_path "/home/jovyan/viral-3dvlm/dvlm/lumina_dimoo/output/lora128_counting_wohead_both/epoch0-iter31999-step1000" \
