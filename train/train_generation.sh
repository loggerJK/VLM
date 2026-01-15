#!/bin/bash
# set -e
# Default GPU
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONNOUSERSITE=1
export WANDB_API_KEY=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
export WANDB_PROJECT=lumina-generation-debug    

# Arguments
DATASET_NAME="heez/pixmo-point-count-desc-all"
if [ -z "$DATASET_NAME" ]; then
    echo "Error: Dataset name is required."
    echo "Usage: bash train/train_generation.sh <dataset_name> [other args...]"
    echo "Example: bash train/train_generation.sh 'lambdalabs/pokemon-blip-captions' --use_lora"
    exit 1
fi
shift # Shift arguments to pass the rest (e.g., --use_lora) to the python script

# Settings
init_from="Alpha-VLLM/Lumina-DiMOO"
data_config="configs/data.yaml" # Dummy
lr=2e-4
wd=0.01
batchsize_per_gpu=1
max_seq_len=5120
exp_name="lumina_gen_512_lora128-$(date +%Y%m%d-%H%M%S)"
output_dir="output/$exp_name"
lora_rank=128

mkdir -p "$output_dir"

echo "Starting training..."
echo "Dataset: $DATASET_NAME"
echo "Output: $output_dir"
echo "Extra Args: $@"

/home/cvlab22/anaconda3/envs/lumina_dimoo/bin/python -m torch.distributed.run --nproc_per_node=2 --master_port=29508 train/train_generation.py \
    --dataset_name "$DATASET_NAME" \
    --batch_size ${batchsize_per_gpu} \
    --accum_iter 64 \
    --epochs 999 \
    --warmup_epochs 0 \
    --lr ${lr} \
    --min_lr 1e-6 \
    --wd ${wd} \
    --clip_grad 1.0 \
    --precision bf16 \
    --image_size 512 \
    --data_parallel fsdp \
    --checkpointing \
    --data_config $data_config \
    --num_workers 4 \
    --output_dir "$output_dir" \
    --max_seq_len ${max_seq_len} \
    --init_from ${init_from} \
    --disable_length_clustering \
    --use_wandb \
    --lora_rank ${lora_rank} \
    --wandb_project "lumina-generation" \
    --wandb_run_name ${exp_name} \
    --validation_interval 100 \
    --save_iteration_interval 500 \
    "$@" \
    2>&1 | tee "$output_dir/output.log"