#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=25001
ngpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')

base_dir="/mnt/data1"
model_path="${base_dir}/jiwon/bagel_train/hf/BAGEL-7B-MoT"
checkpoint_base="${base_dir}/jiwon/bagel_train/checkpoints"
output_base="${base_dir}/dvlm/BAGEL/counting_pixmo/understanding_eval"

MAX_NEW_TOKENS=32

# Add bagel_train to PYTHONPATH
export PYTHONPATH=/mnt/data1/jiwon/bagel_train:${PYTHONPATH:-}

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

echo "========================================"
echo "Running baseline Pixmo counting evaluation (BAGEL)"
echo "========================================"

torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    evaluation_scripts/counting/evaluate_pixmo_multigpu.py \
    --model_path ${model_path} \
    --output_dir ${output_base}/baseline \
    --max_new_tokens ${MAX_NEW_TOKENS}

# ---------------------------------------------------------------------------- #
#                              Checkpoint sweep                                #
# ---------------------------------------------------------------------------- #

ckpt_list=(
    # Add your LoRA checkpoint relative paths here, e.g.:
    # "lora128_counting_und/epoch1"
    # "lora128_counting_und/epoch2"
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running Pixmo counting evaluation for checkpoint: $ckpt"
    echo "Output directory: ${output_base}/${ckpt}"
    echo "========================================"

    torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        evaluation_scripts/counting/evaluate_pixmo_multigpu.py \
        --model_path ${model_path} \
        --lora_ckpt_path ${checkpoint_base}/${ckpt} \
        --output_dir ${output_base}/${ckpt} \
        --max_new_tokens ${MAX_NEW_TOKENS}
done
