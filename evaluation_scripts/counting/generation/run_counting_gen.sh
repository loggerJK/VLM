#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4


# ─────────────────────────────────────────────────────────────────────────────
# Server-specific configuration
# ─────────────────────────────────────────────────────────────────────────────
data_dirname="data1"
MODEL="/mnt/data1/jiwon/BLIP3o/models/BLIP3o-8B"

if echo $USER | grep -q "cvlab17"; then
    echo "Running on cvlab17"
    data_dirname="cvlab22_data1"
    HF_HOME="/home/cvlab17/.cache/huggingface"
    MODEL="/home/cvlab17/models/BLIP3o-8B"
elif echo $USER | grep -q "cvlab22"; then
    echo "Running on cvlab22"
    data_dirname="data1"
    HF_HOME="/mnt/data1/jiwon/.cache/huggingface"
    MODEL="/mnt/data1/jiwon/BLIP3o/models/BLIP3o-8B"
elif echo $USER | grep -q "cvlab12"; then
    echo "Running on cvlab12"
    MODEL="/home/cvlab12/BLIP3o/BLIP3o-Model-8B"
elif echo $USER | grep -q "cvlab15"; then
    echo "Running on cvlab15"
    data_dirname="cvlab22_data1"
    HF_HOME="/mnt/data1/.cache/huggingface"
    MODEL="/mnt/data1/models/BLIP3o-8B"
fi

export HF_HOME
echo "data_dirname is set to $data_dirname"
echo "MODEL is set to $MODEL"

base_dir="/mnt/$data_dirname"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METADATA_FILE="${SCRIPT_DIR}/evaluation_metadata_count.jsonl"
output_base="${base_dir}/dvlm/blip3o/counting/generation"

# torchrun configuration
ngpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29500"}

# ─────────────────────────────────────────────────────────────────────────────
# Baseline
# ─────────────────────────────────────────────────────────────────────────────
# echo "========================================"
# echo "Running counting generation (baseline)"
# echo "Output: ${output_base}/baseline"
# echo "========================================"

# torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
#     "${SCRIPT_DIR}/generate_counting.py" \
#     --model "$MODEL" \
#     --output_dir "${output_base}/baseline" \
#     --metadata_file "$METADATA_FILE" \
#     --n_samples 4

# echo "Baseline generation finished."


# ─────────────────────────────────────────────────────────────────────────────
# Understanding Checkpoints
# ─────────────────────────────────────────────────────────────────────────────
checkpoint_base="${base_dir}/dvlm/blip3o/checkpoints"
ckpt_list=(
    "counting_und/epoch1"
    # "counting_und/epoch2"
    # "counting_und/epoch3"
    # "counting_und/epoch4"
    # "counting_und/epoch5"

    # "counting_und/epoch0_step-500"
    # "counting_und/epoch0_step-1000"
    # "counting_und/epoch0_step-1500"
    # "counting_und/epoch1_step-2000"
    # "counting_und/epoch1_step-2500"
    # "counting_und/epoch1_step-3000"
    # "counting_und/epoch1_step-3500"
    # "counting_und/epoch2_step-4000"
    # "counting_und/epoch2_step-4500"
    # "counting_und/epoch2_step-5000"
    # "counting_und/epoch3_step-5500"
    # "counting_und/epoch3_step-6000"
    # "counting_und/epoch3_step-6500"
    # "counting_und/epoch3_step-7000"
    # "counting_und/epoch4_step-7500"
    # "counting_und/epoch4_step-8000"
    # "counting_und/epoch4_step-8500"
    # "counting_und/epoch5_step-9000"
    # "counting_und/epoch5_step-9500"
    # "counting_und/epoch5_step-10000"
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running counting generation for checkpoint: $ckpt"
    echo "Output: ${output_base}/${ckpt}"
    echo "========================================"

    torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        "${SCRIPT_DIR}/generate_counting.py" \
        --model "${checkpoint_base}/${ckpt}" \
        --base_model "$MODEL" \
        --output_dir "${output_base}/${ckpt}" \
        --metadata_file "$METADATA_FILE" \
        --n_samples 4 \
        --lora_mode "und"

    echo "Checkpoint $ckpt finished."
done

echo "All generation finished."


# ─────────────────────────────────────────────────────────────────────────────
# Generation Checkpoints
# ─────────────────────────────────────────────────────────────────────────────
checkpoint_base="${base_dir}/jiwon/BLIP3o/output"
ckpt_list=(
    # "counting_gen/epoch0"
    # "counting_gen/epoch1"
    # "counting_gen/epoch2"
    # "counting_gen/epoch3"
    # "counting_gen/epoch4"
    # "counting_gen/epoch5"
    # "counting_gen/epoch6"
    # "counting_gen/epoch7"
    # "counting_gen/epoch8"
    # "counting_gen/epoch9"
    # "counting_gen/epoch10"
    # "counting_gen/epoch0_step-250"
    # "counting_gen/epoch0_step-500"
    # "counting_gen/epoch1_step-750"
    # "counting_gen/epoch1_step-1000"
    # "counting_gen/epoch2_step-1250"
    # "counting_gen/epoch2_step-1500"
    # "counting_gen/epoch3_step-1750"
    # "counting_gen/epoch3_step-2000"
    # "counting_gen/epoch4_step-2250"
    # "counting_gen/epoch4_step-2500"
    # "counting_gen/epoch5_step-2750"
    # "counting_gen/epoch5_step-3000"
    # "counting_gen/epoch6_step-3250"
    # "counting_gen/epoch6_step-3500"
    # "counting_gen/epoch7_step-3750"
    # "counting_gen/epoch7_step-4000"
    
    
    # "counting_gen/epoch8_step-4250"
    # "counting_gen/epoch8_step-4500"
    # "counting_gen/epoch9_step-4750"
    # "counting_gen/epoch9_step-5000"
    # "counting_gen/epoch10_step-5250"
    # "counting_gen/epoch10_step-5500"
    # "counting_gen/epoch11_step-5750"
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running counting generation for checkpoint: $ckpt"
    echo "Output: ${output_base}/${ckpt}"
    echo "========================================"

    torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        "${SCRIPT_DIR}/generate_counting.py" \
        --model "${checkpoint_base}/${ckpt}" \
        --base_model "$MODEL" \
        --output_dir "${output_base}/${ckpt}" \
        --metadata_file "$METADATA_FILE" \
        --n_samples 4

    echo "Checkpoint $ckpt finished."
done

echo "All generation finished."
