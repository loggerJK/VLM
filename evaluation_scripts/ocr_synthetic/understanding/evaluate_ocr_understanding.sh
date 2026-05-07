#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

base_dir="/mnt/data1"
model_path="deepseek-ai/Janus-Pro-7B"

NUM_SAMPLES=250
MAX_NEW_TOKENS=128

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

# output_dir_name="baseline"
# torchrun --nproc_per_node=$NGPUS \
#     evaluation_scripts/ocr_synthetic/understanding/evaluate_ocr_understanding.py \
#     --model_path ${model_path} \
#     --output_dir ${base_dir}/dvlm/janus/ocr_synthetic/understanding/${output_dir_name} \
#     --max_new_tokens ${MAX_NEW_TOKENS} \
#     --num_samples ${NUM_SAMPLES}

# ---------------------------------------------------------------------------- #
#                              Checkpoint sweep                                #
# ---------------------------------------------------------------------------- #

lora_ckpt_base="/mnt/data1/dvlm/janus/checkpoints"
ckpt_list=(
    # Add Janus checkpoints here...
    "train[transformer_lora128]_task[ocr]_mode[gen]/epoch0_step-0"
    "train[transformer_lora128]_task[ocr]_mode[gen]/epoch0_step-500"
    "train[transformer_lora128]_task[ocr]_mode[gen]/epoch0_step-1000"
    "train[transformer_lora128]_task[ocr]_mode[gen]/epoch0_step-1500"
    "train[transformer_lora128]_task[ocr]_mode[gen]/epoch1_step-2000"
    "train[transformer_lora128]_task[ocr]_mode[gen]/epoch1_step-2500"
    "train[transformer_lora128]_task[ocr]_mode[gen]/epoch1_step-3000"
    "train[transformer_lora128]_task[ocr]_mode[gen]/epoch2_step-3500"
    "train[transformer_lora128]_task[ocr]_mode[gen]/epoch2_step-4000"
    "train[transformer_lora128]_task[ocr]_mode[gen]/epoch2_step-4500"
    "train[transformer_lora128]_task[ocr]_mode[gen]/epoch3_step-5000"
    "train[transformer_lora128]_task[ocr]_mode[gen]/epoch3_step-5500"
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running OCR synthetic understanding evaluation for: $ckpt"
    echo "========================================"

    torchrun --nproc_per_node=$NGPUS \
        evaluation_scripts/ocr_synthetic/understanding/evaluate_ocr_understanding.py \
        --model_path ${model_path} \
        --lora_ckpt_path ${lora_ckpt_base}/${ckpt} \
        --output_dir ${base_dir}/dvlm/janus/ocr_synthetic/understanding/${ckpt} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --num_samples ${NUM_SAMPLES}
done
