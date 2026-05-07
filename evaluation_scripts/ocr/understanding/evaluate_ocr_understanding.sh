#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

base_dir="/mnt/data1"
model_path="deepseek-ai/Janus-Pro-7B"

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

output_dir_name="baseline"
torchrun --nproc_per_node=$NGPUS \
    evaluation_scripts/ocr/understanding/evaluate_ocr_understanding.py \
    --model_path ${model_path} \
    --output_dir ${base_dir}/dvlm/janus/ocr/understanding/${output_dir_name} \
    --max_new_tokens 512 \
    --num_samples 200

# ---------------------------------------------------------------------------- #
#                              Checkpoint sweep                                #
# ---------------------------------------------------------------------------- #

# ckpt_list=(
#     # Add Janus checkpoints here...
# )
#
# for ckpt in "${ckpt_list[@]}"; do
#     echo "========================================"
#     echo "Running OCR understanding evaluation for: $ckpt"
#     echo "========================================"
#
#     torchrun --nproc_per_node=$NGPUS \
#         evaluation_scripts/ocr/understanding/evaluate_ocr_understanding.py \
#         --model_path ${ckpt} \
#         --output_dir ${base_dir}/dvlm/janus/ocr/understanding/$(basename ${ckpt}) \
#         --max_new_tokens 512 \
#         --num_samples 200
# done
