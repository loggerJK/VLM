#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

base_dir="/mnt/data1"
model_path="deepseek-ai/Janus-Pro-7B"

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

for prompt_type in word sentence; do
    echo "========================================"
    echo "Running baseline OCR generation evaluation for: ${prompt_type}.txt"
    echo "========================================"

    torchrun --nproc_per_node=$NGPUS \
        evaluation_scripts/ocr/generation/evaluate_ocr_generation_detailed_textfile.py \
        --model_path ${model_path} \
        --prompt_file ${SCRIPT_DIR}/${prompt_type}.txt \
        --output_dir ${base_dir}/dvlm/janus/ocr/generation_detailed_textfile/${prompt_type}/baseline \
        --cfg_weight 5.0 \
        --img_size 384 \
        --ocr_model_path zai-org/GLM-OCR
done
