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
    evaluation_scripts/ocr/generation/evaluate_ocr_generation_detailed_book.py \
    --model_path ${model_path} \
    --output_dir ${base_dir}/dvlm/janus/ocr/generation_detailed_book/${output_dir_name} \
    --cfg_weight 5.0 \
    --img_size 384 \
    --ocr_model_path zai-org/GLM-OCR \
    --num_samples 200
