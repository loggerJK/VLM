#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

base_dir="/mnt/data1"
model_path="${base_dir}/jiwon/bagel_train/hf/BAGEL-7B-MoT"
checkpoint_base="${base_dir}/jiwon/bagel_train/checkpoints"

# Add bagel_train to PYTHONPATH
export PYTHONPATH=/mnt/data1/jiwon/bagel_train:${PYTHONPATH:-}

SCRIPT_DIR="evaluation_scripts/ocr_synthetic/generation"

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

echo "========================================"
echo "Running baseline OCR generation evaluation for quotes dataset (BAGEL)"
echo "========================================"

output_base="${base_dir}/dvlm/BAGEL/ocr_synthetic/generation"

# Stage 1: BAGEL image generation (bagel env)
conda run -n bagel --no-banner torchrun --nproc_per_node=$NGPUS \
    ${SCRIPT_DIR}/stage1_generate_images.py \
    --model_path ${model_path} \
    --output_dir ${output_base}/baseline \
    --cfg_scale 4.0 \
    --img_size 512

# Stage 2: OCR extraction + metrics (glm_ocr env)
conda run -n glm_ocr --no-banner torchrun --nproc_per_node=$NGPUS \
    ${SCRIPT_DIR}/stage2_ocr_and_metrics.py \
    --output_dir ${output_base}/baseline \
    --ocr_model_path zai-org/GLM-OCR

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
    echo "Running OCR generation evaluation for checkpoint: $ckpt"
    echo "Output directory: ${output_base}/${ckpt}"
    echo "========================================"

    # Stage 1: BAGEL image generation (bagel env)
    conda run -n bagel --no-banner torchrun --nproc_per_node=$NGPUS \
        ${SCRIPT_DIR}/stage1_generate_images.py \
        --model_path ${model_path} \
        --lora_ckpt_path ${checkpoint_base}/${ckpt} \
        --output_dir ${output_base}/${ckpt} \
        --cfg_scale 4.0 \
        --img_size 512

    # Stage 2: OCR extraction + metrics (glm_ocr env)
    conda run -n glm_ocr --no-banner torchrun --nproc_per_node=$NGPUS \
        ${SCRIPT_DIR}/stage2_ocr_and_metrics.py \
        --output_dir ${output_base}/${ckpt} \
        --ocr_model_path zai-org/GLM-OCR
done
