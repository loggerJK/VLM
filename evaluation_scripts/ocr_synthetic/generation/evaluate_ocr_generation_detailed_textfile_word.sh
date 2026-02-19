#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

base_dir="/mnt/data1"
checkpoint_path="jiwon/Lumina-DiMOO/output"
NUM_SAMPLES=250
prompt_type="word"  # or "word"
OUTPUT_BASE="${base_dir}/dvlm/lumina/ocr_synthetic/generation_${prompt_type}"

run_eval() {
    local output_dir="$1"
    shift  # remaining args passed through (e.g. --lora_ckpt_path ...)

    torchrun --nproc_per_node=$NGPUS \
        ${SCRIPT_DIR}/evaluate_ocr_generation_detailed_textfile.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --output_dir "${output_dir}" \
        --prompt_file ${SCRIPT_DIR}/${prompt_type}.txt \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --height 1024 \
        --width 1024 \
        --ocr_model_path zai-org/GLM-OCR \
        --num_samples ${NUM_SAMPLES} \
        "$@"
}

# ---------------------------------------------------------------------------- #
#                                   baseline                                   #
# ---------------------------------------------------------------------------- #

echo "========================================"
echo "Running baseline OCR generation evaluation for ${prompt_type} dataset"
echo "========================================"

run_eval "${OUTPUT_BASE}/baseline"

# ---------------------------------------------------------------------------- #
#                               OCR_Understanding                              #
# ---------------------------------------------------------------------------- #

ckpt_list=(
    # "lora128_ocr_4_images_wohead/epoch0"
    # "lora128_ocr_4_images_wohead/epoch1"
    # "lora128_ocr_4_images_wohead/epoch2"
    # "lora128_ocr_4_images_wohead/epoch3"
    # "lora128_ocr_4_images_wohead/epoch4"
    # "lora128_ocr_4_images_wohead/epoch5"
    # "lora128_ocr_4_images_wohead/epoch6"
    # "lora128_ocr_4_images_wohead/epoch7"
    # "lora128_ocr_4_images_wohead/epoch8"
    # "lora128_ocr_4_images_wohead/epoch9"
    # "lora128_ocr_4_images_wohead/epoch10"
    # "lora128_ocr_4_images_wohead/epoch11"
    # "lora128_ocr_4_images_wohead/epoch12"
    # "lora128_ocr_4_images_wohead/epoch13"
    # "lora128_ocr_4_images_wohead/epoch14"
    # "lora128_ocr_4_images_wohead/epoch15"
    # "lora128_ocr_4_images_wohead/epoch16"
    # "lora128_ocr_4_images_wohead/epoch17"
    # "lora128_ocr_4_images_wohead/epoch18"
    # Add more checkpoints here...
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running OCR generation evaluation for checkpoint: $ckpt on ${prompt_type} dataset"
    echo "========================================"

    run_eval "${OUTPUT_BASE}/${ckpt}" --lora_ckpt_path "${base_dir}/${checkpoint_path}/${ckpt}"

done

checkpoint_path="dvlm/lumina/checkpoints"
ckpt_list=(
    "lora128_ocr_synthetic_wohead/epoch0"
    "lora128_ocr_synthetic_wohead/epoch1"
    "lora128_ocr_synthetic_wohead/epoch2"
    "lora128_ocr_synthetic_wohead/epoch3"
    "lora128_ocr_synthetic_wohead/epoch4"
    "lora128_ocr_synthetic_wohead/epoch5"
    "lora128_ocr_synthetic_wohead/epoch6"
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running OCR generation evaluation for checkpoint: $ckpt on ${prompt_type} dataset"
    echo "========================================"

    run_eval "${OUTPUT_BASE}/${ckpt}" --lora_ckpt_path "${base_dir}/${checkpoint_path}/${ckpt}"
done
