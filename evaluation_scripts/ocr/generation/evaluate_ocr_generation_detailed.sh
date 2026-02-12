#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') 


base_dir="/mnt/data1"
checkpoint_path="jiwon/Lumina-DiMOO/output"

# ---------------------------------------------------------------------------- #
#                                   baseline                                   #
# ---------------------------------------------------------------------------- #

output_dir_name="baseline"
torchrun --nproc_per_node=$NGPUS \
    evaluation_scripts/ocr/generation/evaluate_ocr_generation_detailed.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir ${base_dir}/dvlm/ocr/generation_detailed_prompt/${output_dir_name} \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --height 1024 \
    --width 1024 \
    --ocr_model_path zai-org/GLM-OCR \
    --num_samples 200


# ---------------------------------------------------------------------------- #
#                               OCR_Understanding                              #
# ---------------------------------------------------------------------------- #

ckpt_list=(
    "lora128_ocr_4_images_wohead/epoch0"
    "lora128_ocr_4_images_wohead/epoch1"
    "lora128_ocr_4_images_wohead/epoch2"
    "lora128_ocr_4_images_wohead/epoch3"
    "lora128_ocr_4_images_wohead/epoch4"
    "lora128_ocr_4_images_wohead/epoch5"
    "lora128_ocr_4_images_wohead/epoch6"
    "lora128_ocr_4_images_wohead/epoch7"
    # Add more checkpoints here...
)

for ckpt in "${ckpt_list[@]}"; do
    output_dir_name=$(echo "$ckpt" | tr '/' '_')
    echo "========================================"
    echo "Running OCR generation evaluation for checkpoint: $ckpt"
    echo "Output directory name: $output_dir_name"
    echo "========================================"

    torchrun --nproc_per_node=$NGPUS \
        evaluation_scripts/ocr/generation/evaluate_ocr_generation_detailed.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/ocr/generation_detailed_prompt/${output_dir_name} \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --height 1024 \
        --width 1024 \
        --ocr_model_path zai-org/GLM-OCR \
        --num_samples 200

done
