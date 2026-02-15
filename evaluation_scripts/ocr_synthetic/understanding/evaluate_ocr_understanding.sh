#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# based on CUDA
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') 

base_dir="/mnt/data1"
checkpoint_path="jiwon/Lumina-DiMOO/output"


# ---------------------------------------------------------------------------- #
#                                   baseline                                   #
# ---------------------------------------------------------------------------- #

output_dir_name="baseline"
torchrun --nproc_per_node=$NGPUS \
    evaluation_scripts/ocr_synthetic/understanding/evaluate_ocr_understanding.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir ${base_dir}/dvlm/lumina/ocr_synthetic/understanding_block128/${output_dir_name} \
    --steps 128 \
    --gen_length 128 \
    --block_length 128 \
    --num_samples 200


# ---------------------------------------------------------------------------- #
#                               OCR_Understanding                              #
# ---------------------------------------------------------------------------- #

ckpt_list=(
    # "lora128_ocr_4_images_wohead/epoch0"
    # "lora128_ocr_4_images_wohead/epoch2"
    # "lora128_ocr_4_images_wohead/epoch4"
    # "lora128_ocr_4_images_wohead/epoch6"
    # "lora128_ocr_4_images_wohead/epoch8"
    # "lora128_ocr_4_images_wohead/epoch10"
    "lora128_ocr_4_images_wohead/epoch11"
    # "lora128_ocr_4_images_wohead/epoch1"
    # "lora128_ocr_4_images_wohead/epoch3"
    # "lora128_ocr_4_images_wohead/epoch5"
    # "lora128_ocr_4_images_wohead/epoch7"
    # "lora128_ocr_4_images_wohead/epoch9"

    # Add more checkpoints here...
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running OCR understanding evaluation for checkpoint: $ckpt"
    echo "Output directory name: $ckpt"
    echo "========================================"

    torchrun --nproc_per_node=$NGPUS \
        evaluation_scripts/ocr_synthetic/understanding/evaluate_ocr_understanding.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/lumina/ocr_synthetic/understanding_block128/${ckpt} \
        --steps 128 \
        --gen_length 128 \
        --block_length 128 \
        --num_samples 200
done


# ---------------------------------------------------------------------------- #
#                                OCR Generation                                #
# ---------------------------------------------------------------------------- #

# checkpoint_path="dvlm/lumina/checkpoints"

# ckpt_list=(
#     # Add more checkpoints here...
#     "lora128_ocr_gen_wohead/epoch0"
#     "lora128_ocr_gen_wohead/epoch1"
#     "lora128_ocr_gen_wohead/epoch2"
#     "lora128_ocr_gen_wohead/epoch3"
# )


# for ckpt in "${ckpt_list[@]}"; do
#     echo "========================================"
#     echo "Running OCR understanding evaluation for checkpoint: $ckpt"
#     echo "Output directory name: $ckpt"
#     echo "========================================"

#     torchrun --nproc_per_node=$NGPUS \
#         evaluation_scripts/ocr_synthetic/understanding/evaluate_ocr_understanding.py \
#         --checkpoint Alpha-VLLM/Lumina-DiMOO \
#         --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#         --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
#         --output_dir ${base_dir}/dvlm/lumina/ocr_synthetic/understanding_block128/${ckpt} \
#         --steps 512 \
#         --gen_length 512 \
#         --block_length 512 \
#         --num_samples 200
# done
