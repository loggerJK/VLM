#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# based on CUDA
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') 

base_dir="/mnt/data1"
checkpoint_path="jiwon/Lumina-DiMOO/output"

STEPS=128
GEN_LENGTH=128
BLOCK_LENGTH=128
NUM_SAMPLES=250
output_dir_label="understanding_block128_givefirsttoken"
give_first_token_flag="--give_first_token" 


# ---------------------------------------------------------------------------- #
#                                   baseline                                   #
# ---------------------------------------------------------------------------- #

output_dir_name="baseline"
torchrun --nproc_per_node=$NGPUS \
    evaluation_scripts/ocr_synthetic/understanding/evaluate_ocr_understanding.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir ${base_dir}/dvlm/lumina/ocr_synthetic/${output_dir_label}/${output_dir_name} \
    --steps ${STEPS} \
    --gen_length ${GEN_LENGTH} \
    --block_length ${BLOCK_LENGTH} \
    --num_samples ${NUM_SAMPLES} ${give_first_token_flag}


# ---------------------------------------------------------------------------- #
#                               OCR_Understanding                              #
# ---------------------------------------------------------------------------- #
checkpoint_path="jiwon/Lumina-DiMOO/output"
ckpt_list=(
    # "lora128_ocr_4_images_wohead/epoch0"
    # "lora128_ocr_4_images_wohead/epoch2"
    # "lora128_ocr_4_images_wohead/epoch4"
    # "lora128_ocr_4_images_wohead/epoch6"
    # "lora128_ocr_4_images_wohead/epoch8"
    # "lora128_ocr_4_images_wohead/epoch10"
    # "lora128_ocr_4_images_wohead/epoch11"
    # "lora128_ocr_4_images_wohead/epoch1"
    # "lora128_ocr_4_images_wohead/epoch3"
    # "lora128_ocr_4_images_wohead/epoch5"
    # "lora128_ocr_4_images_wohead/epoch7"
    # "lora128_ocr_4_images_wohead/epoch9"
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
        --output_dir ${base_dir}/dvlm/lumina/ocr_synthetic/${output_dir_label}/${ckpt} \
        --steps ${STEPS} \
        --gen_length ${GEN_LENGTH} \
        --block_length ${BLOCK_LENGTH} \
        --num_samples ${NUM_SAMPLES} ${give_first_token_flag}

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
    echo "Running OCR understanding evaluation for checkpoint: $ckpt"
    echo "Output directory name: $ckpt"
    echo "========================================"

    torchrun --nproc_per_node=$NGPUS \
        evaluation_scripts/ocr_synthetic/understanding/evaluate_ocr_understanding.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/lumina/ocr_synthetic/${output_dir_label}/${ckpt} \
        --steps ${STEPS} \
        --gen_length ${GEN_LENGTH} \
        --block_length ${BLOCK_LENGTH} \
        --num_samples ${NUM_SAMPLES} \
        ${give_first_token_flag}
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
#         --output_dir ${base_dir}/dvlm/lumina/ocr_synthetic/${output_dir_label}/${ckpt} \
#         --steps ${STEPS} \
#         --gen_length ${GEN_LENGTH} \
#         --block_length ${BLOCK_LENGTH} \
#         --num_samples ${NUM_SAMPLES} ${give_first_token_flag}
# done
