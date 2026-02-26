#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

base_dir="/mnt/data1"
# model_path="${base_dir}/jiwon/bagel_train/hf/BAGEL-7B-MoT"
model_path="/mnt/data1/jiwon/BAGEL/models/BAGEL-7B-MoT"
checkpoint_base="${base_dir}/jiwon/bagel_train/checkpoints"
output_base="${base_dir}/dvlm/BAGEL/ocr_synthetic/understanding"

NUM_SAMPLES=250
MAX_NEW_TOKENS=128

# Add bagel_train to PYTHONPATH
export PYTHONPATH=/mnt/data1/jiwon/bagel_train:${PYTHONPATH:-}

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

echo "========================================"
echo "Running baseline OCR synthetic understanding evaluation (BAGEL)"
echo "========================================"


# output_dir_name="baseline"
# torchrun --nproc_per_node=$NGPUS \
#     evaluation_scripts/ocr_synthetic/understanding/evaluate_ocr_understanding.py \
#     --model_path ${model_path} \
#     --output_dir ${output_base}/${output_dir_name} \
#     --max_new_tokens ${MAX_NEW_TOKENS} \
#     --num_samples ${NUM_SAMPLES}

# ---------------------------------------------------------------------------- #
#                              Checkpoint sweep                                #
# ---------------------------------------------------------------------------- #

ckpt_list=(
    # Add your LoRA checkpoint relative paths here, e.g.:
    "lora128_counting_und/epoch1"
    "lora128_counting_und/epoch2"
    "lora128_counting_und/epoch3"
    "epoch0-step100/"
    "epoch0-step200/"
    "epoch0-step300/"
    "epoch0-step400/"
    "epoch0-step500/"
    "epoch0-step600/"
    "epoch0-step700/"
    "epoch1-step1000/"
    "epoch1-step1100/"
    "epoch1-step1200/"
    "epoch1-step1300/"
    "epoch1-step1400/"
    "epoch1-step800/"
    "epoch1-step900/"
    "epoch2-step1500/"
    "epoch2-step1600/"
    "epoch2-step1700/"
    "epoch2-step1800/"
    "epoch2-step1900/"
    "epoch2-step2000/"
    "epoch2-step2100/"
    "epoch2-step2200/"
    "epoch3-step2300/"
    "epoch3-step2400/"
    "epoch3-step2500/"
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running OCR understanding evaluation for checkpoint: $ckpt"
    echo "Output directory: ${output_base}/${ckpt}"
    echo "========================================"

    torchrun --nproc_per_node=$NGPUS \
        evaluation_scripts/ocr_synthetic/understanding/evaluate_ocr_understanding.py \
        --model_path ${model_path} \
        --lora_ckpt_path ${checkpoint_base}/${ckpt} \
        --output_dir ${output_base}/${ckpt} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --num_samples ${NUM_SAMPLES}
done
