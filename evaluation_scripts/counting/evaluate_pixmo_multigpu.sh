#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=25001
ngpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')

# 사용자 아이디에 따른 HF_HOME 설정
if echo $USER | grep -q "cvlab17"; then
    echo "Running on cvlab17"
    data_dirname="cvlab22_data1"
fi
echo "data_dirname is set to $data_dirname"

base_dir="/mnt/$data_dirname"
# model_path="${base_dir}/jiwon/bagel_train/hf/BAGEL-7B-MoT"
model_path="/home/cvlab17/models/BAGEL-7B-MoT"
checkpoint_base="${base_dir}/jiwon/bagel_train/checkpoints"
output_base="${base_dir}/dvlm/BAGEL/counting_pixmo/understanding_eval"

MAX_NEW_TOKENS=32

# Add bagel_train to PYTHONPATH
export PYTHONPATH=/mnt/data1/jiwon/bagel_train:${PYTHONPATH:-}

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

echo "========================================"
echo "Running baseline Pixmo counting evaluation (BAGEL)"
echo "========================================"

torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    evaluation_scripts/counting/evaluate_pixmo_multigpu.py \
    --model_path ${model_path} \
    --output_dir ${output_base}/baseline \
    --max_new_tokens ${MAX_NEW_TOKENS}

# ---------------------------------------------------------------------------- #
#                              Checkpoint sweep                                #
# ---------------------------------------------------------------------------- #

ckpt_list=(
    # Add your LoRA checkpoint relative paths here, e.g.:

    "lora128_counting_und/epoch1"
    "lora128_counting_und/epoch2"
    "lora128_counting_und/epoch3"


    "lora128_counting_und/epoch0-step100"
    "lora128_counting_und/epoch0-step200"
    "lora128_counting_und/epoch0-step300"
    "lora128_counting_und/epoch0-step400"
    "lora128_counting_und/epoch0-step500"
    "lora128_counting_und/epoch0-step600"
    "lora128_counting_und/epoch0-step700"
    "lora128_counting_und/epoch1-step1000"
    "lora128_counting_und/epoch1-step1100"
    "lora128_counting_und/epoch1-step1200"
    "lora128_counting_und/epoch1-step1300"
    "lora128_counting_und/epoch1-step1400"
    "lora128_counting_und/epoch1-step800"
    "lora128_counting_und/epoch1-step900"
    "lora128_counting_und/epoch2-step1500"
    "lora128_counting_und/epoch2-step1600"
    "lora128_counting_und/epoch2-step1700"
    "lora128_counting_und/epoch2-step1800"
    "lora128_counting_und/epoch2-step1900"
    "lora128_counting_und/epoch2-step2000"
    "lora128_counting_und/epoch2-step2100"
    "lora128_counting_und/epoch2-step2200"
    "lora128_counting_und/epoch3-step2300"
    "lora128_counting_und/epoch3-step2400"
    "lora128_counting_und/epoch3-step2500"
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running Pixmo counting evaluation for checkpoint: $ckpt"
    echo "Output directory: ${output_base}/${ckpt}"
    echo "========================================"

    torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        evaluation_scripts/counting/evaluate_pixmo_multigpu.py \
        --model_path ${model_path} \
        --lora_ckpt_path ${checkpoint_base}/${ckpt} \
        --output_dir ${output_base}/${ckpt} \
        --max_new_tokens ${MAX_NEW_TOKENS}
done
