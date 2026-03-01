# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

export CUDA_VISIBLE_DEVICES=6,7
export MASTER_ADDR=localhost
export MASTER_PORT=25002
ngpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')

# 사용자 아이디에 따른 data_dirname 설정
if echo $USER | grep -q "cvlab17"; then
    echo "Running on cvlab17"
    data_dirname="cvlab22_data1"
fi
echo "data_dirname is set to $data_dirname"

base_dir="/mnt/$data_dirname"
model_path="/home/cvlab17/models/BAGEL-7B-MoT"
checkpoint_base="${base_dir}/jiwon/bagel_train/checkpoints"
output_base="${base_dir}/dvlm/BAGEL/geneval/generation"

# Add bagel_train to PYTHONPATH
export PYTHONPATH=/mnt/${data_dirname:-data1}/jiwon/bagel_train:${PYTHONPATH:-}

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

echo "========================================"
echo "Running baseline GenEval generation (BAGEL)"
echo "========================================"

torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    /mnt/$data_dirname/jiwon/bagel_train/eval/gen/gen_images_mp_sharding.py \
    --output_dir ${output_base}/baseline \
    --metadata_file evaluation_scripts/counting/evaluation_metadata_count.jsonl \
    --batch_size 1 \
    --num_images 4 \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path $model_path

# ---------------------------------------------------------------------------- #
#                              Checkpoint sweep                                #
# ---------------------------------------------------------------------------- #

ckpt_list=(
    "lora128_counting_und/epoch1"
    "lora128_counting_und/epoch2"
    "lora128_counting_und/epoch3"


    # "lora128_counting_und/epoch0-step100"
    # "lora128_counting_und/epoch0-step200"
    # "lora128_counting_und/epoch0-step300"
    # "lora128_counting_und/epoch0-step400"
    # "lora128_counting_und/epoch0-step500"
    # "lora128_counting_und/epoch0-step600"
    # "lora128_counting_und/epoch0-step700"
    # "lora128_counting_und/epoch1-step1000"
    # "lora128_counting_und/epoch1-step1100"
    # "lora128_counting_und/epoch1-step1200"
    # "lora128_counting_und/epoch1-step1300"
    # "lora128_counting_und/epoch1-step1400"
    # "lora128_counting_und/epoch1-step800"
    # "lora128_counting_und/epoch1-step900"
    # "lora128_counting_und/epoch2-step1500"
    # "lora128_counting_und/epoch2-step1600"
    # "lora128_counting_und/epoch2-step1700"
    # "lora128_counting_und/epoch2-step1800"
    # "lora128_counting_und/epoch2-step1900"
    # "lora128_counting_und/epoch2-step2000"
    # "lora128_counting_und/epoch2-step2100"
    # "lora128_counting_und/epoch2-step2200"
    # "lora128_counting_und/epoch3-step2300"
    # "lora128_counting_und/epoch3-step2400"
    # "lora128_counting_und/epoch3-step2500"
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running GenEval generation for checkpoint: $ckpt"
    echo "Output directory: ${output_base}/${ckpt}"
    echo "========================================"

    torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        /mnt/$data_dirname/jiwon/bagel_train/eval/gen/gen_images_mp_sharding.py \
        --output_dir ${output_base}/${ckpt} \
        --metadata_file evaluation_scripts/counting/evaluation_metadata_count.jsonl \
        --batch_size 1 \
        --num_images 4 \
        --resolution 1024 \
        --max_latent_size 64 \
        --model-path $model_path \
        --lora-ckpt-path ${checkpoint_base}/${ckpt}
done
