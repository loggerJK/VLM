#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# 사용자 아이디에 따른 HF_HOME 설정
if echo $USER | grep -q "cvlab17"; then
    echo "Running on cvlab17"
    data_dirname="cvlab22_data1"
fi
echo "data_dirname is set to $data_dirname"



base_dir="/mnt/$data_dirname"
# model_path="${base_dir}/jiwon/bagel_train/hf/BAGEL-7B-MoT"
# model_path="/mnt/$data_dirname/jiwon/BAGEL/models/BAGEL-7B-MoT"
model_path="/home/cvlab17/models/BAGEL-7B-MoT"
checkpoint_base="${base_dir}/jiwon/bagel_train/checkpoints"
output_base="${base_dir}/dvlm/BAGEL/ocr_synthetic/understanding"

NUM_SAMPLES=250
MAX_NEW_TOKENS=128

# Add bagel_train to PYTHONPATH
export PYTHONPATH=/mnt/${data_dirname:-data1}/jiwon/bagel_train:${PYTHONPATH:-}

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

# checkpoint_base="/mnt/data1/dvlm/BAGEL"
ckpt_list=(
    # Add your LoRA checkpoint relative paths here, e.g.:
    "ocr_synthetic_gen/epoch0-step100/"
    # "ocr_synthetic_gen/epoch0-step200/"
    # "ocr_synthetic_gen/epoch0-step300/"
    # "ocr_synthetic_gen/epoch0-step400/"
    # "ocr_synthetic_gen/epoch0-step500/"
    # "ocr_synthetic_gen/epoch0-step600/"
    # "ocr_synthetic_gen/epoch0-step700/"
    # "ocr_synthetic_gen/epoch1-step1000/"
    # "ocr_synthetic_gen/epoch1-step1100/"
    # "ocr_synthetic_gen/epoch1-step1200/"
    # "ocr_synthetic_gen/epoch1-step1300/"
    # "ocr_synthetic_gen/epoch1-step1400/"
    # "ocr_synthetic_gen/epoch1-step800/"
    # "ocr_synthetic_gen/epoch1-step900/"
    # "ocr_synthetic_gen/epoch2-step1500/"
    # "ocr_synthetic_gen/epoch2-step1600/"
    # "ocr_synthetic_gen/epoch2-step1700/"
    # "ocr_synthetic_gen/epoch2-step1800/"
    # "ocr_synthetic_gen/epoch2-step1900/"
    # "ocr_synthetic_gen/epoch2-step2000/"
    # "ocr_synthetic_gen/epoch2-step2100/"
    # "ocr_synthetic_gen/epoch2-step2200/"
    # "ocr_synthetic_gen/epoch3-step2300/"
    # "ocr_synthetic_gen/epoch3-step2400/"
    # "ocr_synthetic_gen/epoch3-step2500/"
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
