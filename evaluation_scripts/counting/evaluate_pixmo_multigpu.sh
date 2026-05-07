#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=25001
ngpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')

base_dir="/mnt/data1"
model_path="deepseek-ai/Janus-Pro-7B"
checkpoint_base="${base_dir}/jiwon/janus_counting-only/checkpoints"
output_base="${base_dir}/dvlm/janus/counting/understanding_eval_pixmo"

MAX_NEW_TOKENS=32

# Add /mnt/data1/jiwon/Janus to PYTHONPATH
export PYTHONPATH=/mnt/data1/jiwon/Janus:${PYTHONPATH:-}

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

# echo "========================================"
# echo "Running baseline Pixmo counting evaluation"
# echo "========================================"

# torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
#     evaluation_scripts/counting/evaluate_pixmo_multigpu.py \
#     --model_path ${model_path} \
#     --output_dir ${output_base}/baseline \
#     --max_new_tokens ${MAX_NEW_TOKENS}

# ---------------------------------------------------------------------------- #
#                              Checkpoint sweep                                #
# ---------------------------------------------------------------------------- #

# ckpt_list=(
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch1"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch2"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch3"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch4"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch5"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch6"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch7"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch8"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch9"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch10"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch11"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch12"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch13"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch14"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch15"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch16"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch17"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch18"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch19"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch20"
# )


checkpoint_base="${base_dir}/dvlm/janus/checkpoints"
ckpt_list=(

    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch1"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch2"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch3"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch4"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch5"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch6"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch7"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch8"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch9"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch10"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch11"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch12"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch13"


    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch0_step-0"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch0_step-100"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch0_step-200"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch0_step-300"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch0_step-400"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch0_step-500"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch1_step-600"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch1_step-700"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch1_step-800"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch1_step-900"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch1_step-1000"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch2_step-1100"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch2_step-1200"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch2_step-1300"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch2_step-1400"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch2_step-1500"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch3_step-1600"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch3_step-1700"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch3_step-1800"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch3_step-1900"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch3_step-2000"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch4_step-2100"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch4_step-2200"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch4_step-2300"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch4_step-2400"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch4_step-2500"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch5_step-2600"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch5_step-2700"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch5_step-2800"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch5_step-2900"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch5_step-3000"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch6_step-3100"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch6_step-3200"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch6_step-3300"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch6_step-3400"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch6_step-3500"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch7_step-3600"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch7_step-3700"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch7_step-3800"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch7_step-3900"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch7_step-4000"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch8_step-4100"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch8_step-4200"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch8_step-4300"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch8_step-4400"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch8_step-4500"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch8_step-4600"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch9_step-4700"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch9_step-4800"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch9_step-4900"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch9_step-5000"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch9_step-5100"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch10_step-5200"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch10_step-5300"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch10_step-5400"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch10_step-5500"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch10_step-5600"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch11_step-5700"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch11_step-5800"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch11_step-5900"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch11_step-6000"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch11_step-6100"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch12_step-6200"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch12_step-6300"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch12_step-6400"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch12_step-6500"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch12_step-6600"
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch13_step-6700"
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
