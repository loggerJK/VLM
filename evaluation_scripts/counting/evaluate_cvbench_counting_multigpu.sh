#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=25001
ngpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')

base_dir="/mnt/data1"
model_path="deepseek-ai/Janus-Pro-7B"

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

echo "========================================"
echo "Running baseline CVBench counting evaluation"
echo "========================================"

torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    evaluation_scripts/counting/evaluate_cvbench_counting_multigpu.py \
    --model_path ${model_path} \
    --output_dir ${base_dir}/dvlm/janus/counting/understanding_eval_cvbench/baseline \
    --max_new_tokens 512

# ---------------------------------------------------------------------------- #
#                              Checkpoint sweep                                #
# ---------------------------------------------------------------------------- #

# ckpt_list=(
#     # Add Janus checkpoints here...
# )
#
# for ckpt in "${ckpt_list[@]}"; do
#     echo "========================================"
#     echo "Running evaluation for: $ckpt"
#     echo "========================================"
#
#     torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
#         evaluation_scripts/counting/evaluate_cvbench_counting_multigpu.py \
#         --model_path ${ckpt} \
#         --output_dir ${base_dir}/dvlm/janus/counting/understanding_eval_cvbench/$(basename ${ckpt}) \
#         --max_new_tokens 512
# done
