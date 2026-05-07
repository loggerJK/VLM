#!/bin/bash
# PixMo Counting Evaluation — Single GPU (MMaDA)
#
# Run from the repository root:
#   bash evaluation/counting/evaluate_pixmo.sh
#
# Override defaults via env vars, e.g.:
#   MODEL_PATH=output/mmada-counting-llada-instruct/checkpoint-5000/unwrapped_model \
#   OUTPUT_DIR=evaluation_results/counting/step5000 \
#   bash evaluation/counting/evaluate_pixmo.sh

set -euo pipefail

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

MODEL_PATH=${MODEL_PATH:-Gen-Verse/MMaDA-8B-MixCoT}
VQ_MODEL_PATH=${VQ_MODEL_PATH:-showlab/magvitv2}
OUTPUT_DIR=${OUTPUT_DIR:-evaluation_results/counting/baseline}
RESOLUTION=${RESOLUTION:-512}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-20}
SEED=${SEED:-42}
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-mmada-counting-eval}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-}
LORA_PATH=${LORA_PATH:-}

WANDB_FLAGS=""
if [ "${USE_WANDB}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT}"
    if [ -n "${WANDB_RUN_NAME}" ]; then
        WANDB_FLAGS="${WANDB_FLAGS} --wandb_run_name ${WANDB_RUN_NAME}"
    fi
fi

LORA_FLAGS=""
if [ -n "${LORA_PATH}" ]; then
    LORA_FLAGS="--lora_path ${LORA_PATH}"
fi

echo "========================================"
echo "PixMo Counting eval (single GPU)"
echo "  MODEL_PATH    : ${MODEL_PATH}"
echo "  VQ_MODEL_PATH : ${VQ_MODEL_PATH}"
echo "  OUTPUT_DIR    : ${OUTPUT_DIR}"
echo "  RESOLUTION    : ${RESOLUTION}"
echo "  MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "  USE_WANDB     : ${USE_WANDB}"
echo "  LORA_PATH     : ${LORA_PATH:-(none)}"
echo "========================================"

python evaluation/counting/evaluate_pixmo.py \
    --model_path "${MODEL_PATH}" \
    --vq_model_path "${VQ_MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --resolution "${RESOLUTION}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    ${WANDB_FLAGS} \
    ${LORA_FLAGS}
