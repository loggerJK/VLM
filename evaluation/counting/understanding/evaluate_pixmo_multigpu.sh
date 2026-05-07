#!/bin/bash
# PixMo Counting Evaluation — Multi-GPU (MMaDA)
#
# Run from the repository root:
#   bash evaluation/counting/evaluate_pixmo_multigpu.sh
#
# Sweeps a list of checkpoint subdirectories under ${CHECKPOINT_BASE}.
# Expected checkpoint layout (from train_mmada_counting.py):
#   ${CHECKPOINT_BASE}/checkpoint-${global_step}/unwrapped_model/
#
# Override defaults via env vars, e.g.:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   CHECKPOINT_BASE=output/mmada-counting-llada-instruct \
#   OUTPUT_BASE=evaluation_results/counting \
#   bash evaluation/counting/evaluate_pixmo_multigpu.sh

set -euo pipefail

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-25001}
ngpus=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')

BASE_MODEL_PATH=${BASE_MODEL_PATH:-Gen-Verse/MMaDA-8B-MixCoT}
VQ_MODEL_PATH=${VQ_MODEL_PATH:-showlab/magvitv2}
CHECKPOINT_BASE=${CHECKPOINT_BASE:-output/mmada-counting-llada-instruct}
OUTPUT_BASE=${OUTPUT_BASE:-evaluation_results/counting}
RESOLUTION=${RESOLUTION:-512}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-20}
SEED=${SEED:-42}
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-mmada-counting-eval}
USE_LORA_CKPT=${USE_LORA_CKPT:-1}

WANDB_FLAGS=""
if [ "${USE_WANDB}" = "1" ]; then
    WANDB_FLAGS="--use_wandb --wandb_project ${WANDB_PROJECT}"
fi

run_eval() {
    local model_path="$1"
    local out_dir="$2"
    local run_name="$3"
    local lora_path="${4:-}"

    local extra_flags="${WANDB_FLAGS}"
    if [ "${USE_WANDB}" = "1" ] && [ -n "${run_name}" ]; then
        extra_flags="${extra_flags} --wandb_run_name ${run_name}"
    fi
    if [ -n "${lora_path}" ]; then
        extra_flags="${extra_flags} --lora_path ${lora_path}"
    fi

    echo "========================================"
    echo "PixMo counting eval"
    echo "  model_path : ${model_path}"
    echo "  lora_path  : ${lora_path:-(none)}"
    echo "  output_dir : ${out_dir}"
    echo "  ngpus      : ${ngpus}"
    echo "========================================"

    torchrun --nproc_per_node=${ngpus} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        evaluation/counting/evaluate_pixmo_multigpu.py \
        --model_path "${model_path}" \
        --vq_model_path "${VQ_MODEL_PATH}" \
        --output_dir "${out_dir}" \
        --resolution "${RESOLUTION}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --seed "${SEED}" \
        ${extra_flags}
}

# ---------------------------------------------------------------------------- #
#                              Baseline (pretrained)                           #
# ---------------------------------------------------------------------------- #

run_eval "${BASE_MODEL_PATH}" "${OUTPUT_BASE}/baseline" "baseline"

# ---------------------------------------------------------------------------- #
#                          Checkpoint sweep (edit me)                          #
# ---------------------------------------------------------------------------- #
#
# Each entry is a subdirectory name under ${CHECKPOINT_BASE}.
# Resolved checkpoint path: ${CHECKPOINT_BASE}/${ckpt}/unwrapped_model
# Resolved output dir     : ${OUTPUT_BASE}/${ckpt}

ckpt_list=(
    # "checkpoint-1000"
    # "checkpoint-2000"
    # "checkpoint-5000"
)

for ckpt in "${ckpt_list[@]}"; do
    ckpt_dir="${CHECKPOINT_BASE}/${ckpt}/unwrapped_model"
    out_dir="${OUTPUT_BASE}/${ckpt}"

    if [ ! -d "${ckpt_dir}" ]; then
        echo "Skipping ${ckpt}: ${ckpt_dir} does not exist."
        continue
    fi

    if [ "${USE_LORA_CKPT}" = "1" ]; then
        # LoRA adapter checkpoint: base = BASE_MODEL_PATH, adapter = ckpt_dir
        run_eval "${BASE_MODEL_PATH}" "${out_dir}" "${ckpt}" "${ckpt_dir}"
    else
        # Legacy full-FT checkpoint: ckpt_dir holds full MMaDA weights
        run_eval "${ckpt_dir}" "${out_dir}" "${ckpt}"
    fi
done
