#!/bin/bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-25005}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
BASE_DIR="${BASE_DIR:-/mnt/data1}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/Janus-Pro-7B}"
PROMPT_FILE="${PROMPT_FILE:-/mnt/data1/jiwon/Lumina-DiMOO/evaluation_scripts/fid/captions_val2014.json}"
OUTPUT_BASE="${OUTPUT_BASE:-${BASE_DIR}/dvlm/janus/fid/generation_eval}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-${BASE_DIR}/dvlm/janus/checkpoints}"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-1000}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-1}"
CFG_WEIGHT="${CFG_WEIGHT:-5.0}"
IMG_SIZE="${IMG_SIZE:-384}"
SEED="${SEED:-65513}"

check_done() {
    local out_dir="$1"
    local expected_png_count=$((EXPECTED_IMAGES * NUM_SAMPLES_PER_PROMPT))

    if [ -d "${out_dir}" ]; then
        local count
        count=$(find "${out_dir}" -maxdepth 1 -name '*.png' | wc -l)
        if [ "${count}" -ge "${expected_png_count}" ]; then
            echo "[SKIP] ${out_dir} already has ${count} images (>= ${expected_png_count})."
            return 0
        fi
        echo "[RESUME] ${out_dir} has ${count}/${expected_png_count} images. Resuming."
    fi

    return 1
}

run_fid() {
    local out_dir="$1"
    local lora_path="${2:-}"

    if check_done "${out_dir}"; then
        return
    fi

    mkdir -p "${out_dir}"

    local cmd=(
        torchrun
        --nproc_per_node="${NGPUS}"
        --rdzv_backend=c10d
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
        "${REPO_ROOT}/evaluation_scripts/fid/fid_inference.py"
        --model_path "${MODEL_PATH}"
        --prompt_files "${PROMPT_FILE}"
        --output_dir "${out_dir}"
        --max_prompts "${EXPECTED_IMAGES}"
        --num_samples_per_prompt "${NUM_SAMPLES_PER_PROMPT}"
        --cfg_weight "${CFG_WEIGHT}"
        --img_size "${IMG_SIZE}"
        --seed "${SEED}"
    )

    if [ -n "${lora_path}" ]; then
        cmd+=(--lora_ckpt_path "${lora_path}")
    fi

    echo "========================================"
    echo "Running Janus FID generation"
    echo "Prompt file: ${PROMPT_FILE}"
    echo "Output directory: ${out_dir}"
    if [ -n "${lora_path}" ]; then
        echo "LoRA checkpoint: ${lora_path}"
    else
        echo "Mode: baseline"
    fi
    echo "Num images: ${EXPECTED_IMAGES}"
    echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
    echo "========================================"

    (
        cd "${REPO_ROOT}"
        "${cmd[@]}"
    )
}

# run_fid "${OUTPUT_BASE}/baseline"


# CHECKPOINT_BASE="${BASE_DIR}/jiwon/janus_counting-only/checkpoints"
# CKPT_LIST=(
#     "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch2"
# )

# for ckpt in "${CKPT_LIST[@]}"; do
#     run_fid "${OUTPUT_BASE}/${ckpt}" "${CHECKPOINT_BASE}/${ckpt}"
# done

CHECKPOINT_BASE="${BASE_DIR}/dvlm/janus/checkpoints"
CKPT_LIST=(
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch1"
)

for ckpt in "${CKPT_LIST[@]}"; do
    run_fid "${OUTPUT_BASE}/${ckpt}" "${CHECKPOINT_BASE}/${ckpt}"
done
