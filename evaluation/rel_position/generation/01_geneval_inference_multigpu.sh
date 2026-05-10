#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MASTER_ADDR=localhost
export MASTER_PORT=29502
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

ngpus=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"


# Optional: source a venv via VENV env var, e.g. VENV=/path/to/.venv/bin/activate
if [ -n "${VENV:-}" ] && [ -f "${VENV}" ]; then
    # shellcheck disable=SC1090
    source "${VENV}"
fi

dirname=${DIRNAME:-data1} # override via DIRNAME env var (e.g. DIRNAME=cvlab22_data1)

base_dir="/mnt/${dirname}"
NUM_SAMPLES=4
PROMPT_FILE="/mnt/${dirname}/jiwon/Lumina-DiMOO/evaluation_scripts/rel_position/generation/evaluation_metadata_diag_rel_positions.jsonl"
CONFIG="${REPO_ROOT}/configs/mmada_rel_position_gen_llada_instruct.yaml"
TASK_NAME="rel_position" # or "counting"


run_inference() {
    local ckpt_path="$1"
    local out_name="$2"
    # New layout: dvlm/mmada/<task>/generation/<lora_name>/<ckpt_dir>/
    local lora_name
    lora_name=$(basename "${CKPT_ROOT}")
    local out_dir="${base_dir}/dvlm/mmada/${TASK_NAME}/generation/${lora_name}/${out_name}"

    # Print
    echo "==============================="
    echo "Running inference with checkpoint: ${ckpt_path}"
    echo "Output will be saved to: ${out_dir}"
    echo "==============================="

    torchrun \
        --nproc_per_node "${ngpus}" \
        --master_addr "${MASTER_ADDR}" \
        --master_port "${MASTER_PORT}" \
        "${SCRIPT_DIR}/inference_geneval_t2i_multigpu.py" \
        --config "${CONFIG}" \
        --prompt_files "${PROMPT_FILE}" \
        --output_dir "${out_dir}" \
        --num_samples "${NUM_SAMPLES}" \
        --checkpoint_path "${ckpt_path}"
}

# Baseline: uses pretrained_model_path from config, no checkpoint override
baseline_out_dir="${base_dir}/dvlm/mmada/${TASK_NAME}/generation/mmada_BASELINE"
echo "==============================="
echo "Running baseline inference (no checkpoint override)"
echo "Output will be saved to: ${baseline_out_dir}"
echo "==============================="
torchrun \
    --nproc_per_node "${ngpus}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${SCRIPT_DIR}/inference_geneval_t2i_multigpu.py" \
    --config "${CONFIG}" \
    --prompt_files "${PROMPT_FILE}" \
    --output_dir "${baseline_out_dir}" \
    --num_samples "${NUM_SAMPLES}"



CKPT_ROOT="/mnt/${dirname}/dvlm/mmada/checkpoints/mmada-rel-position-llada-instruct"

ckpt_list=(
    "checkpoint-500/"
    "checkpoint-1000/"
    "checkpoint-1500/"
    "checkpoint-2000/"
    # "checkpoint-2500/"
    # "checkpoint-3000/"
    # "checkpoint-3500/"
    # "checkpoint-4000/"
    # "checkpoint-4500/"
    # "checkpoint-5000/"
    # "checkpoint-5500/"
    # "checkpoint-6000/"
    # "checkpoint-6500/"
    # "checkpoint-7000/"
    # "checkpoint-7500/"
    # "checkpoint-8000/"
    # "checkpoint-8500/"
    # "checkpoint-9000/"
    # "checkpoint-9500/"
)

for ckpt in "${ckpt_list[@]}"; do
    ckpt_path="${CKPT_ROOT}/${ckpt}"
    out_name="mmada_${ckpt%/}"
    run_inference "${ckpt_path}" "${out_name}"
done
