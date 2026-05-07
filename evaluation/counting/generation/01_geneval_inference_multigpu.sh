#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MASTER_ADDR=localhost
export MASTER_PORT=29501

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

ngpus=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

base_dir="/mnt/data1/jiwon/geneval/results"
NUM_SAMPLES=20
PROMPT_FILE="/mnt/data1/jiwon/geneval/prompts/evaluation_metadata_count.jsonl"
CONFIG="${REPO_ROOT}/configs/mmada_counting_llada_instruct.yaml"

run_inference() {
    local ckpt_path="$1"
    local out_name="$2"
    local out_dir="${base_dir}/dvlm/mmada/counting/${out_name}"

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
torchrun \
    --nproc_per_node "${ngpus}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${SCRIPT_DIR}/inference_geneval_t2i_multigpu.py" \
    --config "${CONFIG}" \
    --prompt_files "${PROMPT_FILE}" \
    --output_dir "${base_dir}/dvlm/mmada/counting/mmada_BASELINE" \
    --num_samples "${NUM_SAMPLES}"

# Add checkpoint runs below:
# run_inference "/path/to/checkpoint-1000" "mmada_ckpt1000"
