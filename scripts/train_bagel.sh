#!/bin/bash
# Universal BAGEL training launcher
#
# Usage:
#   bash scripts/train_bagel.sh --task counting --mode und --num_gpus 2
#   bash scripts/train_bagel.sh --task ocr --mode both --num_gpus 4 --hf_dataset naver-clova-ix/cord-v2
#   bash scripts/train_bagel.sh --task ocr_synthetic --mode gen --num_gpus 8
#
# Override any default with env vars:
#   LR=5e-5 TOTAL_STEPS=10000 bash scripts/train_bagel.sh --task counting --mode und --num_gpus 2
#
# Resume training:
#   RESUME_FROM=/path/to/ckpt WANDB_RUN_ID=abc123 bash scripts/train_bagel.sh ...

set -euo pipefail
source ./.env

# ============================================================
# Parse arguments
# ============================================================
TASK=""
MODE=""
NUM_GPUS=2
HF_DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)       TASK="$2"; shift 2 ;;
        --mode)       MODE="$2"; shift 2 ;;
        --num_gpus)   NUM_GPUS="$2"; shift 2 ;;
        --hf_dataset) HF_DATASET="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$TASK" ] || [ -z "$MODE" ]; then
    echo "Usage: bash scripts/train_bagel.sh --task <counting|ocr|ocr_synthetic> --mode <und|gen|both> --num_gpus <N>"
    exit 1
fi

# Validate task/mode
if [[ ! "$TASK" =~ ^(counting|ocr|ocr_synthetic)$ ]]; then
    echo "Error: --task must be counting, ocr, or ocr_synthetic (got: ${TASK})"
    exit 1
fi
if [[ ! "$MODE" =~ ^(und|gen|both)$ ]]; then
    echo "Error: --mode must be und, gen, or both (got: ${MODE})"
    exit 1
fi
if [ "$TASK" = "ocr" ] && [ -z "$HF_DATASET" ]; then
    echo "Error: --hf_dataset required for task=ocr"
    exit 1
fi

# ============================================================
# Environment
# ============================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}
export PYTHONNOUSERSITE=1
export PYTHONPATH=/mnt/data1/jiwon/bagel_train:${PYTHONPATH:-}
export PYTORCH_ALLOC_CONF=expandable_segments:True

MASTER_PORT=${MASTER_PORT:-29600}

# ============================================================
# FSDP strategy: 2 GPU → FULL_SHARD, 4+ GPU → HYBRID_SHARD
# ============================================================
if [ "$NUM_GPUS" -le 2 ]; then
    SHARDING_STRATEGY="FULL_SHARD"
    NUM_SHARD=${NUM_GPUS}
else
    SHARDING_STRATEGY=${SHARDING_STRATEGY:-"HYBRID_SHARD"}
    NUM_SHARD=${NUM_SHARD:-${NUM_GPUS}}
fi

# ============================================================
# visual_gen: True for gen/both, False for und
# ============================================================
if [ "$MODE" = "und" ]; then
    VISUAL_GEN="False"
else
    VISUAL_GEN="True"
fi

# ============================================================
# Experiment config (overridable via env)
# ============================================================
EXP_NAME="${TASK}_${MODE}_${NUM_GPUS}gpu_$(echo ${SHARDING_STRATEGY} | tr '[:upper:]' '[:lower:]')"
TOTAL_STEPS=${TOTAL_STEPS:-50000}
SAVE_EVERY=${SAVE_EVERY:-2000}
LOG_EVERY=${LOG_EVERY:-10}
LR=${LR:-1e-4}
EXPECTED_TOKENS=${EXPECTED_TOKENS:-2048}
MAX_TOKENS=${MAX_TOKENS:-4096}
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-128}
GRAD_ACCUM=$((EFFECTIVE_BATCH / NUM_GPUS))

# validation only for und/both modes
if [ "$MODE" = "gen" ]; then
    VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-0}
else
    VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-500}
fi

# ============================================================
# Paths
# ============================================================
MODEL_PATH=${MODEL_PATH:-/mnt/data1/jiwon/BAGEL/models/BAGEL-7B-MoT}
RESULTS_DIR=${RESULTS_DIR:-/mnt/data1/jiwon/bagel_train/results/${EXP_NAME}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/mnt/data1/jiwon/bagel_train/checkpoints/${EXP_NAME}}

# ============================================================
# Resume (optional)
# ============================================================
RESUME_ARGS=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARGS="--resume_from ${RESUME_FROM}"
fi
if [ -n "${WANDB_RUN_ID:-}" ]; then
    RESUME_ARGS="${RESUME_ARGS} --wandb_runid ${WANDB_RUN_ID} --wandb_resume must"
fi

# HF dataset arg (only for ocr task)
HF_DATASET_ARGS=""
if [ -n "$HF_DATASET" ]; then
    HF_DATASET_ARGS="--hf_dataset_path ${HF_DATASET}"
fi

mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}"

echo "============================================================"
echo " BAGEL Training"
echo "  Task:     ${TASK}"
echo "  Mode:     ${MODE}"
echo "  GPUs:     ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "  FSDP:     ${SHARDING_STRATEGY} (num_shard=${NUM_SHARD})"
echo "  LR:       ${LR}"
echo "  Steps:    ${TOTAL_STEPS}"
echo "  Results:  ${RESULTS_DIR}"
echo "  GradAccum: ${GRAD_ACCUM} (effective batch: ${EFFECTIVE_BATCH})"
echo "  Ckpt:     ${CHECKPOINT_DIR}"
echo "============================================================"

/home/cvlab22/anaconda3/envs/bagel/bin/torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    /mnt/data1/jiwon/bagel_train/train/pretrain_unified_navit.py \
    --model_path "${MODEL_PATH}" \
    --finetune_from_hf True \
    --layer_module Qwen2MoTDecoderLayer \
    --use_flex True \
    --max_latent_size 64 \
    --sharding_strategy ${SHARDING_STRATEGY} \
    --num_shard ${NUM_SHARD} --num_replicate 1 \
    --task ${TASK} --mode ${MODE} \
    --visual_gen ${VISUAL_GEN} \
    --freeze_vit True \
    --num_workers 4 \
    --expected_num_tokens ${EXPECTED_TOKENS} \
    --max_num_tokens ${MAX_TOKENS} \
    --max_num_tokens_per_sample ${MAX_TOKENS} \
    --results_dir "${RESULTS_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --wandb_project bagel \
    --wandb_name "${EXP_NAME}" \
    --total_steps ${TOTAL_STEPS} \
    --save_every ${SAVE_EVERY} \
    --log_every ${LOG_EVERY} \
    --lr ${LR} \
    --validation_interval ${VALIDATION_INTERVAL} \
    --eval_everything \
    --eval_before_training \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    ${HF_DATASET_ARGS} \
    ${RESUME_ARGS}
