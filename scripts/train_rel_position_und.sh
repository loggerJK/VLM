#!/bin/bash
# Relative-position understanding training with LoRA.
# Dataset: heez/relative-position-new

set -euo pipefail
source /data/mm-llm-backbone_890/personal/sirius/audio_ablation/env.sh
conda activate bagel

# ============================================================
# Settings - edit here
# ============================================================
CUDA="0"
MASTER_PORT=${MASTER_PORT:-29602}
LR=2e-5
TOTAL_STEPS=50000
SAVE_EVERY=100
EXP_NAME=lora128_rel_position_und
EFFECTIVE_BATCH=128
NGPUS=$(echo $CUDA | awk -F',' '{print NF}')
GRAD_ACCUM=$((EFFECTIVE_BATCH / NGPUS))

# Resume (leave empty to train from scratch)
RESUME_FROM=""
LORA_CKPT_PATH=""
WANDB_RUN_ID=""

# ============================================================
# Environment
# ============================================================
export CUDA_VISIBLE_DEVICES=${CUDA}
export PYTHONNOUSERSITE=1
export PYTHONPATH=/mnt/data1/jiwon/bagel_train:${PYTHONPATH:-}
export PYTORCH_ALLOC_CONF=expandable_segments:True

RESULTS_DIR=./results/${EXP_NAME}
CHECKPOINT_DIR=./checkpoints/${EXP_NAME}
mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}"

RESUME_ARGS=""
[ -n "${RESUME_FROM}" ]   && RESUME_ARGS="${RESUME_ARGS} --resume_from ${RESUME_FROM}"
[ -n "${LORA_CKPT_PATH}" ] && RESUME_ARGS="${RESUME_ARGS} --lora_ckpt_path ${LORA_CKPT_PATH}"
[ -n "${WANDB_RUN_ID}" ]   && RESUME_ARGS="${RESUME_ARGS} --wandb_runid ${WANDB_RUN_ID} --wandb_resume must"

echo "============================================================"
echo " BAGEL - rel_position und + LoRA (${NGPUS} GPUs)"
echo "  CUDA=${CUDA}  LR=${LR}  Steps=${TOTAL_STEPS}"
echo "  Ckpt: ${CHECKPOINT_DIR}"
echo "============================================================"

torchrun \
    --nproc_per_node=${NGPUS} \
    --master_port=${MASTER_PORT} \
    ./train/pretrain_unified_navit.py \
    --model_path ./models/ \
    --finetune_from_hf True \
    --layer_module Qwen2MoTDecoderLayer \
    --use_flex True \
    --sharding_strategy NO_SHARD \
    --num_shard 1 --num_replicate 1 \
    --task rel_position --mode und \
    --visual_gen False \
    --freeze_vit True \
    --use_lora True \
    --lora_rank 128 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --num_workers 4 \
    --expected_num_tokens 2048 \
    --max_num_tokens 8192 \
    --max_num_tokens_per_sample 8192 \
    --results_dir "${RESULTS_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --wandb_project bagel-position \
    --wandb_name "${EXP_NAME}" \
    --wandb_offline False \
    --total_steps ${TOTAL_STEPS} \
    --save_every ${SAVE_EVERY} \
    --log_every 1 \
    --lr ${LR} \
    --validation_interval 100 \
    --eval_before_training \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    ${RESUME_ARGS}
