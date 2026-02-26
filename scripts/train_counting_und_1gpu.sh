#!/bin/bash
# Counting understanding — 1 GPU NO_SHARD (requires cpu_offload for full-param)

set -euo pipefail
source ./.env

# ============================================================
# Settings — edit here
# ============================================================
CUDA=3
MASTER_PORT=29600
LR=1e-4
TOTAL_STEPS=50000
SAVE_EVERY=2000
EXP_NAME=counting_und_1gpu
EFFECTIVE_BATCH=128
NGPUS=1
GRAD_ACCUM=$((EFFECTIVE_BATCH / NGPUS))

# Resume (leave empty to train from scratch)
RESUME_FROM=""
WANDB_RUN_ID=""

# ============================================================
export CUDA_VISIBLE_DEVICES=${CUDA}
export PYTHONNOUSERSITE=1
export PYTHONPATH=/mnt/data1/jiwon/bagel_train:${PYTHONPATH:-}
export PYTORCH_ALLOC_CONF=expandable_segments:True

RESULTS_DIR=/mnt/data1/jiwon/bagel_train/results/${EXP_NAME}
CHECKPOINT_DIR=/mnt/data1/jiwon/bagel_train/checkpoints/${EXP_NAME}
mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}"

RESUME_ARGS=""
[ -n "${RESUME_FROM}" ] && RESUME_ARGS="${RESUME_ARGS} --resume_from ${RESUME_FROM}"
[ -n "${WANDB_RUN_ID}" ] && RESUME_ARGS="${RESUME_ARGS} --wandb_runid ${WANDB_RUN_ID} --wandb_resume must"

/home/cvlab22/anaconda3/envs/bagel/bin/torchrun \
    --nproc_per_node=1 \
    --master_port=${MASTER_PORT} \
    /mnt/data1/jiwon/bagel_train/train/pretrain_unified_navit.py \
    --model_path /mnt/data1/jiwon/BAGEL/models/BAGEL-7B-MoT \
    --finetune_from_hf True \
    --layer_module Qwen2MoTDecoderLayer \
    --use_flex True \
    --max_latent_size 64 \
    --sharding_strategy NO_SHARD \
    --num_shard 1 --num_replicate 1 \
    --cpu_offload True \
    --task counting --mode und \
    --visual_gen False \
    --freeze_vit True \
    --num_workers 1 \
    --expected_num_tokens 2048 \
    --max_num_tokens 4096 \
    --max_num_tokens_per_sample 4096 \
    --results_dir "${RESULTS_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --wandb_project bagel \
    --wandb_name "${EXP_NAME}" \
    --total_steps ${TOTAL_STEPS} \
    --save_every ${SAVE_EVERY} \
    --log_every 10 \
    --lr ${LR} \
    --validation_interval 500 \
    --eval_everything \
    --eval_before_training \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    ${RESUME_ARGS}
