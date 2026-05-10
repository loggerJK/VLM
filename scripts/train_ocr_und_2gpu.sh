#!/bin/bash
# OCR understanding — 2 GPU FULL_SHARD

set -euo pipefail
source /data/mm-llm-backbone_890/personal/sirius/audio_ablation/env.sh
conda activate bagel

# ============================================================
# Settings — edit here
# ============================================================
CUDA=2,3
MASTER_PORT=29600
LR=1e-4
TOTAL_STEPS=50000
SAVE_EVERY=2000
EXP_NAME=ocr_und_2gpu
HF_DATASET_PATH=Jiwon-Kang/OCR-Synthetic-Rendered-200K
EFFECTIVE_BATCH=128
NGPUS=2
GRAD_ACCUM=$((EFFECTIVE_BATCH / NGPUS))

# Resume (leave empty to train from scratch)
RESUME_FROM=""
WANDB_RUN_ID=""

# ============================================================
export CUDA_VISIBLE_DEVICES=${CUDA}
export PYTHONNOUSERSITE=1
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export PYTORCH_ALLOC_CONF=expandable_segments:True

RESULTS_DIR=./results/${EXP_NAME}
CHECKPOINT_DIR=./checkpoints/${EXP_NAME}
mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}"

RESUME_ARGS=""
[ -n "${RESUME_FROM}" ] && RESUME_ARGS="${RESUME_ARGS} --resume_from ${RESUME_FROM}"
[ -n "${WANDB_RUN_ID}" ] && RESUME_ARGS="${RESUME_ARGS} --wandb_runid ${WANDB_RUN_ID} --wandb_resume must"

torchrun \
    --nproc_per_node=2 \
    --master_port=${MASTER_PORT} \
    ./train/pretrain_unified_navit.py \
    --model_path ./models/ \
    --finetune_from_hf True \
    --layer_module Qwen2MoTDecoderLayer \
    --use_flex True \
    --max_latent_size 64 \
    --sharding_strategy FULL_SHARD \
    --num_shard 2 --num_replicate 1 \
    --task ocr --mode und \
    --hf_dataset_path "${HF_DATASET_PATH}" \
    --visual_gen False \
    --freeze_vit True \
    --num_workers 4 \
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
