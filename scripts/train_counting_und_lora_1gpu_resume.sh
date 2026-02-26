#!/bin/bash
# Counting understanding training — 1 GPU NO_SHARD + LoRA (rank 128)
# Dataset: heez/pixmo-point-count-gen-und
#
# RESUME from epoch0-step700 checkpoint.
# To change settings, edit the variables below directly.

set -euo pipefail
source ./.env

# ============================================================
# Settings — edit here
# ============================================================
CUDA="4,5,6,7"
MASTER_PORT=${MASTER_PORT:-29601}
LR=2e-5
TOTAL_STEPS=50000
SAVE_EVERY=100
EXP_NAME=lora128_counting_und
EFFECTIVE_BATCH=128
NGPUS=$(echo $CUDA | awk -F',' '{print NF}')
GRAD_ACCUM=$((EFFECTIVE_BATCH / NGPUS))

# Resume (leave empty to train from scratch)
# Resume from epoch0-step700
RESUME_FROM="/mnt/data1/jiwon/bagel_train/checkpoints/lora128_counting_und/epoch0-step700"
# RESUME_FROM=""
LORA_CKPT_PATH="/mnt/data1/jiwon/bagel_train/checkpoints/lora128_counting_und/epoch0-step700"
WANDB_RUN_ID="y9vdxdoj"
# WANDB_RUN_ID="0"

# ============================================================
# Environment
# ============================================================
export CUDA_VISIBLE_DEVICES=${CUDA}
export PYTHONNOUSERSITE=1
export PYTHONPATH=/mnt/data1/jiwon/bagel_train:${PYTHONPATH:-}
export PYTORCH_ALLOC_CONF=expandable_segments:True

RESULTS_DIR=/mnt/data1/jiwon/bagel_train/results/${EXP_NAME}
CHECKPOINT_DIR=/mnt/data1/jiwon/bagel_train/checkpoints/${EXP_NAME}
mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}"

RESUME_ARGS=""
[ -n "${RESUME_FROM}" ]   && RESUME_ARGS="${RESUME_ARGS} --resume_from ${RESUME_FROM}"
[ -n "${LORA_CKPT_PATH}" ] && RESUME_ARGS="${RESUME_ARGS} --lora_ckpt_path ${LORA_CKPT_PATH}"
[ -n "${WANDB_RUN_ID}" ]   && RESUME_ARGS="${RESUME_ARGS} --wandb_runid ${WANDB_RUN_ID} --wandb_resume must"

echo "============================================================"
echo " BAGEL — counting und + LoRA (${NGPUS} GPUs) [RESUME]"
echo "  CUDA=${CUDA}  LR=${LR}  Steps=${TOTAL_STEPS}"
echo "  Resume from: ${RESUME_FROM}"
echo "  Ckpt: ${CHECKPOINT_DIR}"
echo "============================================================"

torchrun \
    --nproc_per_node=${NGPUS} \
    --master_port=${MASTER_PORT} \
    /mnt/data1/jiwon/bagel_train/train/pretrain_unified_navit.py \
    --model_path /mnt/data1/jiwon/BAGEL/models/BAGEL-7B-MoT \
    --finetune_from_hf True \
    --layer_module Qwen2MoTDecoderLayer \
    --use_flex True \
    --max_latent_size 64 \
    --sharding_strategy NO_SHARD \
    --num_shard 1 --num_replicate 1 \
    --task counting --mode und \
    --visual_gen True \
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
    --wandb_project bagel-counting \
    --wandb_name "${EXP_NAME}" \
    --wandb_offline False \
    --total_steps ${TOTAL_STEPS} \
    --save_every ${SAVE_EVERY} \
    --log_every 1 \
    --lr ${LR} \
    --validation_interval 100 \
    --eval_everything \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    ${RESUME_ARGS}
