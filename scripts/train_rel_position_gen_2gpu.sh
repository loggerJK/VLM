#!/bin/bash
# Relative-position generation (T2I) training with LoRA, 2 GPU.
# Dataset: heez/relative-position-new (caption = answer field).
#
# Validation metrics (read carefully):
#   - inline val/pos_acc, val/pos_parse_fail_rate are *understanding* metrics
#     computed by validate_understanding() via model.chat() over the dataset's
#     validation split. They are NOT generation metrics.
#   - For generation-quality numbers (GenEval-style spatial-relation accuracy
#     on synthesized images), run evaluation_scripts/rel_position/run_geneval.sh
#     after training against the saved LoRA checkpoint.
#   - validate_generation() only logs sample images to wandb for visual sanity.
#
# NOTE on --visual_und True:
#   Gen training data has no understanding-loss tokens (sequence_plan: text -> vae_image),
#   so understanding CE loss is naturally zero. visual_und stays True so that
#   validate_understanding() can run inline val/pos_acc via model.chat().
#   Turning it off would NaN/skip those metrics.
#
# NOTE on --expected_num_tokens 1:
#   Intentional. Disables sequence packing so each sample is processed individually.
#   Mirrors the 2GPU policy in train_rel_position_und_2gpu.sh.

set -euo pipefail
source ./.env

# Safety net: ensure HF cache lives on a writable disk even if .env skipped it.
: "${HF_HOME:=/mnt/data1/huggingface}"
export HF_HOME

# ============================================================
# Settings - edit here
# ============================================================
CUDA="0,1"
MASTER_PORT=${MASTER_PORT:-29603}
LR=2e-5
TOTAL_STEPS=50000
SAVE_EVERY=100
EXP_NAME=lora128_rel_position_gen
EFFECTIVE_BATCH=64
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
echo " BAGEL - rel_position gen + LoRA (${NGPUS} GPUs)"
echo "  CUDA=${CUDA}  LR=${LR}  Steps=${TOTAL_STEPS}"
echo "  Effective Batch Size=${EFFECTIVE_BATCH} (Grad Accum Steps=${GRAD_ACCUM})"
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
    --task rel_position --mode gen \
    --visual_gen True \
    --visual_und True \
    --freeze_vit True \
    --use_lora True \
    --lora_rank 128 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --num_workers 4 \
    --expected_num_tokens 1 \
    --max_num_tokens 10240 \
    --max_num_tokens_per_sample 10240 \
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
    --eval_everything \
    --eval_before_training \
    --val_gen_resolution 512 \
    --val_gen_num_images 5 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    ${RESUME_ARGS}
