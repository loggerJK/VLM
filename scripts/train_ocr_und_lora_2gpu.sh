#!/bin/bash
# OCR understanding training — 1 GPU NO_SHARD + LoRA (rank 128)
# Dataset: Jiwon-Kang/OCR-Synthetic-Rendered-200K
#
# LoRA reduces trainable params to ~0.7%, so 1x 48GB GPU fits without CPU offload.
# To change settings, edit the variables below directly.

set -euo pipefail
# source ./.env
cd /data/mm-llm-backbone_890/personal/sirius/audio_ablation/audio_bagel
source ./env.sh
conda activate bagel

# Using HF_HOME
echo "HF_HOME is set to: ${HF_HOME}"
echo "HF_EVALUATE_CACHE is set to: ${HF_EVALUATE_CACHE}"
echo "HF_MODULES_CACHE is set to: ${HF_MODULES_CACHE}"

# ============================================================
# Settings — edit here
# ============================================================
CUDA="0,1"
MASTER_PORT=29601
LR=2e-5
TOTAL_STEPS=50000
SAVE_EVERY=100
EXP_NAME=ocr_und_lora_2gpu
HF_DATASET_PATH=Jiwon-Kang/OCR-Synthetic-Rendered-200K
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
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export PYTORCH_ALLOC_CONF=expandable_segments:True

RESULTS_DIR=./results/${EXP_NAME}
CHECKPOINT_DIR=./checkpoints/${EXP_NAME}
mkdir -p "${RESULTS_DIR}" "${CHECKPOINT_DIR}"

RESUME_ARGS=""
[ -n "${RESUME_FROM}" ]   && RESUME_ARGS="${RESUME_ARGS} --resume_from ${RESUME_FROM}"
[ -n "${LORA_CKPT_PATH}" ] && RESUME_ARGS="${RESUME_ARGS} --lora_ckpt_path ${LORA_CKPT_PATH}"
[ -n "${WANDB_RUN_ID}" ]   && RESUME_ARGS="${RESUME_ARGS} --wandb_runid ${WANDB_RUN_ID} --wandb_resume must"

echo "============================================================"
echo " BAGEL — ocr und + LoRA (${NGPUS} GPUs)"
echo "  CUDA=${CUDA}  LR=${LR}  Steps=${TOTAL_STEPS}"
echo "  Effective Batch Size=${EFFECTIVE_BATCH} (Grad Accum Steps=${GRAD_ACCUM})"
echo "  Ckpt: ${CHECKPOINT_DIR}"
echo "============================================================"

torchrun \
    --nproc_per_node=2 \
    --master_port=${MASTER_PORT} \
    ./train/pretrain_unified_navit.py \
    --model_path ./models/ \
    --finetune_from_hf True \
    --layer_module Qwen2MoTDecoderLayer \
    --use_flex False \
    --max_latent_size 64 \
    --sharding_strategy NO_SHARD \
    --num_shard 1 --num_replicate 1 \
    --task ocr --mode und \
    --hf_dataset_path "${HF_DATASET_PATH}" \
    --visual_gen False \
    --freeze_vit True \
    --use_lora True \
    --lora_rank 128 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
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
