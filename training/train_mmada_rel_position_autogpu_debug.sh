#!/bin/bash
# MMaDA RelPosition Training — Single GPU (smoke test)
#
# Run from the repository root:
#   bash training/train_mmada_rel_position_1gpu.sh
#
# Defaults: 100 train steps, eval every 50 steps. All training params
# come from configs/mmada_rel_position_llada_instruct.yaml unless overridden.
#
# Override via env vars, e.g.:
#   MAX_TRAIN_STEPS=200 EVAL_EVERY=50 OUTPUT_DIR=output/rel_position_smoke \
#   bash training/train_mmada_rel_position_1gpu.sh

set -euo pipefail

cd /data/mm-llm-backbone_890/personal/sirius/audio_ablation/audio_mmada
source env.sh

export WANDB_MODE=offline
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_PROCESSES=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
    NUM_PROCESSES=$(nvidia-smi -L 2>/dev/null | awk 'END {print NR}')
    [ "${NUM_PROCESSES}" -gt 0 ] || NUM_PROCESSES=1
fi

ACCEL_CONFIG=${ACCEL_CONFIG:-accelerate_configs/auto_multi_gpu.yaml}
CONFIG=${CONFIG:-configs/mmada_rel_position_llada_instruct.yaml}
MAX_TRAIN_STEPS=62500
EVAL_EVERY=5
SAVE_EVERY=5
MAX_VAL_REL_POSITION_SAMPLES=5
OUTPUT_DIR=${OUTPUT_DIR:-}
LEARNING_RATE=${LEARNING_RATE:-}
NGPUS=${NUM_PROCESSES}
EFFECTIVE_BATCH_SIZE=64
TRAIN_BATCH_SIZE=1
GRAD_ACCUM_STEPS=$((EFFECTIVE_BATCH_SIZE / (TRAIN_BATCH_SIZE * NGPUS)))


OVERRIDES=(
    "training.max_train_steps=${MAX_TRAIN_STEPS}"
    "experiment.eval_every=${EVAL_EVERY}"
)
[ -n "${OUTPUT_DIR}" ]    && OVERRIDES+=("experiment.output_dir=${OUTPUT_DIR}")
[ -n "${LEARNING_RATE}" ] && OVERRIDES+=("optimizer.params.learning_rate=${LEARNING_RATE}")

echo "========================================"
echo "MMaDA RelPosition training (visible GPUs)"
echo "  ACCEL_CONFIG    : ${ACCEL_CONFIG}"
echo "  NUM_PROCESSES   : ${NUM_PROCESSES}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "  CONFIG          : ${CONFIG}"
echo "  MAX_TRAIN_STEPS : ${MAX_TRAIN_STEPS}"
echo "  EVAL_EVERY      : ${EVAL_EVERY}"
echo "  OUTPUT_DIR      : ${OUTPUT_DIR:-<from YAML>}"
echo "  LEARNING_RATE   : ${LEARNING_RATE:-<from YAML>}"
echo "========================================"

# rel_position 문장형 target 기준: max_seq_length 128 충분

accelerate launch --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_PROCESSES}" \
    training/train_mmada_rel_position.py \
    experiment.name="DEBUG_REL_POSITION" \
    experiment.resume_from_checkpoint=null \
    config="${CONFIG}" \
    training.gradient_accumulation_steps=${GRAD_ACCUM_STEPS} \
    training.batch_size_mmu=${TRAIN_BATCH_SIZE} \
    experiment.log_every=5 \
    experiment.save_every=${SAVE_EVERY} \
    experiment.eval_every=${EVAL_EVERY} \
    experiment.max_val_rel_position_samples=${MAX_VAL_REL_POSITION_SAMPLES} \
    dataset.preprocessing.max_seq_length=128 \
    experiment.validate_before_train=True \
    training.max_train_steps=${MAX_TRAIN_STEPS} \
    "${OVERRIDES[@]}"
