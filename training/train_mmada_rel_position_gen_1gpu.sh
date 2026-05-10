#!/bin/bash
# MMaDA RelPosition-Gen Training — Single GPU (smoke test)
#
# Run from the repository root:
#   bash training/train_mmada_rel_position_gen_1gpu.sh
#
# Defaults: 100 train steps, eval every 50 steps. All training params
# come from configs/mmada_rel_position_gen_llada_instruct.yaml unless overridden.
#
# Override via env vars, e.g.:
#   MAX_TRAIN_STEPS=200 EVAL_EVERY=50 OUTPUT_DIR=output/rel_position_gen_smoke \
#   bash training/train_mmada_rel_position_gen_1gpu.sh

set -euo pipefail

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ACCEL_CONFIG=${ACCEL_CONFIG:-accelerate_configs/1_gpu.yaml}
CONFIG=${CONFIG:-configs/mmada_rel_position_gen_llada_instruct.yaml}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-100}
EVAL_EVERY=${EVAL_EVERY:-50}
OUTPUT_DIR=${OUTPUT_DIR:-}
LEARNING_RATE=${LEARNING_RATE:-}
MAX_VAL_REL_POSITION_GEN_SAMPLES=${MAX_VAL_REL_POSITION_GEN_SAMPLES:-}
MAX_VAL_REL_POSITION_UND_SAMPLES=${MAX_VAL_REL_POSITION_UND_SAMPLES:-}
VALIDATE_BEFORE_TRAIN=${VALIDATE_BEFORE_TRAIN:-True}

OVERRIDES=(
    "training.max_train_steps=${MAX_TRAIN_STEPS}"
    "experiment.eval_every=${EVAL_EVERY}"
    "training.batch_size_lm=0"
    "training.batch_size_mmu=0"
    "experiment.validate_before_train=${VALIDATE_BEFORE_TRAIN}"
)
[ -n "${OUTPUT_DIR}" ]    && OVERRIDES+=("experiment.output_dir=${OUTPUT_DIR}")
[ -n "${LEARNING_RATE}" ] && OVERRIDES+=("optimizer.params.learning_rate=${LEARNING_RATE}")
[ -n "${MAX_VAL_REL_POSITION_GEN_SAMPLES}" ] && OVERRIDES+=("experiment.max_val_rel_position_gen_samples=${MAX_VAL_REL_POSITION_GEN_SAMPLES}")
[ -n "${MAX_VAL_REL_POSITION_UND_SAMPLES}" ] && OVERRIDES+=("experiment.max_val_rel_position_und_samples=${MAX_VAL_REL_POSITION_UND_SAMPLES}")

echo "========================================"
echo "MMaDA RelPosition-Gen training (1 GPU smoke test)"
echo "  ACCEL_CONFIG    : ${ACCEL_CONFIG}"
echo "  CONFIG          : ${CONFIG}"
echo "  MAX_TRAIN_STEPS : ${MAX_TRAIN_STEPS}"
echo "  EVAL_EVERY      : ${EVAL_EVERY}"
echo "  OUTPUT_DIR      : ${OUTPUT_DIR:-<from YAML>}"
echo "  LEARNING_RATE   : ${LEARNING_RATE:-<from YAML>}"
echo "  MAX_VAL_GEN     : ${MAX_VAL_REL_POSITION_GEN_SAMPLES:-<from YAML>}"
echo "  MAX_VAL_UND     : ${MAX_VAL_REL_POSITION_UND_SAMPLES:-<from YAML>}"
echo "  VALIDATE_FIRST  : ${VALIDATE_BEFORE_TRAIN}"
echo "========================================"

accelerate launch --config_file "${ACCEL_CONFIG}" \
    training/train_mmada_rel_position_gen.py \
    experiment.resume_from_checkpoint=null \
    config="${CONFIG}" \
    "${OVERRIDES[@]}"
