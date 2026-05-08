#!/bin/bash
# MMaDA RelPosition Training — 1 node × 8 GPUs (DeepSpeed ZeRO-2)
#
# Run from the repository root:
#   bash training/train_mmada_rel_position_8gpu.sh
#
# All training params come from configs/mmada_rel_position_llada_instruct.yaml
# unless overridden via env vars.
#
# Examples:
#   # 1) Defaults (full 50000 steps from YAML)
#   bash training/train_mmada_rel_position_8gpu.sh
#
#   # 2) Override params
#   MAX_TRAIN_STEPS=20000 EVAL_EVERY=1000 LEARNING_RATE=1e-5 \
#   OUTPUT_DIR=output/rel_position_lr1e-5 \
#   bash training/train_mmada_rel_position_8gpu.sh
#
#   # 3) Use 4 GPUs
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   bash training/train_mmada_rel_position_8gpu.sh
#
#   # 4) Resume latest checkpoint
#   RESUME=1 OUTPUT_DIR=output/rel_position_lr1e-5 \
#   bash training/train_mmada_rel_position_8gpu.sh

set -euo pipefail

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
MASTER_PORT=${MASTER_PORT:-8888}

ACCEL_CONFIG=${ACCEL_CONFIG:-accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml}
CONFIG=${CONFIG:-configs/mmada_rel_position_llada_instruct.yaml}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-}
EVAL_EVERY=${EVAL_EVERY:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
LEARNING_RATE=${LEARNING_RATE:-}
RESUME=${RESUME:-0}

OVERRIDES=()
[ -n "${MAX_TRAIN_STEPS}" ] && OVERRIDES+=("training.max_train_steps=${MAX_TRAIN_STEPS}")
[ -n "${EVAL_EVERY}" ]      && OVERRIDES+=("experiment.eval_every=${EVAL_EVERY}")
[ -n "${OUTPUT_DIR}" ]      && OVERRIDES+=("experiment.output_dir=${OUTPUT_DIR}")
[ -n "${LEARNING_RATE}" ]   && OVERRIDES+=("optimizer.params.learning_rate=${LEARNING_RATE}")
[ "${RESUME}" = "1" ]       && OVERRIDES+=("experiment.resume_from_checkpoint=latest")

echo "========================================"
echo "MMaDA RelPosition training (8 GPU)"
echo "  CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "  ACCEL_CONFIG         : ${ACCEL_CONFIG}"
echo "  CONFIG               : ${CONFIG}"
echo "  MASTER_PORT          : ${MASTER_PORT}"
echo "  MAX_TRAIN_STEPS      : ${MAX_TRAIN_STEPS:-<from YAML>}"
echo "  EVAL_EVERY           : ${EVAL_EVERY:-<from YAML>}"
echo "  OUTPUT_DIR           : ${OUTPUT_DIR:-<from YAML>}"
echo "  LEARNING_RATE        : ${LEARNING_RATE:-<from YAML>}"
echo "  RESUME               : ${RESUME}"
echo "========================================"

accelerate launch --config_file "${ACCEL_CONFIG}" \
    --main_process_port="${MASTER_PORT}" \
    training/train_mmada_rel_position.py \
    config="${CONFIG}" \
    ${OVERRIDES[@]+"${OVERRIDES[@]}"}
