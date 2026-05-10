#!/bin/bash
# MMaDA RelPosition-Gen Training — 1 node × 8 GPUs (DeepSpeed ZeRO-2)
#
# Run from the repository root:
#   bash training/train_mmada_rel_position_gen_8gpu.sh
#
# All training params come from configs/mmada_rel_position_gen_llada_instruct.yaml
# unless overridden via env vars.
#
# Examples:
#   # 1) Defaults (full max_train_steps from YAML)
#   bash training/train_mmada_rel_position_gen_8gpu.sh
#
#   # 2) Override params
#   MAX_TRAIN_STEPS=20000 EVAL_EVERY=1000 LEARNING_RATE=1e-5 \
#   OUTPUT_DIR=output/rel_position_gen_lr1e-5 \
#   bash training/train_mmada_rel_position_gen_8gpu.sh
#
#   # 3) Use 4 GPUs
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   bash training/train_mmada_rel_position_gen_8gpu.sh
#
#   # 4) Resume latest checkpoint
#   RESUME=1 OUTPUT_DIR=output/rel_position_gen_lr1e-5 \
#   bash training/train_mmada_rel_position_gen_8gpu.sh

set -euo pipefail

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
MASTER_PORT=${MASTER_PORT:-8888}

ACCEL_CONFIG=${ACCEL_CONFIG:-accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml}
CONFIG=${CONFIG:-configs/mmada_rel_position_gen_llada_instruct.yaml}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-}
EVAL_EVERY=${EVAL_EVERY:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
LEARNING_RATE=${LEARNING_RATE:-}
MAX_VAL_REL_POSITION_GEN_SAMPLES=${MAX_VAL_REL_POSITION_GEN_SAMPLES:-}
MAX_VAL_REL_POSITION_UND_SAMPLES=${MAX_VAL_REL_POSITION_UND_SAMPLES:-}
VALIDATE_BEFORE_TRAIN=${VALIDATE_BEFORE_TRAIN:-True}
RESUME=${RESUME:-0}

OVERRIDES=(
    "training.batch_size_lm=0"
    "training.batch_size_mmu=0"
    "experiment.validate_before_train=${VALIDATE_BEFORE_TRAIN}"
)
[ -n "${MAX_TRAIN_STEPS}" ] && OVERRIDES+=("training.max_train_steps=${MAX_TRAIN_STEPS}")
[ -n "${EVAL_EVERY}" ]      && OVERRIDES+=("experiment.eval_every=${EVAL_EVERY}")
[ -n "${OUTPUT_DIR}" ]      && OVERRIDES+=("experiment.output_dir=${OUTPUT_DIR}")
[ -n "${LEARNING_RATE}" ]   && OVERRIDES+=("optimizer.params.learning_rate=${LEARNING_RATE}")
[ -n "${MAX_VAL_REL_POSITION_GEN_SAMPLES}" ] && OVERRIDES+=("experiment.max_val_rel_position_gen_samples=${MAX_VAL_REL_POSITION_GEN_SAMPLES}")
[ -n "${MAX_VAL_REL_POSITION_UND_SAMPLES}" ] && OVERRIDES+=("experiment.max_val_rel_position_und_samples=${MAX_VAL_REL_POSITION_UND_SAMPLES}")
if [ "${RESUME}" = "1" ]; then
    OVERRIDES+=("experiment.resume_from_checkpoint=latest")
else
    OVERRIDES+=("experiment.resume_from_checkpoint=null")
fi

echo "========================================"
echo "MMaDA RelPosition-Gen training (8 GPU)"
echo "  CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "  ACCEL_CONFIG         : ${ACCEL_CONFIG}"
echo "  CONFIG               : ${CONFIG}"
echo "  MASTER_PORT          : ${MASTER_PORT}"
echo "  MAX_TRAIN_STEPS      : ${MAX_TRAIN_STEPS:-<from YAML>}"
echo "  EVAL_EVERY           : ${EVAL_EVERY:-<from YAML>}"
echo "  OUTPUT_DIR           : ${OUTPUT_DIR:-<from YAML>}"
echo "  LEARNING_RATE        : ${LEARNING_RATE:-<from YAML>}"
echo "  MAX_VAL_GEN          : ${MAX_VAL_REL_POSITION_GEN_SAMPLES:-<from YAML>}"
echo "  MAX_VAL_UND          : ${MAX_VAL_REL_POSITION_UND_SAMPLES:-<from YAML>}"
echo "  VALIDATE_FIRST       : ${VALIDATE_BEFORE_TRAIN}"
echo "  RESUME               : ${RESUME}"
echo "========================================"

accelerate launch --config_file "${ACCEL_CONFIG}" \
    --main_process_port="${MASTER_PORT}" \
    training/train_mmada_rel_position_gen.py \
    config="${CONFIG}" \
    ${OVERRIDES[@]+"${OVERRIDES[@]}"}
