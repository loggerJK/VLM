#!/bin/bash
# MMaDA Counting-Gen Training — DEBUG (small steps, frequent eval, validate_before_train)
#
# Run from the repository root:
#   bash training/train_mmada_counting_gen_autogpu_debug.sh
#
# Override via env vars:
#   MAX_TRAIN_STEPS=20 EVAL_EVERY=5 \
#   bash training/train_mmada_counting_gen_autogpu_debug.sh

set -euo pipefail



source env.sh

echo $HF_HOME
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export WANDB_MODE=${WANDB_MODE:-online}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_PROCESSES=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
    NUM_PROCESSES=$(nvidia-smi -L 2>/dev/null | awk 'END {print NR}')
    [ "${NUM_PROCESSES}" -gt 0 ] || NUM_PROCESSES=1
fi

ACCEL_CONFIG=${ACCEL_CONFIG:-accelerate_configs/auto_multi_gpu.yaml}
CONFIG=${CONFIG:-configs/mmada_counting_gen_llada_instruct.yaml}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-50000}
EVAL_EVERY=${EVAL_EVERY:-500}
SAVE_EVERY=${SAVE_EVERY:-500}
MAX_VAL_COUNTING_GEN_SAMPLES=${MAX_VAL_COUNTING_GEN_SAMPLES:-4}
OUTPUT_DIR=${OUTPUT_DIR:-}
LEARNING_RATE=${LEARNING_RATE:-}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-512}
NGPUS=${NUM_PROCESSES}
EFFECTIVE_BATCH_SIZE=${EFFECTIVE_BATCH_SIZE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=$((EFFECTIVE_BATCH_SIZE / (TRAIN_BATCH_SIZE * NGPUS)))
[ "${GRAD_ACCUM_STEPS}" -lt 1 ] && GRAD_ACCUM_STEPS=1

OVERRIDES=(
    "training.max_train_steps=${MAX_TRAIN_STEPS}"
    "experiment.eval_every=${EVAL_EVERY}"
    "dataset.preprocessing.max_seq_length=${MAX_SEQ_LENGTH}"
)
[ -n "${OUTPUT_DIR}" ]    && OVERRIDES+=("experiment.output_dir=${OUTPUT_DIR}")
[ -n "${LEARNING_RATE}" ] && OVERRIDES+=("optimizer.params.learning_rate=${LEARNING_RATE}")

echo "========================================"
echo "MMaDA Counting-Gen DEBUG run (visible GPUs)"
echo "  ACCEL_CONFIG               : ${ACCEL_CONFIG}"
echo "  NUM_PROCESSES              : ${NUM_PROCESSES}"
echo "  CUDA_VISIBLE_DEVICES       : ${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "  CONFIG                     : ${CONFIG}"
echo "  MAX_TRAIN_STEPS            : ${MAX_TRAIN_STEPS}"
echo "  EVAL_EVERY                 : ${EVAL_EVERY}"
echo "  SAVE_EVERY                 : ${SAVE_EVERY}"
echo "  MAX_VAL_COUNTING_GEN_SAMP. : ${MAX_VAL_COUNTING_GEN_SAMPLES}"
echo "  EFFECTIVE_BATCH_SIZE       : ${EFFECTIVE_BATCH_SIZE}"
echo "  TRAIN_BATCH_SIZE (per-GPU) : ${TRAIN_BATCH_SIZE}"
echo "  GRAD_ACCUM_STEPS           : ${GRAD_ACCUM_STEPS}"
echo "  MAX_SEQ_LENGTH             : ${MAX_SEQ_LENGTH}"
echo "  OUTPUT_DIR                 : ${OUTPUT_DIR:-<from YAML>}"
echo "  LEARNING_RATE              : ${LEARNING_RATE:-<from YAML>}"
echo "========================================"

accelerate launch --config_file "${ACCEL_CONFIG}" \
    --num_processes "${NUM_PROCESSES}" \
    training/train_mmada_counting_gen.py \
    experiment.name="DEBUG_COUNTING_GEN" \
    experiment.resume_from_checkpoint=null \
    config="${CONFIG}" \
    training.gradient_accumulation_steps=${GRAD_ACCUM_STEPS} \
    training.batch_size_t2i=${TRAIN_BATCH_SIZE} \
    training.batch_size_lm=0 \
    training.batch_size_mmu=0 \
    experiment.log_every=5 \
    experiment.save_every=${SAVE_EVERY} \
    experiment.eval_every=${EVAL_EVERY} \
    experiment.max_val_counting_gen_samples=${MAX_VAL_COUNTING_GEN_SAMPLES} \
    experiment.validate_before_train=True \
    training.max_train_steps=${MAX_TRAIN_STEPS} \
    "${OVERRIDES[@]}"
