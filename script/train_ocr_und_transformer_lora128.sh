#!/bin/bash

# -----------------------------------------------------------------------------
# Janus-Pro-7B OCR Understanding Training Script
# -----------------------------------------------------------------------------
# Usage:
#   bash script/train_ocr_und_transformer_lora128.sh
#   bash script/train_ocr_und_transformer_lora128.sh ./checkpoints/.../step-500  # resume
# -----------------------------------------------------------------------------

# [Settings] Mode, Task & Resume
MODE="und"
TASK="ocr"
RESUME_CKPT="${1:-}"

# [Settings] WandB API Key
source ./.env
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export WANDB_NAME="train[transformer_lora128]_task[ocr]_mode[und]"
export WANDB_PROJECT="janus-ocr"

# [Settings] Model & Data
MODEL_PATH="deepseek-ai/Janus-Pro-7B"
OUTPUT_DIR="./checkpoints/${WANDB_NAME}"

# [Settings] OCR-specific
OCR_NUM_SAMPLES=200000
OCR_IMAGE_WIDTH=512

# [Settings] Training hyperparameters
TUNING_MODE="transformer_lora"
LORA_R=128
LORA_ALPHA=32
BATCH_SIZE=1
EPOCHS=100
LR=4e-5
GRAD_ACCUM_STEPS=$((128 / NUM_GPUS))
USE_GRAD_CHECKPOINT=0
NUM_WORKERS=32
SAVE_STEPS=500
LOG_FREQ=10
USE_8BIT_ADAM=1

# WandB login
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging into WandB..."
    wandb login "$WANDB_API_KEY"
else
    echo "WANDB_API_KEY is not set. Assuming already logged in or disabled."
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Multi-GPU launch command
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCH_CMD="accelerate launch --num_processes $NUM_GPUS --main_process_port 29500"
else
    LAUNCH_CMD="python"
fi

# Task-specific args
TASK_ARGS="--task $TASK --mode $MODE --ocr_num_samples $OCR_NUM_SAMPLES --ocr_image_width $OCR_IMAGE_WIDTH"
if [ -n "$RESUME_CKPT" ]; then
    TASK_ARGS="$TASK_ARGS --resume_checkpoint $RESUME_CKPT"
fi
if [ "$USE_8BIT_ADAM" -eq 1 ]; then
    TASK_ARGS="$TASK_ARGS --use_8bit_adam"
fi

echo "================================================================"
echo "Training Configuration"
echo "----------------------------------------------------------------"
echo "Model Path       : $MODEL_PATH"
echo "Output Dir       : $OUTPUT_DIR"
echo "Tuning Mode      : $TUNING_MODE"
echo "Num GPUs         : $NUM_GPUS"
echo "Launch Cmd       : $LAUNCH_CMD"
echo "Grad Accum Steps : $GRAD_ACCUM_STEPS"
echo "Batch Size       : $BATCH_SIZE"
echo "Learning Rate    : $LR"
echo "Grad Checkpoint  : $USE_GRAD_CHECKPOINT"
echo "Save Steps       : $SAVE_STEPS"
echo "Epochs           : $EPOCHS"
echo "Log Freq         : $LOG_FREQ"
echo "Mode             : $MODE"
echo "Task             : $TASK"
echo "OCR Num Samples  : $OCR_NUM_SAMPLES"
echo "OCR Image Width  : $OCR_IMAGE_WIDTH"
echo "Resume Ckpt      : ${RESUME_CKPT:-none}"
echo "Task Args        : $TASK_ARGS"
echo "================================================================"

# Run training
$LAUNCH_CMD train_counting.py \
    --tuning_mode "$TUNING_MODE" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --gradient_checkpointing "$USE_GRAD_CHECKPOINT" \
    --save_steps "$SAVE_STEPS" \
    --log_freq "$LOG_FREQ" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --run_name "$WANDB_NAME" \
    $TASK_ARGS
