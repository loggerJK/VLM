#!/bin/bash
source /home/work/.project/conda_init.sh
conda activate janus

# -----------------------------------------------------------------------------
# Janus-Pro-7B Pointing Task Training Script
# -----------------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_NAME=${WANDB_NAME:-"train[transformer_ONLY]_dset[pixmo-point-count-concatenated]_ngpu2_bs1_accum64_lr4e-5_ep3_full"}
NUM_GPUS=2

# Assuming wandb token is already set or cached. If not, uncomment below:
# export WANDB_API_KEY="YOUR_KEY_HERE"

MODEL_PATH="deepseek-ai/Janus-Pro-7B"
DATA_PATH=${DATA_PATH:-"./data/pixmo-point-count-concat_0-10"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/${WANDB_NAME}"}

# Hyperparameters
TUNING_MODE="transformer_ONLY"
BATCH_SIZE=1
EPOCHS=3
LR=4e-5
GRAD_ACCUM_STEPS=32
USE_GRAD_CHECKPOINT=1
NUM_WORKERS=8
SAVE_STEPS=500
LOG_FREQ=50

# Create output directory
mkdir -p "$OUTPUT_DIR"

if [ "$NUM_GPUS" -gt 1 ]; then
    # Use accelerate for multi-GPU
    # Using specific port to avoid collisions
    LAUNCH_CMD="accelerate launch --num_processes $NUM_GPUS --main_process_port 29501"
else
    LAUNCH_CMD="python"
fi

echo "================================================================"
echo "Training Configuration"
echo "----------------------------------------------------------------"
echo "Model Path       : $MODEL_PATH"
echo "Data Path        : $DATA_PATH"
echo "Output Dir       : $OUTPUT_DIR"
echo "Tuning Mode      : $TUNING_MODE"
echo "Num GPUs         : $NUM_GPUS"
echo "Launch Cmd       : $LAUNCH_CMD"
echo "Grad Accum Steps : $GRAD_ACCUM_STEPS"
echo "Batch Size       : $BATCH_SIZE"
echo "Learning Rate    : $LR"
echo "================================================================"

$LAUNCH_CMD train_points.py \
    --tuning_mode "$TUNING_MODE" \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
    --gradient_checkpointing "$USE_GRAD_CHECKPOINT" \
    --save_steps "$SAVE_STEPS" \
    --log_freq "$LOG_FREQ" \
    --num_workers "$NUM_WORKERS"
