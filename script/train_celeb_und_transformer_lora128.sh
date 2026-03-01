#!/bin/bash

# -----------------------------------------------------------------------------
# Janus-Pro-7B Generation Training Script
# -----------------------------------------------------------------------------
# Usage:
#   bash script/train_counting_gen_transformer_lora128.sh
#   bash script/train_counting_gen_transformer_lora128.sh ./checkpoints/.../step-500  # resume
# -----------------------------------------------------------------------------

# [설정] Mode, Task & Resume (스크립트 인자)
MODE="und"
TASK="celeb"
RESUME_CKPT="${1:-}"

# [설정] WandB API Key (.env 파일에서 로드)
source ./.env
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export WANDB_NAME="train[transformer_lora128]_mode[und]_dset[celeb]"
export WANDB_PROJECT="janus-celeb"

# [설정] 사전 학습된 Janus 모델 경로
MODEL_PATH="deepseek-ai/Janus-Pro-7B"
DATA_PATH="heez/pixmo-point-count-gen-und"
GEN_DATA_PATH="heez/pixmo-point-count-gen-und"
OUTPUT_DIR="./checkpoints/${WANDB_NAME}"

# [설정] 학습 하이퍼파라미터
TUNING_MODE="transformer_lora"      # 'lora' 또는 'full'
LORA_R=128                     # LoRA Rank (TUNING_MODE가 'lora'일 때만 사용)
LORA_ALPHA=32                # LoRA Alpha (TUNING_MODE가 'lora'일 때만 사용)
BATCH_SIZE=1         # Device당 배치 사이즈
EPOCHS=100
LR=4e-5
GRAD_ACCUM_STEPS=$((128 / NUM_GPUS))     # Gradient Accumulation Steps, Total 128
USE_GRAD_CHECKPOINT=0  # 1=True, 0=False (메모리 절약)
NUM_WORKERS=32          # Dataloader Workers
SAVE_STEPS=500           # 매 n 스텝마다 체크포인트 저장
LOG_FREQ=10             # 매 n 스텝마다 Validation
USE_8BIT_ADAM=1          # 1=8bit AdamW (bitsandbytes), 0=기본 AdamW

# WandB 로그인
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging into WandB..."
    wandb login "$WANDB_API_KEY"
else
    echo "WANDB_API_KEY is not set. Assuming already logged in or disabled."
fi

# 결과 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# Multi-GPU 실행 명령어 설정 (Accelerate 사용)
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCH_CMD="accelerate launch --num_processes $NUM_GPUS --main_process_port 29500"
else
    LAUNCH_CMD="python"
fi

# Task별 인자 구성
TASK_ARGS="--task $TASK --mode $MODE --gen_data_path $GEN_DATA_PATH --gen_img_size 384"
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
echo "Data Path        : $DATA_PATH"
echo "Gen Data Path    : $GEN_DATA_PATH"
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
echo "Resume Ckpt      : ${RESUME_CKPT:-none}"
echo "Task Args        : $TASK_ARGS"
echo "================================================================"

# 스크립트 실행
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
