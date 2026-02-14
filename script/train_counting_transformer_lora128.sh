#!/bin/bash

# -----------------------------------------------------------------------------
# Janus-Pro-7B Counting Task Training Script
# -----------------------------------------------------------------------------

# [설정] WandB API Key (.env 파일에서 로드)
source ./.env
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export WANDB_NAME="train[transformer_lora128]_dset[heez_pixmo_point_count]"

# [설정] 사전 학습된 Janus 모델 경로
# 실제 모델 가중치(config.json, pytorch_model.bin 등)가 있는 디렉토리로 변경해주세요.
MODEL_PATH="deepseek-ai/Janus-Pro-7B"
DATA_PATH="heez/pixmo-point-count-gen-und"
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
LOG_FREQ=250             # 매 n 스텝마다 Validation
TASK="counting"          # 'pointing' (question/answer) or 'counting' (question_count/answer_count)

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
# NUM_GPUS가 1보다 크면 accelerate launch를 사용하여 DDP 학습 진행
if [ "$NUM_GPUS" -gt 1 ]; then
    # --multi_gpu 옵션은 accelerate config가 없을 때 유용할 수 있지만,
    # 명시적으로 num_processes를 주는 것이 안전합니다.
    # 포트 충돌 방지를 위해 main_process_port를 랜덤하게 설정하는 것도 좋습니다.
    LAUNCH_CMD="accelerate launch --num_processes $NUM_GPUS --main_process_port 29500"
else
    LAUNCH_CMD="python -m pdb"
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
echo "Batch Size      : $BATCH_SIZE"
echo "Learning Rate    : $LR"
echo "Grad Checkpoint  : $USE_GRAD_CHECKPOINT"
echo "Save Steps       : $SAVE_STEPS"
echo "Epochs           : $EPOCHS"
echo "Log Freq         : $LOG_FREQ"
echo "Task             : $TASK"
echo "================================================================"

# 스크립트 실행
$LAUNCH_CMD train_counting.py \
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
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --task "$TASK" 
