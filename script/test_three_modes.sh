#!/bin/bash
# Smoke test: counting / generation / both 모드 각각 3 step 실행
set -e

source ./.env
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=offline
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/media/dataset1/huggingface/datasets}"

MODEL_PATH="deepseek-ai/Janus-Pro-7B"
DATA_PATH="heez/pixmo-point-count-gen-und"
OUTPUT_BASE="./checkpoints/test_modes"
COMMON_ARGS="--tuning_mode transformer_lora \
    --model_path $MODEL_PATH \
    --lora_r 16 --lora_alpha 32 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing 0 \
    --num_workers 4 \
    --save_steps 9999 --log_freq 9999 \
    --epochs 1 --max_steps 3"

echo "================================================================"
echo "[TEST 1/3] mode=und, task=counting"
echo "================================================================"
python train_counting.py \
    --task counting --mode und \
    --data_path "$DATA_PATH" \
    --output_dir "${OUTPUT_BASE}/counting" \
    $COMMON_ARGS
echo "[TEST 1/3] mode=und PASSED"

echo ""
echo "================================================================"
echo "[TEST 2/3] mode=gen"
echo "================================================================"
python train_counting.py \
    --task counting --mode gen \
    --gen_data_path "$DATA_PATH" \
    --output_dir "${OUTPUT_BASE}/generation" \
    $COMMON_ARGS
echo "[TEST 2/3] mode=gen PASSED"

echo ""
echo "================================================================"
echo "[TEST 3/3] mode=both, task=counting"
echo "================================================================"
python train_counting.py \
    --task counting --mode both \
    --data_path "$DATA_PATH" \
    --gen_data_path "$DATA_PATH" \
    --output_dir "${OUTPUT_BASE}/both" \
    $COMMON_ARGS
echo "[TEST 3/3] mode=both PASSED"

echo ""
echo "================================================================"
echo "ALL 3 MODES PASSED"
echo "================================================================"

# Cleanup test checkpoints
rm -rf "${OUTPUT_BASE}"
