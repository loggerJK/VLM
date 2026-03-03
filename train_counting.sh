#!/bin/bash

# Usage:
#   bash train_counting.sh gen                              # Fresh start, all 8 GPUs
#   bash train_counting.sh gen output/epoch0_step-500       # Resume from explicit checkpoint
#   bash train_counting.sh both                             # Train both gen + und
#   CUDA_VISIBLE_DEVICES=4,5 bash train_counting.sh gen     # Use GPU 4,5 only
#
# Auto-resume is enabled by default (--auto_resume). If a checkpoint exists in OUTPUT_FOLDER,
# training will resume automatically.
# Gradient accumulation is auto-adjusted to keep effective batch size = 128.

source ./.env

# ──── GPU Configuration ────
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

MODE="${1:-gen}"
TASK="counting"
RESUME_CKPT="${2:-}"

# export HF_HOME=${HF_HOME:-/mnt/data1/jiwon/.cache/huggingface}
export OUTPUT_FOLDER=${OUTPUT_FOLDER:-/mnt/data1/jiwon/BLIP3o/output/counting_${MODE}}
export WANDB_PROJECT="blip3o-counting"
export WANDB_NAME="train[lora128]_mode[${MODE}]_task[${TASK}]_dset[pixmo_count]_gpu${NUM_GPUS}"

# Adjust gradient_accumulation to keep effective batch=128
# effective_batch = per_device_batch * grad_accum * num_gpus = 1 * GRAD_ACCUM * NUM_GPUS
GRAD_ACCUM=$((128 / NUM_GPUS))

mkdir -p ${OUTPUT_FOLDER}

echo ">>> GPUs: ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} devices)"
echo ">>> Gradient accumulation: ${GRAD_ACCUM} (effective batch: ${NUM_GPUS} x 1 x ${GRAD_ACCUM} = $((NUM_GPUS * GRAD_ACCUM)))"
echo ">>> Output: ${OUTPUT_FOLDER}"

RESUME_ARGS=""
if [ -n "$RESUME_CKPT" ]; then
    RESUME_ARGS="--resume_ckpt $RESUME_CKPT"
fi

torchrun --nproc_per_node=$NUM_GPUS \
    blip3o/train/train.py \
    --model_name_or_path /home/cvlab22/.cache/huggingface/hub/models--BLIP3o--BLIP3o-Model-8B/snapshots/c2edfc20814d4624c8d73ca3de351ebc3fa86508  \
    --version qwen \
    --task counting \
    --mode $MODE \
    --data_type counting \
    --hf_dataset_path heez/pixmo-point-count-gen-und \
    --gen_vision_tower eva-clip-E-14-plus \
    --gen_projector_type mlp2x_gelu \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --train_gen_components latent_queries \
    --optim adamw_bnb_8bit \
    --bf16 True \
    --output_dir ${OUTPUT_FOLDER} \
    --num_train_epochs 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_steps 250 \
    --validation_interval 250 \
    --log_freq 250 \
    --validation_samples 100 \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --lr_scheduler_type cosine \
    --model_max_length 2048 \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --dataloader_num_workers 4 \
    --gen_pooling early_pool2d_4 \
    --n_query 64 \
    --n_und_query 0 \
    --report_to wandb \
    --auto_resume \
    --run_name "$WANDB_NAME" \
    $RESUME_ARGS
    # --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \

