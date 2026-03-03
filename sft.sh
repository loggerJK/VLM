#!/bin/bash

conda activate  blip3o


export HF_HOME=/HF/Home/
export OUTPUT_FOLDER=/Your/Model/Output/
export IMG_FOLDER=/Your/SFT/Image/Folder

# New optional arguments:
#   --use_lora                        Enable LoRA (default: False)
#   --lora_r 128                      LoRA rank
#   --lora_alpha 32                   LoRA alpha
#   --lora_target_modules "q_proj,k_proj,v_proj,o_proj"
#   --optim adamw_bnb_8bit            8-bit AdamW optimizer
#   --task counting|ocr_synthetic     Task type (uses --data_type mix by default)
#   --mode gen|und|both               Training mode
#   --validation_interval 500         Steps between validations
#   --log_freq 250                    Steps between generation logging
#   --auto_resume                     Auto-resume from latest checkpoint
#   --resume_from_checkpoint PATH     Resume from specific checkpoint


torchrun --nproc_per_node=8 \
    blip3o/train/train_mem.py \
    --deepspeed ./deepspeed_scripts/zero1.json \
    --model_name_or_path Your/Pretrain/Model  \
    --version qwen \
    --data_type "mix" \
    --image_folder ${IMG_FOLDER} \
    --gen_vision_tower eva-clip-E-14-plus \
    --gen_projector_type mlp2x_gelu \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_FOLDER} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.003 \
    --lr_scheduler_type cosine \
    --model_max_length 512 \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --gen_pooling early_pool2d_4 \
    --n_query 64 \
    --n_und_query 0 \
    --report_to none \
    --run_name blip3o_qwen_vl_sft



