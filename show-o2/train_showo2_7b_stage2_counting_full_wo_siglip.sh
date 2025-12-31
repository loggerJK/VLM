#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="showo2-7b-stage-2-counting"
export WANDB_API_KEY="f9831e23517e27f7ecac9b54bc2cdcabb3af8c33" 
export WANDB_NAME="train[full_wo_siglip]_dset[pixmo_4-20]_ngpu1_bs1_accum128_ep3_lr1e-5"
# export WANDB_NAME="TEST"



# Config: configs/showo2_7b_stage_2_counting.yaml
# Dataset path is passed as an argument as requested

accelerate launch \
    --num_processes 1 \
    --mixed_precision 'no' \
    train_stage_two_counting.py \
    config=configs/showo2_7b_stage_2_counting_full_wo_siglip.yaml 
    # --config_file /home/work/.project/jiwon/showo/accelerate_configs/8_gpus_deepspeed_zero2.yaml \
# python -m pdb \
