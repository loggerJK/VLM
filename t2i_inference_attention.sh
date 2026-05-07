#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=25025
ngpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')

base_dir="/mnt/data1"
model_path="deepseek-ai/Janus-Pro-7B"
prompt_file="/data/mm-llm-backbone_890/personal/heidi/ufo/abl/VLM_ori/count_subset.jsonl"
output_base="${base_dir}/dvlm/janus/counting/attention"
counting_lora="${base_dir}/jiwon/janus_counting-only/checkpoints/train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch2"
generation_lora="${base_dir}/dvlm/janus/checkpoints/train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch1"
N_SAMPLES=20

export PYTHONPATH=/mnt/data1/jiwon/Janus:${PYTHONPATH:-}

torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    inference_t2i_attention.py \
    --model_path ${model_path} \
    --prompt_file ${prompt_file} \
    --output_dir ${output_base}/baseline \
    --cfg_weight 5.0 \
    --img_size 384 \
    --n_samples ${N_SAMPLES}

torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    inference_t2i_attention.py \
    --model_path ${model_path} \
    --lora_ckpt_path ${counting_lora} \
    --prompt_file ${prompt_file} \
    --output_dir ${output_base}/counting_lora_epoch2 \
    --cfg_weight 5.0 \
    --img_size 384 \
    --n_samples ${N_SAMPLES}

torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    inference_t2i_attention.py \
    --model_path ${model_path} \
    --lora_ckpt_path ${generation_lora} \
    --prompt_file ${prompt_file} \
    --output_dir ${output_base}/generation_lora_epoch1 \
    --cfg_weight 5.0 \
    --img_size 384 \
    --n_samples ${N_SAMPLES}
