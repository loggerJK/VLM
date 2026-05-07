    #!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=25015
ngpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')

base_dir="/mnt/data1"
model_path="deepseek-ai/Janus-Pro-7B"
checkpoint_base="${base_dir}/jiwon/janus_counting-only/checkpoints"
output_base="${base_dir}/dvlm/janus/counting/generation_eval_geneval_more_samples"
metadata_file="${base_dir}/jiwon/geneval/prompts/evaluation_metadata_count.jsonl"
N_SAMPLES=20

export PYTHONPATH=/mnt/data1/jiwon/Janus:${PYTHONPATH:-}

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

echo "========================================"
echo "Running baseline GenEval counting generation evaluation"
echo "========================================"

torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    evaluation_scripts/counting/generation/evaluate_counting_generation_multigpu.py \
    --model_path ${model_path} \
    --metadata_file ${metadata_file} \
    --output_dir ${output_base}/baseline \
    --cfg_weight 5.0 \
    --img_size 384 \
    --n_samples ${N_SAMPLES} \
    --skip_grid

# ---------------------------------------------------------------------------- #
#                              Checkpoint sweep                                #
# ---------------------------------------------------------------------------- #

checkpoint_base="${base_dir}/jiwon/janus_counting-only/checkpoints"
ckpt_list=(
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch1"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch2"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch3"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch4"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch5"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch6"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch7"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch8"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch9"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch10"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch11"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch12"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch13"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch14"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch15"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch16"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch17"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch18"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch19"
    # "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch20"
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running GenEval counting generation for checkpoint: $ckpt"
    echo "Output directory: ${output_base}/${ckpt}"
    echo "========================================"

    torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        evaluation_scripts/counting/generation/evaluate_counting_generation_multigpu.py \
        --model_path ${model_path} \
        --lora_ckpt_path ${checkpoint_base}/${ckpt} \
        --metadata_file ${metadata_file} \
        --output_dir ${output_base}/${ckpt} \
        --cfg_weight 5.0 \
        --img_size 384 \
        --n_samples ${N_SAMPLES}

done


checkpoint_base="${base_dir}/dvlm/janus/checkpoints"
ckpt_list=(
    "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch1"
    # "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch2"
    # "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch3"
    # "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch4"
    # "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch5"
    # "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch6"
    # "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch7"
    # "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch8"
    # "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch9"
    # "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch10"
    # "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch11"
    # "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch12"
    # "train[transformer_lora128]_mode[gen]_dset[heez_pixmo_point_count]/epoch13"
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running GenEval counting generation for checkpoint: $ckpt"
    echo "Output directory: ${output_base}/${ckpt}"
    echo "========================================"

    torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        evaluation_scripts/counting/generation/evaluate_counting_generation_multigpu.py \
        --model_path ${model_path} \
        --lora_ckpt_path ${checkpoint_base}/${ckpt} \
        --metadata_file ${metadata_file} \
        --output_dir ${output_base}/${ckpt} \
        --cfg_weight 5.0 \
        --img_size 384 \
        --n_samples ${N_SAMPLES}

done
