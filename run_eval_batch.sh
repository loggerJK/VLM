#!/bin/bash

# Initialize Conda explicitly
source /home/cvlab22/anaconda3/etc/profile.d/conda.sh
conda activate janus

# Metadata file path
METADATA="/mnt/data1/jiwon/evaluation_metadata_count.jsonl"
JANUS_DIR="/mnt/data1/jiwon/Janus"
SCRIPT="${JANUS_DIR}/geneval_janus_evalulate_multigpu.py"

# Checkpoints list
checkpoints=(
    "/mnt/data1/jiwon/Janus/checkpoints/train[full]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-600"
    "/mnt/data1/jiwon/Janus/checkpoints/train[transformer_ONLY]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-12000"
    "/mnt/data1/jiwon/Janus/checkpoints/train[transformer_ONLY]_dset[pixmo-point-count-concatenated]_ngpu2_bs1_accum64_lr4e-5_ep3_full/checkpoint-15000"
    "/mnt/data1/jiwon/Janus/checkpoints/train[transformer_ONLY]_dset[pixmo-points_0-20]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-15000"
    "/mnt/data1/jiwon/Janus/checkpoints/train[transformer]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-600"
)

# Loop through checkpoints
for model_path in "${checkpoints[@]}"; do
    # Extract folder name and checkpoint name for output directory
    # e.g. train[full].../checkpoint-600
    parent_dir=$(basename $(dirname "$model_path"))
    ckpt_name=$(basename "$model_path")
    
    out_dir="${JANUS_DIR}/results_geneval_count_extended/${parent_dir}/${ckpt_name}"
    
    echo "----------------------------------------------------------------"
    echo "Starting evaluation for:"
    echo "Model: $model_path"
    echo "Output: $out_dir"
    echo "----------------------------------------------------------------"
    
    # Run evaluation
    # Using master_port to avoid port conflict
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29600 "$SCRIPT" \
        "$METADATA" \
        --model "$model_path" \
        --outdir "$out_dir"
        
    echo "Finished evaluation for $ckpt_name"
    echo ""
done

echo "All evaluations completed."
