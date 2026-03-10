#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

BASE_DIR="/mnt/data1/jiwon/geneval"
EVAL_BASE_DIR="/mnt/data1/dvlm/lumina/counting/generation_eval"
# RESULTS_ROOT="/mnt/data1/heeji/geneval/results"

# Move to a neutral directory to avoid local mmcv/mmdetection folders
cd /mnt/data1/jiwon || exit 1

echo "Starting Evaluation..."
# echo "Results Root: $RESULTS_ROOT"

# Iterate over all subdirectories in EVAL_BASE_DIR
parents=( "$EVAL_BASE_DIR"/*/* )

for ((i=${#parents[@]}-1; i>=0; i--)); do
    parent="${parents[i]}"
    echo "Checking directory: $parent"
    if [ ! -d "$parent" ]; then
        continue
    fi

    # Condition 1: Check if '00089' subdirectory exists
    if [ ! -d "$parent/00089" ]; then
        continue
    fi

    # Condition 2: Check if evaluation_results.jsonl does NOT exist
    if [ -f "$parent/evaluation_results.jsonl" ]; then
        continue
    fi

    echo "--------------------------------------------------------"
    echo "Evaluating Model Result: $parent"
    echo "--------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$BASE_DIR" conda run -n geneval python -u "$BASE_DIR/evaluation/evaluate_images_counting.py" \
        "$parent" \
        --outfile "$parent/evaluation_results.jsonl" \
        --model-path "$BASE_DIR/ckpt"

    echo "Done: $parent"
    echo ""
done
