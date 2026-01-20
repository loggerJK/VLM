#!/bin/bash

BASE_DIR="/mnt/data1/jiwon/geneval"
# RESULTS_ROOT="/mnt/data1/heeji/geneval/results"

# Move to a neutral directory to avoid local mmcv/mmdetection folders
cd /mnt/data1/jiwon || exit 1

echo "Starting Evaluation..."
# echo "Results Root: $RESULTS_ROOT"

# ============================================================
# Hard-coded parent list (EDIT THIS LIST)
# ============================================================
PARENTS=(
    # "/mnt/data1/heeji/geneval/lumina_lora128_counting"
    # "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead"
    # "$RESULTS_ROOT/debug_run"  # (optional) include if you want
    # "/mnt/data1/heeji/geneval/lumina_base_512"
    # "/mnt/data1/heeji/geneval/lumina_lora128_counting_res512"
    # "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_res512"
    "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_epoch6-iter45311-step12000"
    "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_epoch8-iter41215-step15400"
)

for parent in "${PARENTS[@]}"; do
    if [ ! -d "$parent" ]; then
        echo "[SKIP] Not found: $parent"
        echo ""
        continue
    fi

    echo "--------------------------------------------------------"
    echo "Evaluating Model Result: $parent"
    echo "--------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH="$BASE_DIR" conda run -n geneval python -u "$BASE_DIR/evaluation/evaluate_images_counting.py" \
        "$parent" \
        --outfile "$parent/evaluation_results.jsonl" \
        --model-path "$BASE_DIR/ckpt"

    echo "Done: $parent"
    echo ""
done