#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

BASE_DIR="/mnt/data1/jiwon/geneval"
EVAL_BASE_DIR="/mnt/data1/dvlm/lumina/counting/generation_eval"
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
    # "/mnt/data1/heeji/geneval/lumina_base_512"
    # "/mnt/data1/heeji/geneval/lumina_lora128_counting_res512"
    # "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_res512"
    # "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_epoch6-iter45311-step12000"
    # "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_epoch8-iter41215-step15400"
    # "/mnt/data1/heeji/geneval/lora128_counting_wohead_range0-7_resumeEpoch10_lr3e-6_epoch15"
    # "/mnt/data1/heeji/geneval/lumina_train_generation_wohead_1024_epoch4-iter3583-step2100"
    # "/mnt/data1/heeji/geneval/lora128_counting_wohead_range0-7_resumeEpoch10_lr3e-6_epoch13"
    # "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_resumeEpoch12_lr3e-6_epoch13"
    # "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_resumeEpoch12_lr3e-6_epoch14"
    # "/mnt/data1/heeji/geneval/lumina_lora128_counting_wohead_resumeEpoch12_lr3e-6_epoch15"

    "lumina_train_generation_wohead_1024/epoch0"
    "lumina_train_generation_wohead_1024/epoch1"
    "lumina_train_generation_wohead_1024/epoch2"
    "lumina_train_generation_wohead_1024/epoch3"
    "lumina_train_generation_wohead_1024/epoch4"
    "lora128_counting_wohead_both/epoch0"
    "lora128_counting_wohead_both/epoch1"
    "lora128_counting_wohead_both/epoch2"
    "lora128_counting_wohead_both/epoch3"
    "lora128_counting_wohead_both/epoch4"
)

for parent in "${PARENTS[@]}"; do
    # reference:  output_dir_name=$(echo "$ckpt" | tr '/' '_')
    parent="${EVAL_BASE_DIR}/lumina_${parent}"
    
    if [ ! -d "$parent" ]; then
        echo "[SKIP] Not found: $parent"
        echo ""
        continue
    fi

    echo "--------------------------------------------------------"
    echo "Evaluating Model Result: $parent"
    echo "--------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$BASE_DIR" conda run -n geneval python -u "$BASE_DIR/evaluation/evaluate_images_counting.py" \
        "$parent" \
        --outfile "$parent/evaluation_results.jsonl" \
        --model-path "$BASE_DIR/ckpt"

    echo "Done: $parent"
    echo ""
done