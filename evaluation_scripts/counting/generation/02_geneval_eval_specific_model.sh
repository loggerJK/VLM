#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID

BASE_DIR="/mnt/data1/jiwon/geneval"
EVAL_BASE_DIR="/mnt/data1/dvlm/janus/counting/generation_eval_geneval"

# Move to a neutral directory to avoid local mmcv/mmdetection folders
cd /mnt/data1/jiwon || exit 1

echo "Starting Evaluation..."

# ============================================================
# Hard-coded parent list (EDIT THIS LIST)
# ============================================================
PARENTS=(
    "baseline"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch1"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch2"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch3"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch4"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch5"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch6"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch7"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch8"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch9"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch10"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch11"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch12"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch13"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch14"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch15"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch16"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch17"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch18"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch19"
    "train[transformer_lora128]_dset[heez_pixmo_point_count]/epoch20"
)

for parent in "${PARENTS[@]}"; do
    parent="${EVAL_BASE_DIR}/${parent}"

    if [ ! -d "$parent" ]; then
        echo "[SKIP] Not found: $parent"
        echo ""
        continue
    fi

    echo "--------------------------------------------------------"
    echo "Evaluating Model Result: $parent"
    echo "--------------------------------------------------------"

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$BASE_DIR" conda run -n geneval python -u "$BASE_DIR/evaluation/evaluate_images_counting.py" \
        "$parent" \
        --outfile "$parent/evaluation_results.jsonl" \
        --model-path "$BASE_DIR/ckpt"

    echo "Done: $parent"
    echo ""
done
