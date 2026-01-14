#!/bin/bash
set -e

# Activate conda environment
unset PYTHONPATH
source /home/work/.project/anaconda3/etc/profile.d/conda.sh
conda activate lumina_dimoo
export PYTHONNOUSERSITE=1

# Output directory
output_dir="output/pixmo_count_evaluation_len20"
mkdir -p "$output_dir"

echo "Starting evaluation..."
echo "Output Directory: $output_dir"

python evaluate_pixmo.py \
    --checkpoint /home/work/.project/jiwon/lumina_dimoo/output/Lumina-DiMOO-pointing-full/epoch0-iter31999 \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir "$output_dir" 2>&1 | tee "$output_dir/evaluation.log"
