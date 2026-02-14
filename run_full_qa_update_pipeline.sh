#!/bin/bash
source /home/work/.project/conda_init.sh
conda activate janus

# 1. Update QA formats and save to disk
python -u data_preprocess/update_dataset_qa.py > update_qa.log 2>&1

# 2. Start training in the background
# We need to make sure WANDB_NAME and DATA_PATH are correct
export WANDB_NAME="train[transformer_ONLY]_dset[pixmo-point-count-concatenated-qaFixed]_ngpu2_bs1_accum64_lr4e-5_ep3_full"
export DATA_PATH="./data/pixmo-point-count-concat_0-10-qaFixed"

# Use setsid to ensure training survives
setsid nohup bash script/point/train_points_transformer_ONLY.sh > train_points.log 2>&1 < /dev/null &

# 3. Push to Hub in the background
nohup python data_preprocess/push_updated_datasets.py > push_qa_fixed.log 2>&1 < /dev/null &

echo "Pipeline launched."
echo "Data update -> Training (BG) -> Hub Push (BG)"
