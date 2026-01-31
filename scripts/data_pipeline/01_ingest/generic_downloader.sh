#!/bin/bash

# Configure dataset parameters
SAVE_PATH="./data/pixmo_counting_filtered_imgContained"
MAX_SAMPLES=None # Set a number (e.g., 10000) or 'None' for all
PUSH_TO_HUB="Jiwon-Kang/pixmo-points-filtered-below10_imgContained" # Set repo name or leave empty to disable

mkdir -p "$(dirname "$SAVE_PATH")"

CMD="python prepare_dataset.py --save_path \"$SAVE_PATH\""

if [ "$MAX_SAMPLES" != "None" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ -n "$PUSH_TO_HUB" ]; then
    CMD="$CMD --push_to_hub \"$PUSH_TO_HUB\""
fi

eval "$CMD"
