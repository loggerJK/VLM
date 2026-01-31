import os
import datasets

# Set cache dir to larger storage
cache_dir = "/home/work/.project/cache/huggingface/datasets"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = cache_dir

from datasets import load_dataset, load_from_disk, concatenate_datasets, Features, Value, Image, Sequence
import numpy as np

# Configuration
NUM_PROC = 64
LOCAL_POINTS_PATH = "/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-points-filtered_0-20_imgContained/train"
HUB_COUNT_REPO = "Jiwon-Kang/pixmo-count-filtered-imgContained-castInt64"

REPO_0_10 = "Jiwon-Kang/pixmo-point-count-concat_0-10"
REPO_0_20 = "Jiwon-Kang/pixmo-point-count-concat_0-20"

def normalize_and_format(points, label, count):
    # points, label, count are values now, not example dict
    
    # 1. Normalize Points
    raw_points = points
    normalized_points = []
    
    if isinstance(raw_points, dict):
        xs = raw_points.get('x', [])
        ys = raw_points.get('y', [])
        if xs and ys:
            for x, y in zip(xs, ys):
                normalized_points.append({'x': x, 'y': y})
    elif isinstance(raw_points, list):
        normalized_points = raw_points
    
    # 2. Get Label and Count
    # label and count are passed directly
    if label is None: label = 'object'
    if count is None: count = 0
    
    # 3. Format Question
    question = f"Locate all {label} and count the total number of {label}"
    
    # 4. Format Answer
    formatted_points_str_list = []
    for idx, pt in enumerate(normalized_points, 1):
        try:
            px = float(pt['x'])
            py = float(pt['y'])
            formatted_points_str_list.append(f'x{idx}="{px:.1f}" y{idx}="{py:.1f}"')
        except:
            pass
            
    points_str = " ".join(formatted_points_str_list)
    answer = f'<points {points_str} alt={label}>{label}</points>. Total number of {label} is **{count}**.'
    
    return {
        "question": question,
        "answer": answer,
        "points": normalized_points,
        "count": int(count)
    }

def process_dataset(ds, range_max):
    print(f"Processing for range 0-{range_max}...")
    
    # Filter using ONLY count column
    # Filter function receives the values of input_columns
    ds_filtered = ds.filter(
        lambda count: 0 <= count <= range_max, 
        num_proc=NUM_PROC, 
        desc=f"Filter 0-{range_max}",
        input_columns=['count'] 
    )
    print(f"Filtered count: {len(ds_filtered)}")
    
    # Format using only necessary columns
    ds_formatted = ds_filtered.map(
        normalize_and_format, 
        num_proc=NUM_PROC, 
        desc=f"Format 0-{range_max}",
        input_columns=['points', 'label', 'count']
    )
    
    return ds_formatted

def main():
    # 1. Load Source Datasets
    print(f"Loading Points dataset from {LOCAL_POINTS_PATH}...")
    ds_points = load_from_disk(LOCAL_POINTS_PATH)
    print(f"Points Dataset Size: {len(ds_points)}")
    
    print(f"Loading Count dataset from {HUB_COUNT_REPO}...")
    ds_count = load_dataset(HUB_COUNT_REPO, split="train")
    print(f"Count Dataset Size: {len(ds_count)}")
    
    # Apply formatting early
    print("Formatting Source: Points...")
    ds_points = ds_points.map(
        normalize_and_format, 
        num_proc=NUM_PROC, 
        desc="Norm Points Source",
        input_columns=['points', 'label', 'count']
    )
    
    print("Formatting Source: Count...")
    ds_count = ds_count.map(
        normalize_and_format, 
        num_proc=NUM_PROC, 
        desc="Norm Count Source",
        input_columns=['points', 'label', 'count']
    )
    
    # Ensure both datasets have the same columns for concatenation
    all_cols = list(set(ds_points.column_names) | set(ds_count.column_names))
    
    # Add missing columns with None
    def add_missing_cols(dummy_val, current_cols):
        # dummy_val is the value of 'count'
        updates = {}
        for col in all_cols:
            if col not in current_cols:
                updates[col] = None
        return updates

    print("Aligning schemas...")
    if set(ds_points.column_names) != set(all_cols):
        ds_points = ds_points.map(
            lambda x: add_missing_cols(x, ds_points.column_names), 
            num_proc=NUM_PROC, 
            desc="Align Points Schema",
            input_columns=['count'] 
        )
    
    if set(ds_count.column_names) != set(all_cols):
        ds_count = ds_count.map(
            lambda x: add_missing_cols(x, ds_count.column_names), 
            num_proc=NUM_PROC, 
            desc="Align Count Schema",
            input_columns=['count'] 
        )
    
    # --- Create 0-10 Version ---
    print("\n=== Creating 0-10 Version ===")
    ds_p_10 = process_dataset(ds_points, 10)
    ds_c_10 = process_dataset(ds_count, 10)
    
    ds_concat_10 = concatenate_datasets([ds_p_10, ds_c_10])
    print(f"Concatenated 0-10 Size: {len(ds_concat_10)}")
    ds_concat_10 = ds_concat_10.shuffle(seed=42)
    
    path_10 = "data/pixmo-point-count-concat_0-10"
    if not os.path.exists(path_10):
        ds_concat_10.save_to_disk(path_10)
    try:
        ds_concat_10.push_to_hub(REPO_0_10)
    except Exception as e:
        print(f"Push failed: {e}")

    # --- Create 0-20 Version ---
    print("\n=== Creating 0-20 Version ===")
    ds_p_20 = process_dataset(ds_points, 20)
    ds_c_20 = process_dataset(ds_count, 20)
    
    ds_concat_20 = concatenate_datasets([ds_p_20, ds_c_20])
    print(f"Concatenated 0-20 Size: {len(ds_concat_20)}")
    ds_concat_20 = ds_concat_20.shuffle(seed=42)
    
    path_20 = "data/pixmo-point-count-concat_0-20"
    if not os.path.exists(path_20):
        ds_concat_20.save_to_disk(path_20)
    try:
        ds_concat_20.push_to_hub(REPO_0_20)
    except Exception as e:
        print(f"Push failed: {e}")

    print("All done.")

if __name__ == "__main__":
    main()
