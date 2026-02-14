import os
import datasets
from datasets import load_from_disk

# Configuration
CACHE_DIR = "/home/work/.project/cache/huggingface/datasets"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

# Paths
LOCAL_PATH_10 = "data/pixmo-point-count-concat_0-10"
LOCAL_PATH_20 = "data/pixmo-point-count-concat_0-20"

OUTPUT_PATH_10 = "data/pixmo-point-count-concat_0-10-qaFixed"
OUTPUT_PATH_20 = "data/pixmo-point-count-concat_0-20-qaFixed"

NUM_PROC = 64

def update_question(label):
    if label is None: label = 'object'
    
    # Generic example string as requested
    response_example = f'<points x1="<number of x1>" y1="<number of y1>" x2="<number of x2>" y2="<number of y2>" ... x_n="<number of x_n>" y_n="<number of y_n>" alt={label}>{label}</points>. Total number of {label} is **<count>**.'
    
    new_question = f"Locate all {label} and count the total number of {label}. Response Example : {response_example}"
    
    return {"question": new_question}

def process_and_save(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"[SKIP] Input path not found: {input_path}")
        return

    print(f"Loading {input_path}...")
    ds = load_from_disk(input_path)
    
    print(f"Updating Question Format...")
    # Optimization: input_columns=['label'] to avoid decoding image
    ds_updated = ds.map(update_question, num_proc=NUM_PROC, desc="Updating QA", input_columns=['label'])
    
    print(f"Saving to {output_path}...")
    ds_updated.save_to_disk(output_path)
    print(f"Saved {output_path}")

def main():
    print("Processing 0-10...")
    process_and_save(LOCAL_PATH_10, OUTPUT_PATH_10)
    
    print("Processing 0-20...")
    process_and_save(LOCAL_PATH_20, OUTPUT_PATH_20)
    
    print("All done.")

if __name__ == "__main__":
    main()