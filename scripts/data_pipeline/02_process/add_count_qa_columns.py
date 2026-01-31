import os
from datasets import load_dataset, load_from_disk

# Configuration
REPO_ID = "Jiwon-Kang/pixmo-point-count-concat_0-20-qaFixed"
LOCAL_LOAD_PATH = "data/pixmo-point-count-concat_0-20-qaFixed"
LOCAL_SAVE_PATH = "data/pixmo-point-count-concat_0-20-qaFixed-updated"
NUM_PROC = 64

def process_columns(label, count, points):
    if label is None: label = "object"
    if count is None: count = 0
    if points is None: points = []
    
    # --- 1. Create New Counting QA ---
    question_count = f"How many {label} are there in the image? Response Example : There are **<number>** {label} in the image."
    answer_count = f"There are **{count}** {label} in the image."
    
    # --- 2. Update Existing Pointing QA (Remove counting info) ---
    
    # Generate points string
    formatted_points_str_list = []
    for idx, pt in enumerate(points, 1):
        try:
            px = float(pt['x'])
            py = float(pt['y'])
            formatted_points_str_list.append(f'x{idx}="{px:.1f}" y{idx}="{py:.1f}"')
        except:
            pass
    points_str = " ".join(formatted_points_str_list)
    
    # Pointing Question: Remove "and count..." and "Total number..." from example
    response_example_point = f'<points x1="<number of x1>" y1="<number of y1>" x2="<number of x2>" y2="<number of y2>" ... x_n="<number of x_n>" y_n="<number of y_n>" alt={label}>{label}</points>'
    question_point = f"Locate all {label}. Response Example : {response_example_point}"
    
    # Pointing Answer: Remove "Total number is..."
    answer_point = f'<points {points_str} alt={label}>{label}</points>'
    
    return {
        "question": question_point,
        "answer": answer_point,
        "question_count": question_count,
        "answer_count": answer_count
    }

def main():
    if os.path.exists(LOCAL_LOAD_PATH):
        print(f"Loading dataset from local disk: {LOCAL_LOAD_PATH}...")
        ds = load_from_disk(LOCAL_LOAD_PATH)
    else:
        print(f"Local path {LOCAL_LOAD_PATH} not found. Loading from Hub: {REPO_ID}...")
        ds = load_dataset(REPO_ID)
    
    print("Processing: Updating QA columns (Removing Count info) and adding Count QA...")
    ds_updated = ds.map(
        process_columns,
        input_columns=["label", "count", "points"],
        num_proc=NUM_PROC,
        desc="Refactoring QA columns"
    )
    
    print(f"Saving updated dataset locally to {LOCAL_SAVE_PATH}...")
    ds_updated.save_to_disk(LOCAL_SAVE_PATH)
    
    print(f"Pushing updated dataset to Hub: {REPO_ID}...")
    try:
        ds_updated.push_to_hub(REPO_ID)
        print("Successfully updated the dataset on the Hub!")
    except Exception as e:
        print(f"Failed to push to Hub: {e}")

if __name__ == "__main__":
    main()