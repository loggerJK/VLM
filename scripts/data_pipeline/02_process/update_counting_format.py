import os
from datasets import load_from_disk

# Configuration
INPUT_PATH = "data/pixmo-point-count-concat_0-20-qaFixed-updated"
OUTPUT_PATH = "data/pixmo-point-count-concat_0-20-qaFixed-final"
NUM_PROC = 64

def format_points_string(points, label):
    """
    Convert points list data to <points x1="..." y1="..." ... alt=label>label</points> format
    """
    formatted_points = []
    # points format: [{'x': 10.1, 'y': 20.2}, ...]
    for idx, pt in enumerate(points, 1):
        try:
            px = float(pt['x'])
            py = float(pt['y'])
            formatted_points.append(f'x{idx}="{px:.1f}" y{idx}="{py:.1f}"')
        except (KeyError, ValueError, TypeError):
            continue
            
    points_str = " ".join(formatted_points)
    # Using double quotes for alt text consistently
    return f'<points {points_str} alt="{label}">{label}</points>'

def regenerate_qa(example):
    label = example.get('label', 'object')
    count = example.get('count', 0)
    points = example.get('points', [])
    
    if label is None: label = 'object'
    if count is None: count = 0
    
    # 1. Regenerate Points XML from raw data (Safest way)
    points_xml = format_points_string(points, label)
    
    # 2. Define New Counting Sentences
    # For Answer: include actual count
    answer_count_sentence = f"There are **{count}** {label} in the image."
    # For Question (Response Example): use <number> placeholder
    question_count_example = f"There are **<number>** {label} in the image."
    
    # 3. Construct New Question
    # Instructions + Response Example (points + new counting format example)
    new_question = (
        f"Locate all {label} and count the total number of {label}. "
        f"Response Example : {points_xml}. {question_count_example}"
    )
    
    # 4. Construct New Answer
    # points + new counting format with actual count
    new_answer = f"{points_xml}. {answer_count_sentence}"
    
    # Note: question_count and answer_count columns are kept as-is automatically 
    # since we only return the keys we want to update in this function.
    return {
        "question": new_question,
        "answer": new_answer
    }

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input path not found: {INPUT_PATH}")
        return

    print(f"Loading dataset from {INPUT_PATH}...")
    ds = load_from_disk(INPUT_PATH)
    
    print("Regenerating Question and Answer using 'label', 'count', and 'points' columns...")
    
    # Use map to regenerate based on raw columns
    ds_updated = ds.map(
        regenerate_qa,
        num_proc=NUM_PROC,
        desc="Regenerating QA with New Format"
    )
    
    print(f"Saving finalized dataset to {OUTPUT_PATH}...")
    ds_updated.save_to_disk(OUTPUT_PATH)
    print("Done. You can now push this to the Hub if needed.")

if __name__ == "__main__":
    main()
