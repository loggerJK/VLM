from datasets import load_from_disk

data_paths = [
    "/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-count-filtered-imgContained",
    "/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-point-count-concatenated-dataset-qaFixed/train"
]

# Note: The second path pointed to 'train' subdir directly in previous ls output, 
# but usually load_from_disk expects the root if it has dataset_info.json. 
# Let's try loading the root for the second one as well if possible, or the train subdir.
# Looking at previous ls, 'pixmo-point-count-concatenated-dataset-qaFixed' had a 'train' subdir 
# and inside 'train' there was 'dataset_info.json'. So loading 'pixmo-point-count-concatenated-dataset-qaFixed' might fail if it's not standard structure.
# But 'pixmo-point-count-concatenated-dataset-qaFixed/train' has dataset_info.json.
# So I will use the path that contains dataset_info.json.

data_paths = [
    "/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-count-filtered-imgContained",
    "/home/work/.project/jiwon/deepseek-janus-pro-lora/data/pixmo-point-count-concatenated-dataset-qaFixed/train"
]

for path in data_paths:
    print(f"--- Loading {path} ---")
    try:
        ds = load_from_disk(path)
        print("Features:", ds.features)
        print("Columns:", ds.column_names)
        if 'train' in ds:
             print("Columns in train:", ds['train'].column_names)
    except Exception as e:
        print(f"Error loading {path}: {e}")
