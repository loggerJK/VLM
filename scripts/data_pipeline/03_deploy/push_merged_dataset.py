import os
from datasets import load_from_disk
from huggingface_hub import login

# Set cache dir to larger storage just in case
cache_dir = "/home/work/.project/cache/huggingface/datasets"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = cache_dir

# Token provided by user

def main():
    print("Logging in to Hugging Face Hub...")
    login(token=HF_TOKEN)
    
    # 1. Push 0-10 Dataset
    local_path_10 = "data/pixmo-point-count-concat_0-10"
    repo_10 = "Jiwon-Kang/pixmo-point-count-concat_0-10"
    
    if os.path.exists(local_path_10):
        print(f"Loading {local_path_10}...")
        ds_10 = load_from_disk(local_path_10)
        print(f"Pushing to {repo_10}...")
        try:
            ds_10.push_to_hub(repo_10)
            print(f"Successfully pushed {repo_10}")
        except Exception as e:
            print(f"Failed to push {repo_10}: {e}")
    else:
        print(f"Directory not found: {local_path_10}")

    # 2. Push 0-20 Dataset
    local_path_20 = "data/pixmo-point-count-concat_0-20"
    repo_20 = "Jiwon-Kang/pixmo-point-count-concat_0-20"
    
    if os.path.exists(local_path_20):
        print(f"Loading {local_path_20}...")
        ds_20 = load_from_disk(local_path_20)
        print(f"Pushing to {repo_20}...")
        try:
            ds_20.push_to_hub(repo_20)
            print(f"Successfully pushed {repo_20}")
        except Exception as e:
            print(f"Failed to push {repo_20}: {e}")
    else:
        print(f"Directory not found: {local_path_20}")

if __name__ == "__main__":
    main()
