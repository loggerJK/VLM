import os
import datasets
from datasets import load_from_disk
from huggingface_hub import login


OUTPUT_PATH_10 = "data/pixmo-point-count-concat_0-10-qaFixed"
OUTPUT_PATH_20 = "data/pixmo-point-count-concat_0-20-qaFixed"

REPO_10 = "Jiwon-Kang/pixmo-point-count-concat_0-10-qaFixed"
REPO_20 = "Jiwon-Kang/pixmo-point-count-concat_0-20-qaFixed"

def main():
    login(token=HF_TOKEN)
    
    if os.path.exists(OUTPUT_PATH_10):
        print(f"Pushing {OUTPUT_PATH_10} to Hub...")
        ds = load_from_disk(OUTPUT_PATH_10)
        ds.push_to_hub(REPO_10)
        
    if os.path.exists(OUTPUT_PATH_20):
        print(f"Pushing {OUTPUT_PATH_20} to Hub...")
        ds = load_from_disk(OUTPUT_PATH_20)
        ds.push_to_hub(REPO_20)

if __name__ == "__main__":
    main()
