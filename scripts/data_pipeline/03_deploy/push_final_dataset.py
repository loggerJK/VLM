import os
from datasets import load_from_disk
from huggingface_hub import login

# Configuration
LOCAL_PATH = "data/pixmo-point-count-concat_0-20-qaFixed-v5"
REPO_ID = "Jiwon-Kang/pixmo-point-count-concat_0-20-qaFixed"

def main():
    if not os.path.exists(LOCAL_PATH):
        print(f"Error: Local dataset not found at {LOCAL_PATH}")
        print("Please run 'scripts/data_pipeline/02_process/update_counting_format.py' first.")
        return

    print(f"Loading dataset from {LOCAL_PATH}...")
    ds = load_from_disk(LOCAL_PATH)
    
    print(f"Pushing to Hub: {REPO_ID}")
    
    try:
        # max_shard_size handles the splitting of large files during push
        ds.push_to_hub(REPO_ID)
        print("Successfully pushed to Hub!")
    except Exception as e:
        print(f"Failed to push to Hub: {e}")
        print("Tip: Ensure you are logged in using `huggingface-cli login` or have set the HF_TOKEN environment variable.")

if __name__ == "__main__":
    main()
