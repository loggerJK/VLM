import os
from datasets import load_from_disk
from huggingface_hub import login

# Configuration
LOCAL_PATH = "data/pixmo-point-count-concat_0-20-qaFixed-updated"
REPO_ID = "Jiwon-Kang/pixmo-point-count-concat_0-20-qaFixed"

def main():
    print("Logging in to Hugging Face Hub...")
    try:
        login(token=HF_TOKEN)
        print("Login successful.")
    except Exception as e:
        print(f"Login failed: {e}")
        return

    if not os.path.exists(LOCAL_PATH):
        print(f"Error: Local path '{LOCAL_PATH}' does not exist.")
        return

    print(f"Loading dataset from {LOCAL_PATH}...")
    try:
        ds = load_from_disk(LOCAL_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Pushing dataset to {REPO_ID}...")
    try:
        ds.push_to_hub(REPO_ID)
        print("\n[Success] Dataset successfully pushed to Hugging Face Hub!")
        print(f"View it at: https://huggingface.co/datasets/{REPO_ID}")
    except Exception as e:
        print(f"\n[Failed] Error pushing to Hub: {e}")

if __name__ == "__main__":
    main()
