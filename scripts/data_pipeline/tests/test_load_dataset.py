from datasets import load_dataset, load_from_disk
import os

path = "data/pixmo-point-count-concat_0-10"

print("--- Testing load_from_disk ---")
try:
    ds = load_from_disk(path)
    print(f"Success: {ds}")
except Exception as e:
    print(f"Failed: {e}")

print("\n--- Testing load_dataset ---")
try:
    ds = load_dataset(path, split="train")
    print(f"Success: {ds}")
except Exception as e:
    print(f"Failed: {e}")
