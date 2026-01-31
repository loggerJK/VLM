from datasets import load_dataset

def check_columns():
    print("Checking columns of 'Jiwon-Kang/pixmo-count-filtered-imgContained-castInt64'...")
    try:
        ds = load_dataset("Jiwon-Kang/pixmo-count-filtered-imgContained-castInt64", split="train", streaming=True)
        sample = next(iter(ds))
        print("Columns found:", list(sample.keys()))
        if 'points' in sample:
            print("Sample points:", sample['points'])
        else:
            print("'points' column NOT found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_columns()
