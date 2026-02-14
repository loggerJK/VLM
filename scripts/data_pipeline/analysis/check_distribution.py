from datasets import load_dataset
from tqdm import tqdm

def check():
    ds = load_dataset("allenai/pixmo-points", split="train", streaming=True)
    
    total = 0
    filtered = 0
    
    # Check first 10000 samples to estimate
    limit = 10000
    for i, item in tqdm(enumerate(ds), total=limit):
        if i >= limit:
            break
        
        points = item.get('points', [])
        if len(points) <= 10:
            filtered += 1
        total += 1
        
    print(f"Total checked: {total}")
    print(f"Filtered (<= 10): {filtered}")
    print(f"Ratio: {filtered/total:.2%}")

if __name__ == "__main__":
    check()
