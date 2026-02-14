import os
from datasets import load_dataset, Features, Value, Image
import requests
from io import BytesIO
from PIL import Image as PILImage
import random

def process_and_download(example):
    # Helper to return consistent schema for invalid rows
    def return_invalid():
        return {
            "image": None, 
            "question": None, 
            "answer": None, 
            "count": None, 
            "label": None, 
            "points": None, 
            "valid": False
        }

    # 1. Check count limit (Redundant if filtered before, but safe)
    points = example.get('points', [])
    if len(points) > 10:
        return return_invalid()
    
    # 2. Download Image
    url = example.get('image_url')
    if not url:
        return return_invalid()
        
    try:
        response = requests.get(url, timeout=3)
        if response.status_code != 200:
             return return_invalid()
             
        image_data = BytesIO(response.content)
        image = PILImage.open(image_data)
        image.load()
        if image.mode != "RGB":
             image = image.convert("RGB")
        # Strip metadata
        clean_image = PILImage.frombytes(image.mode, image.size, image.tobytes())
        
        label = example.get('label', 'object')
        
        # 3. Format
        question = f"Locate all {label} and count the total number of {label}"
        
        formatted_points = []
        for idx, pt in enumerate(points, 1):
            px = float(pt['x'])
            py = float(pt['y'])
            formatted_points.append(f'x{idx}="{px:.1f}" y{idx}="{py:.1f}"')
        
        points_str = " ".join(formatted_points)
        count = len(points)
        answer = f'<points {points_str} alt={label}>{label}</points>. Total number of {label} is **{count}**.'
        
        return {
            "image": clean_image, 
            "question": question, 
            "answer": answer, 
            "count": count,
            "label": label,
            "points": points,
            "valid": True
        }
    except Exception:
        return return_invalid()

def main():
    print("Loading dataset...")
    # Load metadata only first
    ds = load_dataset("allenai/pixmo-points", split="train")
    
    print(f"Original size: {len(ds)}")
    
    # Filter first by count to avoid downloading unnecessary images
    ds_filtered = ds.filter(lambda x: len(x['points']) <= 10, num_proc=64)
    print(f"Filtered size (count <= 10): {len(ds_filtered)}")
    
    # Now map to download images
    print("Downloading images and formatting...")
    # Using reasonably high proc for IO
    ds_with_images = ds_filtered.map(process_and_download, num_proc=64)
    
    # Filter out invalid downloads
    ds_final = ds_with_images.filter(lambda x: x["valid"], num_proc=64)
    print(f"Final valid size: {len(ds_final)}")
    
    # Remove 'valid' and 'image_url' if not needed, keep others
    columns_to_keep = ["image", "question", "answer", "count", "label", "points"]
    ds_final = ds_final.select_columns(columns_to_keep)
    
    # Split
    print("Splitting dataset...")
    # Use 1000 for validation
    split_ds = ds_final.train_test_split(test_size=1000, seed=42)
    split_ds['validation'] = split_ds.pop('test') # Rename test to validation
    
    # Save to disk
    output_path = "data/pixmo-point-imgContained_0-10"
    if not os.path.exists(output_path):
        print(f"Saving to {output_path}...")
        split_ds.save_to_disk(output_path)
    
    # Push to Hub
    target_repo = "Jiwon-Kang/pixmo-point-imgContained_0-10"
    print(f"Pushing to Hub: {target_repo}...")
    try:
        split_ds.push_to_hub(target_repo)
    except Exception as e:
        print(f"Failed to push to hub: {e}")
    
    print("Done.")

if __name__ == "__main__":
    main()