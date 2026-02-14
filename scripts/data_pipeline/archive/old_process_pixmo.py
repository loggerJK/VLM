import os
from datasets import load_dataset, Features, Image as DatasetImage, Value
from PIL import Image
import requests
from io import BytesIO

def download_image(example):
    url = example['image_url']
    if not url:
        return {"image": None, "valid": False}
        
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_data = BytesIO(response.content)
        image = Image.open(image_data)
        image.load()
        
        # Ensure image is in RGB mode
        if image.mode != "RGB":
             image = image.convert("RGB")
        
        # Strip metadata (like large ICC profiles) by recreating the image from pixels
        # This prevents "Decompressed data too large" errors downstream
        clean_image = Image.frombytes(image.mode, image.size, image.tobytes())
             
        return {"image": clean_image, "valid": True}
    except Exception as e:
        return {"image": None, "valid": False}

def process_dataset():
    from datasets import load_from_disk
    local_path = "data/pixmo_counting_filtered"
    print(f"Loading local dataset from {local_path}...")
    try:
        dataset = load_from_disk(local_path)
    except Exception as e:
        print(f"Error loading local dataset: {e}")
        return

    print(f"Initial row count: {len(dataset)}")
    
    # dataset = dataset.filter(lambda x, idx  : idx < 100, with_indices=True)

    print("Downloading images...")
    # Using num_proc=16 for network bound task
    dataset_with_images = dataset.map(download_image, num_proc=32)

    # Filter out invalid rows, using input_columns to avoid decoding images
    dataset_filtered = dataset_with_images.filter(lambda x: x["valid"])
    
    print(f"Row count after filtering: {len(dataset_filtered)}")
    
    # Remove the temporary 'valid' column
    final_dataset = dataset_filtered.remove_columns(["valid"])
    
    # Save to disk
    output_path = "data/pixmo_processed"
    print(f"Saving dataset to {output_path}...")
    final_dataset.save_to_disk(output_path)
    
    # Push to Hub
    target_repo = "Jiwon-Kang/pixmo-points-filtered-below10_imgContained"
    print(f"Pushing dataset to Hugging Face Hub: {target_repo}...")
    final_dataset.push_to_hub(target_repo)
    
    print("Done.")

if __name__ == "__main__":
    process_dataset()
