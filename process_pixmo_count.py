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
        response = requests.get(url, timeout=3)
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
        
        # label을 받아서 Question / Answer 형태로 변환하는 로직 추가 가능
            
        label = example.get('label', '<object>')
        count = example.get('count', 0)
        count = int(count)
        
        # 2) 질문/답변 포맷팅
        # 데이터셋에는 label / count column이 존재해. label을 <object>로 간주하자.
        # question = "How many <object> are there in the image? Response Example : There are **<number>** of <object> in the image."
        # Answer = "There are **<number>** of <object> in the image."
        question = f"How many {label} are there in the image? Response Example : There are **<number>** of {label} in the image."
        answer = f"There are **{count}** of {label} in the image."
             
        return {"image": clean_image, "question": question, "answer": answer, "valid": True}
    except Exception as e:
        return {"image": None, "question": None, "answer": None, "valid": False}

def process_dataset():
    from datasets import load_from_disk, load_dataset
    for split in ['validation', 'test' , 'train']:
        output_path = f"data/pixmo-count-filtered-imgContained/{split}"
        if os.path.exists(output_path):
            print(f"Dataset for split '{split}' already exists at {output_path}. Skipping processing.")
            # Just push to hub if not already done
            dataset = load_from_disk(output_path)
            target_repo = "Jiwon-Kang/pixmo-count-filtered-imgContained"
            print(f"Pushing dataset to Hugging Face Hub: {target_repo}...")
            dataset.push_to_hub(target_repo, split=split)
            continue
        
        dataset = load_dataset("allenai/pixmo-count", split=split)
        # dataset = dataset.filter(lambda x, idx  : idx < 100, with_indices=True)

        print(f"Initial row count: {len(dataset)}")
        

        print("Downloading images...")
        # Using num_proc=16 for network bound task
        dataset_with_images = dataset.map(download_image, num_proc=256)

        # Filter out invalid rows, using input_columns to avoid decoding images
        dataset_filtered = dataset_with_images.filter(lambda x: x["valid"], num_proc=256)
        
        print(f"Row count after filtering: {len(dataset_filtered)}")
        
        # Remove the temporary 'valid' column
        final_dataset = dataset_filtered.remove_columns(["valid"])
        
        # Save to disk
        print(f"Saving dataset to {output_path}...")
        final_dataset.save_to_disk(output_path)
        
        # Push to Hub
        target_repo = "Jiwon-Kang/pixmo-count-filtered-imgContained"
        print(f"Pushing dataset to Hugging Face Hub: {target_repo}...")
        final_dataset.push_to_hub(target_repo, split=split)
        
        print("Done.")

if __name__ == "__main__":
    process_dataset()
