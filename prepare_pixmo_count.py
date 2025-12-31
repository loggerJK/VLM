import os
import requests
from io import BytesIO
from PIL import Image
from datasets import load_dataset, Dataset
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import login

login("your_huggingface_token_here")  # Replace with your actual token or use environment variable

def is_valid_image(url):
    try:
        response = requests.head(url, timeout=2)
        if response.status_code == 200:
            try:
                # Image.open(BytesIO(response.content)).verify()
                return True
            except:
                return False
        return False
    except:
        return False

def download_image(url):
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            return img
    except:
        return None

def process_item(item):
    """단일 아이템 처리 및 필터링 로직"""
    count = item.get('count', 0)
    count = int(count)
    
    # 3) count <= 20 필터링
    if count > 20:
        return None
    
    # 0개 객체도 제외
    if count <= 0 :
        return None

    
    img_url = item.get('image_url')
    if not img_url:
        return None

    # 4) 이미지 다운로드 및 유효성 검사
    image = download_image(img_url)
    if image is None:
        return None
    # if not is_valid_image(img_url):
    #     return None
    
    label = item.get('label', 'object')
    
    # 2) 질문/답변 포맷팅
    # 데이터셋에는 label / count column이 존재해. label을 <object>로 간주하자.
    # question = "How many <object> are there in the image? Response Example : There are **<number>** of <object> in the image."
    # Answer = "There are **<number>** of <object> in the image."
    question = f"How many {label} are there in the image? Response Example : There are **<number>** of {label} in the image."
    answer = f"There are **{count}** of {label} in the image."
    
    return {
        "image": image,
        "image_url": img_url,
        "question": question,
        "answer": answer,
        "count": count,
        "label": label
    }

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Janus counting task")
    parser.add_argument("--save_path", type=str, default="./data/pixmo_counting_filtered", help="Path to save the processed dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to collect (optional)")
    parser.add_argument("--push_to_hub", type=str, default="Jiwon-Kang/pixmo-points-filtered-below10", help="Hugging Face Hub repo ID to push the dataset to (e.g., 'username/dataset-name')")
    args = parser.parse_args()

    save_path = args.save_path
    
    if os.path.exists(save_path):
        print(f"Dataset already exists at {save_path}. Skipping preparation.")
        if args.push_to_hub:
            print(f"Loading existing dataset from {save_path} to push to Hub...")
            hf_dataset = Dataset.load_from_disk(save_path)
            print(f"Pushing dataset to Hugging Face Hub: {args.push_to_hub}...")
            hf_dataset.push_to_hub(args.push_to_hub)
            print("Successfully pushed to Hub.")
        return

    # 1) 로드
    # print("Loading dataset in streaming mode...")
    dataset = load_dataset("allenai/pixmo-points", split="train", streaming=False)
    
    processed_data = []
    
    print("Processing and filtering data (Target: valid images with 0 < count <= 20)...")
    
    count = 0
    max_workers = 1
    
    # tqdm은 streaming dataset의 길이를 모르므로 total을 비워두거나 추정치를 넣습니다.
    pbar = tqdm(total=len(dataset), desc="Processing items")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, item) for item in dataset]
        
        for future in as_completed(futures):
            pbar.update(1)
            try:
                result = future.result()
                if result:
                    processed_data.append(result)
                    count += 1
                    pbar.set_postfix(valid_samples=len(processed_data))
                    
                    if args.max_samples and count >= args.max_samples:
                        print(f"Reached max samples limit: {args.max_samples}")
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
            except Exception as e:
                # 개별 아이템 처리 중 에러가 전체 프로세스를 멈추지 않도록 함
                continue

    print(f"Collected {len(processed_data)} valid samples.")
    
    if len(processed_data) == 0:
        print("No valid data collected. Check network or filtering conditions.")
        return

    # HuggingFace Dataset으로 변환 및 저장
    hf_dataset = Dataset.from_list(processed_data)
    hf_dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")

    if args.push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {args.push_to_hub}...")
        hf_dataset.push_to_hub(args.push_to_hub, split)
        print("Successfully pushed to Hub.")

if __name__ == "__main__":
    main()
