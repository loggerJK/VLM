import os
import torch
from datasets import load_from_disk
from transformers import AutoProcessor
import numpy as np
from tqdm import tqdm

MODEL_PATH = "deepseek-ai/Janus-Pro-7B"
DATA_PATH = "data/pixmo-point-count-concat_0-10-qaFixed"

def check_tokens():
    print(f"Loading processor from {MODEL_PATH}...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        tokenizer = processor
    
    print(f"Loading dataset from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print("Dataset path does not exist.")
        return
        
    ds = load_from_disk(DATA_PATH)
    # Check split using isinstance
    import datasets
    if isinstance(ds, datasets.DatasetDict):
        if 'train' in ds:
            ds = ds['train']
        else:
            print("No train split found in DatasetDict.")
            return
    
    print(f"Dataset size: {len(ds)}")
    
    # Random sample 100 items
    sample_size = min(10, len(ds))
    print(f"Sampling {sample_size} items...")
    indices = np.random.choice(len(ds), sample_size, replace=False)
    ds_sample = ds.select(indices)
    
    print("Extracting texts...")
    questions = ds_sample['question']
    answers = ds_sample['answer']
    
    input_tokens_list = []
    new_tokens_list = []
    total_tokens_list = []
    
    print("Tokenizing...")
    # Image tokens: 576 (standard for Janus 384x384)
    num_image_tokens = 576
    
    for q, a in zip(questions, answers):
        # Text part of input
        prompt_text = f"<|User|>: <image_placeholder>\n{q}\n<|Assistant|>:"
        input_ids = tokenizer(prompt_text).input_ids
        num_input_text_tokens = len(input_ids)
        
        # Total Input = Text + Image - 1
        num_input_total = num_input_text_tokens + num_image_tokens - 1
        
        # Output tokens
        target_ids = tokenizer(a).input_ids
        num_output_tokens = len(target_ids)
        
        input_tokens_list.append(num_input_total)
        new_tokens_list.append(num_output_tokens)
        total_tokens_list.append(num_input_total + num_output_tokens)
        
    print("\n--- Token Statistics (Sample N={}) ---".format(sample_size))
    print(f"Input Tokens (User + Image):")
    print(f"  Min: {np.min(input_tokens_list)}")
    print(f"  Max: {np.max(input_tokens_list)}")
    print(f"  Mean: {np.mean(input_tokens_list):.2f}")
    print(f"  Median: {np.median(input_tokens_list):.2f}")
    print(f"  95th %: {np.percentile(input_tokens_list, 95):.2f}")
    
    print(f"\nGenerated Tokens (Assistant Answer):")
    print(f"  Min: {np.min(new_tokens_list)}")
    print(f"  Max: {np.max(new_tokens_list)}")
    print(f"  Mean: {np.mean(new_tokens_list):.2f}")
    print(f"  Median: {np.median(new_tokens_list):.2f}")
    print(f"  95th %: {np.percentile(new_tokens_list, 95):.2f}")
    
    print(f"\nTotal Context Length:")
    print(f"  Max: {np.max(total_tokens_list)}")
    print(f"  95th %: {np.percentile(total_tokens_list, 95):.2f}")

if __name__ == "__main__":
    check_tokens()