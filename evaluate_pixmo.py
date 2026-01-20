import os
import argparse
import json
import torch
import random
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from transformers import AutoTokenizer, set_seed
from diffusers import VQModel

# Import from project
from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.image_utils import (
    calculate_vq_params, 
    generate_crop_size_list, 
    var_center_crop, 
    add_break_line,
    encode_img_with_breaks_fixed
)
from generators.text_understanding_generator import generate_text_understanding
from utils.prompt_utils import generate_multimodal_understanding_prompt

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
    

def extract_number_fixed(text):
    """Extract number from text pattern **number**, number, or English words (zero-nine)."""
    text = text.lower()
    
    # 1. Try **number**
    match = re.search(r"\*\*(\d+)\*\*", text)
    if match:
        return int(match.group(1))
    
    # 2. Try English words (zero to nine)
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10
    }
    for word, num in word_to_num.items():
        # Match whole word to avoid partial matches (e.g. 'one' in 'bone')
        if re.search(r"\b" + word + r"\b", text):
            return num

    # 3. Try plain digits
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
        
    return -1

def extract_number(text):
    # Try to find **<number>** pattern
    match = re.search(r'\*\*(\d+)\*\*', text)
    if match:
        return int(match.group(1))
    
    # Fallback: find any number
    match = re.search(r'\d+', text)
    if match:
        return int(match.group(0))
    
    return -1 # Not found

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pixmo Counting")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--vae_ckpt", type=str, default="./vae_ckpt", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--steps", type=int, default=128, help="Generation steps")
    parser.add_argument("--gen_length", type=int, default=1024, help="Generation length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--block_length", type=int, default=256, help="Block length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path (if any)")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution (assumed square)")

    args = parser.parse_args()

    set_all_seeds(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Dataset
    print("Loading dataset...")
    try:
        dataset = load_dataset("Jiwon-Kang/pixmo-count-filtered-imgContained", split="validation", streaming=True) 
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Load Models
    print("Loading models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.vae_ckpt, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
    )
    if args.lora_ckpt_path:
        print(f"Loading LoRA weights from {args.lora_ckpt_path}...")
        model.load_adapter(args.lora_ckpt_path)
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)
    
    # Special Tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]
    EOA = SPECIAL_TOKENS["answer_end"]
    
    # Eval loop
    true_labels = []
    pred_labels = []
    
    results = []

    print("Starting inference...")
    iterable_dataset = dataset
    for i, item in tqdm(enumerate(iterable_dataset), total=501):
        image = item['image']
        question = item['question']
        gt_answer = item['answer']
        question = question.replace('**<number>** of', '**<number>**')
        
        # Ground Truth
        gt_num = extract_number(gt_answer)
        if gt_num == -1:
            # Try to parse it loosely if exact match fails
            # Sometimes answer is just "5"
            try:
                gt_num = int(gt_answer.strip())
            except:
                pass
        
        if gt_num == -1:
            print(f"Warning: Could not extract number from GT: {gt_answer}")
            continue
            
        # Prepare Input
        input_prompt = generate_multimodal_understanding_prompt(question)
        input_ids = tokenizer(input_prompt)['input_ids']
        
        # Image Preprocessing (Logic from inference_mmu.py)
        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
        # We need to make sure image is PIL
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).resize((512,512))
        
        processed_image = var_center_crop(image, crop_size_list=crop_size_list)
        image_width, image_height = processed_image.size
        
        vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)
        
        # Encode Image
        input_img_token, (H, W) = encode_img_with_breaks_fixed(processed_image, vqvae)
        # Re-wrap logic from inference_mmu.py
        img_token = add_break_line(input_img_token[1:-1], H, W, new_number=NEW_LINE)
        img_token = [126349] + img_token + [126350]
        input_img_token = img_token
        
        # Build Sequence
        input_token = input_ids[:-1] + input_img_token + input_ids[-1:]
        code_start = len(input_token) + 1
        
        # Mask Sequence
        input_token = input_token + [BOA] + args.gen_length*[MASK]
        #  EOA = <\answer> 
        input_ids_tensor = torch.tensor(input_token, device=device).unsqueeze(0)
        
        # Generate
        out_new = generate_text_understanding(
            model, input_ids_tensor,
            steps=args.steps, 
            gen_length=args.gen_length, 
            block_length=args.block_length, 
            temperature=args.temperature, 
            remasking='low_confidence',
            code_start=code_start
        )

        text_new = tokenizer.batch_decode(
            out_new[:, code_start : -1], 
            skip_special_tokens=True
        )[0]
        print(text_new)
        pred_num = extract_number_fixed(text_new)
        
        true_labels.append(gt_num)
        pred_labels.append(pred_num)
        
        results.append({
            "question": question,
            "gt_answer": gt_answer,
            "pred_answer": text_new,
            "gt_num": gt_num,
            "pred_num": pred_num
        })

    # Metrics
    if not true_labels:
        print("No valid samples processed.")
        return

    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"Total Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix
    unique_labels = sorted(list(set(true_labels + pred_labels)))
    # Filter valid labels if they are non-negative
    valid_labels = [l for l in unique_labels if l >= 0]
    
    cm = confusion_matrix(true_labels, pred_labels, labels=valid_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=valid_labels, yticklabels=valid_labels, cmap='viridis')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix (Acc: {accuracy:.4f})')
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    print(f"Confusion matrix saved to {os.path.join(args.output_dir, 'confusion_matrix.png')}")
    
    # Save raw results
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
