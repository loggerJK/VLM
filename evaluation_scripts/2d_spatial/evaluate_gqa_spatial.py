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
    

def extract_choice(text, choices=None):
    """
    Extracts the choice from the model output, mapping it to (A) or (B).
    It first robustly checks for the text of the choices (e.g., 'left', 'right')
    and then falls back to searching for '(A)' or '(B)' or a standalone 'A' or 'B'.
    """
    if not text:
        return None
    
    clean_text_lower = text.strip().lower()

    # 1. Check if the output text matches one of the given choices.
    if choices and len(choices) == 2:
        choice_a_text = str(choices[0]).lower()
        choice_b_text = str(choices[1]).lower()

        # Use lookarounds to match whole words/phrases.
        # This prevents matching substrings inside other words (e.g., 'right' in 'copyright').
        pattern_a = r'(?<!\w)' + re.escape(choice_a_text) + r'(?!\w)'
        pattern_b = r'(?<!\w)' + re.escape(choice_b_text) + r'(?!\w)'

        found_a = re.search(pattern_a, clean_text_lower)
        found_b = re.search(pattern_b, clean_text_lower)
        
        # If both are found, the one that appears first is chosen.
        if found_a and found_b:
            if found_a.start() < found_b.start():
                return "(A)"
            else:
                return "(B)"
        elif found_a:
            return "(A)"
        elif found_b:
            return "(B)"

    # 2. If no choice text is found, fall back to extracting letter identifiers (A/B).
    
    # Search for (A) or (B), case-insensitive. This is the most explicit form.
    match = re.search(r'\(([A-B])\)', text, re.IGNORECASE)
    if match:
        return f"({match.group(1).upper()})"
    
    # Search for A or B at the start of the string, not followed by another letter.
    # This handles "A." or "A is correct".
    stripped_text = text.strip()
    if stripped_text: # check if not empty
        first_char = stripped_text[0].upper()
        if first_char in ['A', 'B']:
            if len(stripped_text) == 1 or not stripped_text[1].isalpha():
                return f"({first_char})"

    # As a last resort, find the last standalone 'A' or 'B' in the text.
    # This is common in phrases like "The answer is A".
    matches = re.findall(r'\b([A-B])\b', text, re.IGNORECASE)
    if matches:
        last_match = matches[-1].upper()
        return f"({last_match})"

    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate BLINK Spatial Relation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--vae_ckpt", type=str, default="./vae_ckpt", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="blink_eval_results", help="Output directory")
    parser.add_argument("--steps", type=int, default=128, help="Generation steps")
    parser.add_argument("--gen_length", type=int, default=1024, help="Generation length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--block_length", type=int, default=256, help="Block length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path (if any)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val or test)")

    args = parser.parse_args()

    set_all_seeds(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Dataset
    print(f"Loading CVBench spatial relation dataset (split: {args.split})...")
    try:
        dataset = load_dataset("nyu-visionx/CV-Bench", "2D", split="test")
        original_len = len(dataset)
        dataset = dataset.filter(lambda x: x['task'] == 'Relation')
        print(f"Filtered dataset to keep only 'Relation' task. {len(dataset)} examples (from {original_len}).")
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
    model.eval()
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
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        image = item['image']
        # The prompt field usually contains the question and choices
        # prompt_text = item['prompt']
        prompt_text = item['question'] + " " + " or ".join([f"{ch}" for j, ch in enumerate(item.get('choices', []))])
        gt_answer = item['answer'] # e.g., "(B)"
        choices = item.get('choices', None)
        
        # Prepare Input
        input_prompt = generate_multimodal_understanding_prompt(prompt_text)
        input_ids = tokenizer(input_prompt)['input_ids']
        
        # Image Preprocessing
        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB").resize((512,512))
        else:
            image = image.convert("RGB")
        
        processed_image = var_center_crop(image, crop_size_list=crop_size_list)
        
        # Encode Image
        input_img_token, (H, W) = encode_img_with_breaks_fixed(processed_image, vqvae)
        img_token = add_break_line(input_img_token[1:-1], H, W, new_number=NEW_LINE)
        img_token = [126349] + img_token + [126350]
        input_img_token = img_token
        
        # Build Sequence
        input_token = input_ids[:-1] + input_img_token + input_ids[-1:]
        code_start = len(input_token) + 1
        
        # Mask Sequence
        input_token = input_token + [BOA] + args.gen_length*[MASK]
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
        
        pred_choice = extract_choice(text_new, choices)
        
        true_labels.append(gt_answer)
        # If extraction fails, we mark it as incorrect by providing something that won't match
        # print question, gt_answer, pred_choice
        print(f"Q: {prompt_text} \n Choices: {choices} \n GT: {gt_answer} \n Model Answer: {text_new} \n Pred: {pred_choice}")
        pred_labels.append(pred_choice if pred_choice else "None")
        
        results.append({
            "idx": item.get('idx', i),
            "question": item['question'],
            "prompt": prompt_text,
            "gt_answer": gt_answer,
            "model_output": text_new,
            "pred_choice": pred_choice
        })
        
        
        if i % 10 == 0:
            print(f"\nModel Output: {text_new}")
            print(f"Extracted: {pred_choice} | GT: {gt_answer}")

    # Metrics
    if not true_labels:
        print("No samples processed.")
        return

    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\nTotal Accuracy: {accuracy:.4f}")
    
    # Save raw results
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()