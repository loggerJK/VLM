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
import torch.distributed as dist
from datetime import timedelta

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

# No gradient
torch.set_grad_enabled(False)

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
    

def extract_choice(text, choices):
    """
    Extracts the choice from the model output, mapping it to (A), (B), (C), etc.
    It restricts the search to the valid range of choices provided.
    """
    if not text:
        return None
    
    clean_text_lower = text.strip().lower()
    valid_indices = range(len(choices)) if choices else range(26) # Default to A-Z if no choices given (fallback) 
    valid_letters = [chr(ord('A') + i) for i in valid_indices]

    # 1. Check if the output text matches one of the given choices content.
    if choices:
        found_matches = []
        for i, choice in enumerate(choices):
            choice_text = str(choice).lower()
            # Use lookarounds to match whole words/phrases.
            pattern = r'(?<!\w)' + re.escape(choice_text) + r'(?!\w)'
            match = re.search(pattern, clean_text_lower)
            if match:
                found_matches.append((match.start(), chr(ord('A') + i)))
        
        # If matches are found, the one that appears first is chosen.
        if found_matches:
            found_matches.sort()
            return f"({found_matches[0][1]})"

    # 2. If no choice text is found, fall back to extracting letter identifiers (A, B, C...). 
    # We only accept letters that are valid for the given choices.
    
    # Search for (X), case-insensitive.
    matches = re.findall(r'\(([A-Z])\)', text, re.IGNORECASE)
    for m in matches:
        if m.upper() in valid_letters:
            return f"({m.upper()})"
    
    # Search for X at the start of the string.
    stripped_text = text.strip()
    if stripped_text:
        first_char = stripped_text[0].upper()
        if first_char in valid_letters:
            if len(stripped_text) == 1 or not stripped_text[1].isalpha():
                return f"({first_char})"

    # Find the last standalone letter in the text.
    matches = re.findall(r'\b([A-Z])\b', text, re.IGNORECASE)
    if matches:
        # Check in reverse order
        for m in reversed(matches):
            if m.upper() in valid_letters:
                return f"({m.upper()})"

    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate CVBench Counting Multigpu")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--vae_ckpt", type=str, default="./vae_ckpt", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--steps", type=int, default=128, help="Generation steps")
    parser.add_argument("--gen_length", type=int, default=1024, help="Generation length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--block_length", type=int, default=256, help="Block length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora_ckpt_path", type=str, default=None, help="LoRA checkpoint path (if any)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (val or test)")

    args = parser.parse_args()
    
    # Initialize Distributed Training
    if "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(days=1))
        device = torch.device("cuda", local_rank)
        print(f"[Rank {rank}] Distributed initialized. World Size: {world_size}")
    else:
        rank = 0
        world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Distributed environment not detected. Running in single process mode.")

    set_all_seeds(args.seed)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
        
    # Check if results.json file already exists (Skip if so)
    output_json_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(output_json_path):
        if rank == 0:
            print(f"Results file {output_json_path} already exists. Skipping inference.")
        if world_size > 1:
            dist.destroy_process_group()
        return

    # Load Dataset
    print(f"[Rank {rank}] Loading CVBench counting dataset (split: {args.split})...")
    try:
        dataset = load_dataset("nyu-visionx/CV-Bench", "2D", split="test")
        original_len = len(dataset)
        dataset = dataset.filter(lambda x: x['task'] == 'Count')
        if rank == 0:
            print(f"Filtered dataset to keep only 'Count' task. {len(dataset)} examples (from {original_len}).")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Load Models
    print(f"[Rank {rank}] Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.vae_ckpt, trust_remote_code=True)
    
    # Use specific device map for distributed
    if world_size > 1:
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map=device_map,
    )
    if args.lora_ckpt_path:
        print(f"[Rank {rank}] Loading LoRA weights from {args.lora_ckpt_path}...")
        model.load_adapter(args.lora_ckpt_path)
    model.eval()
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)

    # Special Tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]
    EOA = SPECIAL_TOKENS["answer_end"]

    results = []

    print(f"[Rank {rank}] Starting inference...")
    print(f"Dataset size: {len(dataset)}")
    
    # Create local dataset shard for distributed inference
    local_indices = list(range(rank, len(dataset), world_size))
    local_dataset = dataset.select(local_indices)

    # Use tqdm with accurate total for this rank
    iterator = tqdm(local_dataset, total=len(local_dataset), desc=f"Rank {rank}")
    
    for item in iterator:
        # IMPORTANT: Fix seed per sample to ensure consistency regardless of GPU count
        set_all_seeds(args.seed)

        image = item['image']
        prompt_text_raw = item['prompt']
        gt_answer = item['answer'] # e.g., "(B)"
        choices = item.get('choices', [])
        
        # Construct prompt with choices
        prompt_text = prompt_text_raw
        if choices:
            prompt_text += "\nSelect from the following choices."
            for c_idx, choice in enumerate(choices):
                letter = chr(ord('A') + c_idx)
                prompt_text += f"\n({letter}) {choice}"

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
        
        print(f"Rank {rank} - Prompt: {prompt_text_raw}")
        print(f"Rank {rank} - Generated Text: {text_new}")
        print(f"Rank {rank} - Ground Truth Answer: {gt_answer}")
        print("-----------------------------------------------------")
        
        pred_choice = extract_choice(text_new, choices)
        
        results.append({
            "prompt": prompt_text_raw,
            "gt_answer": gt_answer,
            "pred_answer": text_new,
            "pred_choice": pred_choice,
            "choices": choices
        })

    # Gather results from all ranks
    if world_size > 1:
        all_results = [None for _ in range(world_size)]
        dist.barrier()
        dist.all_gather_object(all_results, results)
        # Flatten list of lists
        if rank == 0:
            results = [item for sublist in all_results for item in sublist]
    
    # Process metrics only on rank 0
    if rank == 0:
        true_labels = [r['gt_answer'] for r in results]
        pred_labels = [r['pred_choice'] for r in results]

        if not true_labels:
            print("No valid samples processed.")
            return

        # Simple accuracy calc (comparing extracted choice vs GT choice string)
        # Note: gt_answer is like "(B)" and pred_choice is like "(B)"
        
        # Handle cases where extraction failed (None)
        pred_labels_safe = [p if p else "N/A" for p in pred_labels]
        
        accuracy = accuracy_score(true_labels, pred_labels_safe)
        print(f"Total Accuracy: {accuracy:.4f}")
        
        # Save raw results
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {os.path.join(args.output_dir, 'results.json')}")

        # Confusion Matrix
        unique_labels = sorted(list(set(true_labels + pred_labels_safe)))
        
        cm = confusion_matrix(true_labels, pred_labels_safe, labels=unique_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=unique_labels, yticklabels=unique_labels, cmap='viridis')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(f'Confusion Matrix (Acc: {accuracy:.4f})')
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        plt.close()
        
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
