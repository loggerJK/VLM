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
from transformers import AutoModelForCausalLM, set_seed
import torch.distributed as dist
from datetime import timedelta

from janus.models import MultiModalityCausalLM, VLChatProcessor

# No gradient
torch.set_grad_enabled(False)

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    valid_indices = range(len(choices)) if choices else range(26)
    valid_letters = [chr(ord('A') + i) for i in valid_indices]

    # 1. Check if the output text matches one of the given choices content.
    if choices:
        found_matches = []
        for i, choice in enumerate(choices):
            choice_text = str(choice).lower()
            pattern = r'(?<!\w)' + re.escape(choice_text) + r'(?!\w)'
            match = re.search(pattern, clean_text_lower)
            if match:
                found_matches.append((match.start(), chr(ord('A') + i)))

        if found_matches:
            found_matches.sort()
            return f"({found_matches[0][1]})"

    # 2. Fall back to extracting letter identifiers (A, B, C...).
    matches = re.findall(r'\(([A-Z])\)', text, re.IGNORECASE)
    for m in matches:
        if m.upper() in valid_letters:
            return f"({m.upper()})"

    stripped_text = text.strip()
    if stripped_text:
        first_char = stripped_text[0].upper()
        if first_char in valid_letters:
            if len(stripped_text) == 1 or not stripped_text[1].isalpha():
                return f"({first_char})"

    matches = re.findall(r'\b([A-Z])\b', text, re.IGNORECASE)
    if matches:
        for m in reversed(matches):
            if m.upper() in valid_letters:
                return f"({m.upper()})"

    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate CVBench Counting Multigpu (Janus)")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B", help="Janus model path")
    parser.add_argument("--lora_ckpt_path", type=str, default=None,
                        help="Path to LoRA checkpoint directory (optional)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
        local_rank = 0
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

    # Load Janus Model
    print(f"[Rank {rank}] Loading Janus model from {args.model_path}...")
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = model.to(torch.bfloat16).to(device).eval()

    # Load LoRA weights if provided
    if args.lora_ckpt_path is not None:
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(
            model.language_model, args.lora_ckpt_path
        )
        if rank == 0:
            print(f"Loaded LoRA weights from {args.lora_ckpt_path}")

    results = []

    print(f"[Rank {rank}] Starting inference...")
    print(f"Dataset size: {len(dataset)}")

    # Create local dataset shard for distributed inference
    local_indices = list(range(rank, len(dataset), world_size))
    local_dataset = dataset.select(local_indices)

    iterator = tqdm(local_dataset, total=len(local_dataset), desc=f"Rank {rank}")

    for item in iterator:
        set_all_seeds(args.seed)

        image = item['image']
        prompt_text_raw = item['prompt']
        gt_answer = item['answer']
        choices = item.get('choices', [])

        # Construct prompt with choices
        prompt_text = prompt_text_raw
        if choices:
            prompt_text += "\nSelect from the following choices."
            for c_idx, choice in enumerate(choices):
                letter = chr(ord('A') + c_idx)
                prompt_text += f"\n({letter}) {choice}"

        # Ensure image is PIL
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        # Prepare Janus input
        conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{prompt_text}"},
            {"role": "<|Assistant|>", "content": ""},
        ]

        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=[image], force_batchify=True
        ).to(device)
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        # Generate
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        text_new = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

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
        if rank == 0:
            results = [item for sublist in all_results for item in sublist]

    # Process metrics only on rank 0
    if rank == 0:
        true_labels = [r['gt_answer'] for r in results]
        pred_labels = [r['pred_choice'] for r in results]

        if not true_labels:
            print("No valid samples processed.")
            return

        pred_labels_safe = [p if p else "N/A" for p in pred_labels]

        accuracy = accuracy_score(true_labels, pred_labels_safe)
        print(f"Total Accuracy: {accuracy:.4f}")

        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {os.path.join(args.output_dir, 'results.json')}")

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
