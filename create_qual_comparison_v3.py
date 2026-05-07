import os
import json
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# --- Configuration ---
ROOT_DIR = "/mnt/data1/jiwon/Janus/results_geneval_count_extended"
OUTPUT_DIR = "/mnt/data1/jiwon/Janus/qual_comparison_results"
METADATA_FILE = "/mnt/data1/jiwon/geneval/prompts/evaluation_metadata_count.jsonl"

MODELS = [
    {"path": "janus_pro_7b_baseline", "label": "Baseline"},
    {"path": "train_full_ckpt600", "label": "Full (Fine-tune)\nStep 600"},
    {"path": "train_transformer_ckpt600", "label": "Trans (Fine-tune)\nStep 600"},
    {"path": "train_transformer_only_count_ckpt12000", "label": "Trans Only (Count)\nStep 12000"},
    {"path": "train_transformer_only_point_count_ckpt15000", "label": "Trans Only (Concat)\nStep 15000"},
    {"path": "train_transformer_only_points_ckpt15000", "label": "Trans Only (Point 0-20)\nStep 15000"}
]

SAMPLES_PER_PROMPT = 4
FONT_SIZE_HEADER = 40 
FONT_SIZE_LABEL = 30
HEADER_HEIGHT = 80
LABEL_HEIGHT = 100 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Metadata (Preserve File Order)
prompts = []
with open(METADATA_FILE, 'r') as f:
    for line in f:
        data = json.loads(line)
        prompts.append(data['prompt'])

print(f"Loaded {len(prompts)} prompts from master metadata.")

# --- Process ---
for idx, prompt_text in tqdm(enumerate(prompts), total=len(prompts), desc="Generating Grids"):
    prompt_id_str = f"{idx:05d}"
    
    model_images = []
    img_w, img_h = 384, 384 
    
    for model in MODELS:
        m_path = os.path.join(ROOT_DIR, model['path'], prompt_id_str, "samples")
        samples = []
        for s_idx in range(SAMPLES_PER_PROMPT):
            img_path = os.path.join(m_path, f"{s_idx:05d}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                img_w, img_h = img.size
                samples.append(img)
            else:
                # Grey placeholder if missing
                img = Image.new('RGB', (img_w, img_h), color=(200, 200, 200))
                samples.append(img)
        model_images.append(samples)
        
    grid_w = img_w * len(MODELS)
    grid_h = img_h * SAMPLES_PER_PROMPT + LABEL_HEIGHT + HEADER_HEIGHT
    
    result_img = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(result_img)
    
    try:
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE_HEADER)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE_LABEL)
    except:
        header_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        
    # Draw Header (Prompt)
    draw.text((20, 20), f"Prompt: {prompt_text} (ID: {prompt_id_str})", fill=(0, 0, 0), font=header_font)
    
    # Draw Model Labels
    for m_idx, model in enumerate(MODELS):
        x = m_idx * img_w
        y = HEADER_HEIGHT
        label = model['label']
        draw.multiline_text((x + 10, y + 10), label, fill=(0, 0, 0), font=label_font, align="center")
        
    # Paste Images
    for m_idx in range(len(MODELS)):
        for s_idx in range(SAMPLES_PER_PROMPT):
            img = model_images[m_idx][s_idx]
            x = m_idx * img_w
            y = HEADER_HEIGHT + LABEL_HEIGHT + s_idx * img_h
            result_img.paste(img, (x, y))
            
    # Save with simple index-based name to match prompt order
    filename = f"{prompt_id_str}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    result_img.save(save_path)

print(f"Successfully generated 90 grids in {OUTPUT_DIR}")
