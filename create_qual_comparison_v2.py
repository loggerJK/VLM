import os
import json
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# --- Configuration ---
ROOT_DIR = "/mnt/data1/jiwon/Janus/results_geneval_count_extended"
OUTPUT_DIR = "/mnt/data1/jiwon/Janus/qual_comparison_results_v2" # New dir
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

# 1. Load and Sort Metadata (CLASS-FIRST)
# We want to group by class, then by count.
items = []
with open(METADATA_FILE, 'r') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        # Extract class and count for sorting
        # {"tag": "counting", "include": [{"class": "person", "count": 2}], ...}
        cls = data['include'][0]['class']
        cnt = data['include'][0]['count']
        items.append({
            "orig_index": i,
            "class": cls,
            "count": cnt,
            "prompt": data['prompt']
        })

# Sort: Class name alphabetically, then Count numerically
# This will group all 'person' together (2 to 10), then 'apple' (wait, alphabetical order)
# Actually, let's keep the order of classes as they first appeared if possible, but sorting is more predictable.
items.sort(key=lambda x: (x['class'], x['count']))

print(f"Loaded {len(items)} prompts. Sorted in Class-First order.")

# --- Process ---
for item in tqdm(items, desc="Generating Sorted Grids"):
    idx = item['orig_index']
    prompt_id_str = f"{idx:05d}"
    prompt_text = item['prompt']
    cls_name = item['class']
    count_val = item['count']
    
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
        
    draw.text((20, 20), f"Prompt: {prompt_text} (Folder: {prompt_id_str})", fill=(0, 0, 0), font=header_font)
    
    for m_idx, model in enumerate(MODELS):
        x = m_idx * img_w
        y = HEADER_HEIGHT
        label = model['label']
        draw.multiline_text((x + 10, y + 10), label, fill=(0, 0, 0), font=label_font, align="center")
        
    for m_idx in range(len(MODELS)):
        for s_idx in range(SAMPLES_PER_PROMPT):
            img = model_images[m_idx][s_idx]
            x = m_idx * img_w
            y = HEADER_HEIGHT + LABEL_HEIGHT + s_idx * img_h
            result_img.paste(img, (x, y))
            
    # New Filename: class_count.png (e.g., person_02.png)
    filename = f"{cls_name}_{count_val:02d}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    result_img.save(save_path)

print(f"Done. Sorted results saved to {OUTPUT_DIR}")
