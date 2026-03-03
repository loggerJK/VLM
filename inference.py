import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from diffusers.schedulers import DDIMScheduler
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel, CLIPImageProcessor
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
import pdb
import copy
import sys
import argparse
import json
from tqdm import tqdm
import shortuuid
from blip3o.constants import *
from blip3o.conversation import conv_templates, SeparatorStyle
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from blip3o.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import math
import requests
from blip3o.conversation import conv_templates, SeparatorStyle
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import base64
from io import BytesIO
from qwen_vl_utils import process_vision_info
from pipeline_llava_gen import EmuVisualGenerationPipeline

import re, random

# model_path = sys.argv[1]
# model_path = "/home/cvlab22/.cache/huggingface/hub/models--BLIP3o--BLIP3o-Model-8B/snapshots/c2edfc20814d4624c8d73ca3de351ebc3fa86508"
model_path = "/home/cvlab12/BLIP3o/BLIP3o-Model-8B"
diffusion_path = model_path + "/diffusion-decoder"



processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


device_1 = 0


disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, multi_model, context_len = load_pretrained_model(model_path, None, model_name)




scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_path, subfolder="scheduler")
unet = UNet2DConditionModel.from_pretrained(diffusion_path, subfolder="unet", torch_dtype=torch.bfloat16, variant="bf16")
vae = AutoencoderKL.from_pretrained(diffusion_path, subfolder="vae", torch_dtype=torch.bfloat16, variant="bf16")
feature_extractor = CLIPImageProcessor.from_pretrained(diffusion_path, subfolder="feature_extractor")

pipe = EmuVisualGenerationPipeline(
    tokenizer=tokenizer,
    multimodal_encoder=multi_model,
    scheduler=scheduler,
    unet=unet,
    vae=vae,
    feature_extractor=feature_extractor,
    safety_checker=None,
)


pipe.vae.to(f'cuda:{device_1}')
pipe.unet.to(f'cuda:{device_1}')



def create_image_grid(images, rows, cols):
    """Creates a grid of images and returns a single PIL Image."""

    assert len(images) == rows * cols

    width, height = images[0].size
    grid_width = width * cols
    grid_height = height * rows

    grid_image = Image.new('RGB', (grid_width, grid_height))

    for i, image in enumerate(images):
        x = (i % cols) * width
        y = (i // cols) * height
        grid_image.paste(image, (x, y))

    return grid_image


def add_template(prompt):
   conv = conv_templates['qwen'].copy()
   conv.append_message(conv.roles[0], prompt[0])
   conv.append_message(conv.roles[1], None)
   prompt = conv.get_prompt()
   return [prompt]



def set_global_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)






prompt = "A photo of cute cat"
set_global_seed(seed=42)
gen_images = []
for i in range(4):
    gen_img = pipe(add_template([f"Please generate image based on the following caption: {prompt}"]), guidance_scale=3.0)
    gen_images.append(gen_img.image)
print(f"finish {prompt}")



grid_image = create_image_grid(gen_images, 2, 2)
grid_image.save(f"{prompt[:100]}.png")









## if you train the image reconstruction, please use the following code to do the inference 

## your image path
# img = 1001615122.jpg

# set_global_seed(seed=42)
# gen_images = []
# inputs = add_template(["<image>\nPlease reconstruct the given image."])
# inputs.append(Image.open(f"{img}"))
# gen_img = pipe(inputs, guidance_scale=3.0)
# gen_images.append(gen_img.image)
# grid_image = create_image_grid(gen_images, 1, 1)
# gri
