import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import re
# 현재 PATH를 PYTHONPATH에 추가
from copy import deepcopy
from typing import (
    Any,
    AsyncIterable,
    Callable,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)
import requests
from io import BytesIO

from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from datasets import load_dataset

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae
from safetensors.torch import load_file

model_path = "/data/mm-llm-backbone_890/personal/sirius/audio_ablation/audio_bagel/models"  # Download from https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

# LLM config preparing
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# ViT config preparing
vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

# VAE loading
vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

# Bagel config preparing
config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config, 
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model      = SiglipVisionModel(vit_config)
    model          = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

# Tokenizer Preparing
tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

# Image Transform Preparing
vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

# max_mem_per_gpu = "48GiB"  # Modify it according to your GPU setting. On an A100, 80 GiB is sufficient to load on a single GPU.

device_map = infer_auto_device_map(
    model,
    # max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
    # no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    clean_result=True,
)
from pprint import pprint
pprint(device_map)

same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

# Thanks @onion-liu: https://github.com/ByteDance-Seed/Bagel/pull/8
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=False,
    dtype=torch.bfloat16,
    force_hooks=True,
    offload_folder="/tmp/offload"
)

model = model.eval()
print('Model loaded')


from inferencer import InterleaveInferencer

inferencer = InterleaveInferencer(
    model=model, 
    vae_model=vae_model, 
    tokenizer=tokenizer, 
    vae_transform=vae_transform, 
    vit_transform=vit_transform, 
    new_token_ids=new_token_ids
)

# SEED
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------------------------------------- #
#                       Understanding                      #
# -------------------------------------------------------- #
inference_hyper=dict(
    max_think_token_n=1000,
    do_sample=False,
    # text_temperature=0.3,
)

def extract_position_label(text):
    matches = re.findall(r"(top-left|top-right|bottom-left|bottom-right)", str(text).lower())
    return matches[0] if len(matches) == 1 else None


ds = load_dataset("heez/relative-position-new", split="validation", streaming=True)

predictions = []
references = []
correct = 0
parse_fail = 0
total = 0
for i, data in enumerate(ds):
    if i >= 100:  # Limit to first 100 samples
        break

    image = data['image']
    prompt = data['question']
    
    output_dict = inferencer(image=image, text=prompt, understanding_output=True, **inference_hyper)
    # print(output_dict['text'])
    answer = output_dict['text']
    
    print (f"Question:\n {prompt}")
    print(f"Predicted:\n {answer}")
    print (f"Answer:\n {data['answer']}")
    predictions.append(answer)
    references.append(data['answer'])

    pred_pos = extract_position_label(answer)
    gt_pos = extract_position_label(data['answer'])
    if pred_pos is None or gt_pos is None:
        parse_fail += 1
    if pred_pos is not None and gt_pos is not None and pred_pos == gt_pos:
        correct += 1
    total += 1
    print ("-"*50)
    
acc = correct / total if total else 0.0
parse_fail_rate = parse_fail / total if total else 0.0
print(f"Final Accuracy: {acc:.4f}")
print(f"Parse Fail Rate: {parse_fail_rate:.4f}")
# Save to file
with open("relpos_i2t_understanding_results.txt", "w") as f:
    f.write(f"Final Accuracy: {acc:.4f}\n")
    f.write(f"Parse Fail Rate: {parse_fail_rate:.4f}\n")
