import os
import io
import re
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import time
import torch, gc
import glob
import transformers
import tokenizers
import random
import numpy as np
from blip3o.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_IDX
from torch.utils.data import Dataset
from blip3o.train.blip3o_trainer import blip3oTrainer
from blip3o import conversation as conversation_lib
from blip3o.model import *
from blip3o.mm_utils import tokenizer_image_token
from PIL import Image, ImageFile
from datasets import load_dataset, concatenate_datasets
from pathlib import Path
from datasets.utils.logging import set_verbosity_info
from transformers import logging as tf_logging
from transformers import TrainerCallback
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from pipeline_llava_gen import EmuVisualGenerationPipeline


ImageFile.LOAD_TRUNCATED_IMAGES = True
transform_und_images = T.Compose([T.Resize(448, interpolation=InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(448)])

set_verbosity_info()
tf_logging.set_verbosity_info()

local_rank = None




def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    gen_vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_gen_mlp_adapter: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    gen_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    n_query: Optional[int] = field(default=729)  # clip 576, siglip 729
    n_und_query: Optional[int] = field(default=729)  # clip 576, siglip 729
    gen_pooling: Optional[str] = field(default="all")  # options are: pool2d_3, pool2d_9, seq_3, seq_9, seq_27
    # Resume
    resume_ckpt: Optional[str] = field(default=None, metadata={"help": "Path to checkpoint to resume from"})
    auto_resume: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    journeyDB_folder: Optional[str] = field(default=None)
    shortcaption_image_folder: Optional[str] = field(default=None)
    data_type: Optional[str] = field(default="mix")
    image_aspect_ratio: str = "square"
    # New task/mode arguments
    task: str = field(default="counting", metadata={"help": "Task type: counting | ocr_synthetic"})
    mode: str = field(default="gen", metadata={"help": "Training mode: und | gen | both"})
    hf_dataset_path: str = field(default="heez/pixmo-point-count-gen-und", metadata={"help": "HuggingFace dataset path"})
    ocr_num_samples: int = field(default=200000, metadata={"help": "Number of OCR samples to use"})
    ocr_image_width: int = field(default=512, metadata={"help": "OCR image width"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj", metadata={"help": "Comma-separated LoRA target modules"})
    train_gen_components: str = field(
        default="",
        metadata={"help": "Comma-separated gen components to train with LoRA. Options: latent_queries,dit,down_projector"}
    )
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    # Validation
    validation_interval: int = field(default=500, metadata={"help": "Validation interval in global steps"})
    validation_samples: int = field(default=100, metadata={"help": "Number of validation samples"})
    log_freq: int = field(default=250, metadata={"help": "Generation validation logging frequency"})


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=[""]):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, vision_tower: str):
    """Collects the state dict and dump to disk."""

    # if getattr(trainer.args, "tune_vision_model", False):

    if trainer.deepspeed:
        torch.cuda.synchronize()


    # Only save Adapter
    keys_to_match = ["mm_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "mm_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))

    keys_to_match = ["gen_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "gen_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"gen_projector.bin"))

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):


    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation



def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    und_placeholder = "<|vision_start|>" + "<|image_pad|>" * data_args.n_und_query + "<|vision_end|>"
    gen_placeholder = ""
    # "[IMG]" + "<image>" * data_args.n_query + "[/IMG]"
    inst_type = None
    for source in sources:  # [instance]
        for sentence in source:
            if sentence["from"] == "human" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, und_placeholder).strip()
                inst_type = "und"
            elif sentence["from"] == "gpt" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, gen_placeholder).strip()
                inst_type = "gen"
    return sources, inst_type





def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id



        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )




def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id



        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )



def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        # assert DEFAULT_IMAGE_TOKEN in source[0]['value'] or DEFAULT_IMAGE_TOKEN in source[1]['value']
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


# ===================== Helper: image processing =====================

def img_process(images, processor, image_aspect_ratio):
    if image_aspect_ratio == "pad":
        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        images = [expand2square(img, tuple(int(x * 255) for x in processor.image_mean)) for img in images]
        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
    else:
        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
    return images


# ===================== New Dataset: CountingGenDataset =====================

class CountingGenDataset(Dataset):
    """Counting generation dataset from HuggingFace (heez/pixmo-point-count-gen-und)."""

    def __init__(self, tokenizer, data_args):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args

        rank0_print(f"[CountingGenDataset] Loading {data_args.hf_dataset_path} split=train ...")
        ds = load_dataset(data_args.hf_dataset_path, split="train", num_proc=64)
        # Filter: only rows with descriptions (for generation)
        self.dataset = ds.filter(lambda x: x.get('descriptions') is not None and x['descriptions'] != '', num_proc=64)
        self.dataset = self.dataset.shuffle(seed=42)
        rank0_print(f"[CountingGenDataset] Loaded {len(self.dataset)} generation samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        while True:
            try:
                item = self.dataset[i]
                caption = item['descriptions']
                image = item['image']
                if not isinstance(image, Image.Image):
                    image = Image.open(io.BytesIO(image)).convert("RGB")
                else:
                    image = image.convert("RGB")

                conversations = [
                    {"from": "human", "value": f"Please generate image based on: {caption}"},
                    {"from": "gpt", "value": "<image>"},
                ]

                sources, inst_type = preprocess_multimodal(
                    copy.deepcopy([conversations]), self.data_args
                )
                data_dict = preprocess(sources, self.tokenizer, has_image=True)
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

                data_dict["gen_image"] = img_process(
                    [image],
                    self.data_args.gen_image_processor,
                    self.data_args.image_aspect_ratio,
                )
                data_dict["ids"] = f"counting_gen_{i}"
                return data_dict

            except Exception as e:
                print(f"[CountingGenDataset] Error at index {i}: {e}")
                i = random.randint(0, len(self.dataset) - 1)
                continue


# ===================== New Dataset: OCRSyntheticGenDataset =====================

class OCRSyntheticGenDataset(Dataset):
    """OCR synthetic generation dataset from agentlans/high-quality-english-sentences."""

    def __init__(self, tokenizer, data_args):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args

        rank0_print(f"[OCRSyntheticGenDataset] Loading agentlans/high-quality-english-sentences ...")
        raw_ds = load_dataset("agentlans/high-quality-english-sentences", split="train", num_proc=64)
        n_samples = min(data_args.ocr_num_samples, len(raw_ds))
        self.dataset = raw_ds.select(range(n_samples))
        self.dataset = self.dataset.shuffle(seed=42)
        rank0_print(f"[OCRSyntheticGenDataset] Loaded {len(self.dataset)} OCR samples")

        self.ocr_width = data_args.ocr_image_width

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        while True:
            try:
                item = self.dataset[i]
                text = item['text']

                # Render OCR image
                from blip3o.ocr_render import generate_image as ocr_generate_image
                image = ocr_generate_image(
                    '# ' + text,
                    template="random",
                    width=self.ocr_width,
                    height=self.ocr_width,
                    quality=100,
                )
                image = image.convert("RGB")

                conversations = [
                    {"from": "human", "value": f"Please generate image based on: {text}"},
                    {"from": "gpt", "value": "<image>"},
                ]

                sources, inst_type = preprocess_multimodal(
                    copy.deepcopy([conversations]), self.data_args
                )
                data_dict = preprocess(sources, self.tokenizer, has_image=True)
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

                data_dict["gen_image"] = img_process(
                    [image],
                    self.data_args.gen_image_processor,
                    self.data_args.image_aspect_ratio,
                )
                data_dict["ids"] = f"ocr_gen_{i}"
                return data_dict

            except Exception as e:
                print(f"[OCRSyntheticGenDataset] Error at index {i}: {e}")
                i = random.randint(0, len(self.dataset) - 1)
                continue


# ===================== Original Dataset =====================

class LazySupervisedMixDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedMixDataset, self).__init__()

        self.data_args = data_args
        list_data_dict = []


        # load journeyDB_T2I data with json file
        # train_dataset = load_dataset("json", data_files='/fsx/sfr/data/jiuhai/hub/datasets--JourneyDB--JourneyDB/snapshots/e191aa61ca37e5e4418707ade4df5deb5c6d5d8f/data/train/train_caption_only.jsonl', split="train", num_proc=64)
        # if args.journeyDB_folder is not None:
        #     train_dataset = load_dataset("json", data_files=os.path.join(args.journeyDB_folder, "data/train/train_caption_only.jsonl"), split="train", num_proc=64)
        #     train_dataset = train_dataset.add_column('type', len(train_dataset) * ['journeyDB_T2I'])
        #     train_dataset = train_dataset.add_column('image', len(train_dataset) * [None])
        #     train_dataset = train_dataset.rename_column("caption", "txt")
        #     train_dataset = train_dataset.rename_column("img_path", "image_path")
        #     train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
        #         ["txt", "image", "type", "image_path"])])
        #     print(f"finish loading journeyDB {len(train_dataset)}")



        ###################################### text to image #######################################
        data_files = glob.glob(os.path.join(self.data_args.image_folder, "*.tar"))
        ## text to image
        train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=128)
        train_dataset = train_dataset.rename_column("jpg", "image")
        train_dataset = train_dataset.add_column('type', len(train_dataset) * ['T2I'])
        train_dataset = train_dataset.add_column('image_path', len(train_dataset) * [None])
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
            ["image", "txt", "type", "image_path"])])
        print(f"finish loading image {len(train_dataset)}")
        list_data_dict.append(train_dataset)


        if len(list_data_dict) > 1:
            list_data_dict = concatenate_datasets(list_data_dict)
        else:
            list_data_dict = list_data_dict[0]
        list_data_dict = list_data_dict.shuffle(seed=42)

        rank0_print(f"Totoal number of training instance: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        while True:
            sources = self.list_data_dict[i]

            if sources["type"] == "T2I" or sources["type"] == "journeyDB_T2I":
                sources["conversations"] = [
                    {"from": "human", "value": f"Please generate image based on the following caption: {sources['txt']}"},
                    {"from": "gpt", "value": "<image>"},
                ]


            elif sources["type"] == "I2I" or sources["type"] == "journeyDB_I2I":
                sources["conversations"] = [
                    {
                        "from": "human",
                        "value": f"<image>\nPlease reconstruct the given image.",
                    },
                    {"from": "gpt", "value": ""},
                ]

            else:
                raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")

            if "image" in sources:

                if sources["type"] == "T2I" or sources["type"] == "I2I":
                    image_files = self.list_data_dict[i]["image"]
                else:
                    image_files = self.list_data_dict[i]["image_path"]

                if not isinstance(image_files, list):
                    image_files = [image_files]

                images = []

                def read_bin_as_bytesio(bin_file_path):
                    with open(bin_file_path, "rb") as f:
                        return io.BytesIO(f.read())

                for img in image_files:
                    try:
                        if sources["type"] == "T2I" or sources["type"] == "I2I":
                            img = img.convert("RGB")
                        elif sources["type"] == "journeyDB_T2I" or sources["type"] == "journeyDB_I2I":
                            if sources["type"] == "journeyDB_T2I" or sources["type"] == "journeyDB_I2I":
                                image_path = os.path.join(args.journeyDB_folder, "data", "train", "imgs", img)
                            else:
                                raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")
                            img = Image.open(image_path).convert("RGB")
                        images.append(img)
                    except Exception as e:
                        print(f"Error opening image {img}: {e}")
                        images = None
                        break  # Skip to the next image if there's an error

                if not images is None:
                    try:
                        temp = img_process(
                            images,
                            self.data_args.gen_image_processor,
                            self.data_args.image_aspect_ratio,
                        )
                    except Exception as e:
                        print(f"Error wrong number of channels: {e}")
                        images = None


                # If no valid images were found, randomly pick another item
                if images is None:
                    print(sources)
                    print(f"warning false image!!!!!!")
                    i = random.randint(0, len(self.list_data_dict) - 1)
                    continue


                sources, inst_type = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
            else:
                sources = copy.deepcopy([sources["conversations"]])
            data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

            # image exist in the data
            if "image" in self.list_data_dict[i]:
                if inst_type == "gen":
                    data_dict["gen_image"] = img_process(
                        images,
                        self.data_args.gen_image_processor,
                        self.data_args.image_aspect_ratio,
                    )

                elif inst_type == "und":

                    resized_images = [transform_und_images(img) for img in images]

                    image_inputs = self.data_args.image_processor(resized_images, return_tensors="pt")

                    data_dict["und_image"] = image_inputs.pixel_values
                    data_dict["grid_thw"] = image_inputs.image_grid_thw
                    data_dict["gen_image"] = img_process(
                        resized_images,
                        self.data_args.gen_image_processor,
                        self.data_args.image_aspect_ratio,
                    )

            elif self.data_args.is_multimodal:
                crop_size = self.data_args.image_processor.crop_size
                data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

            data_dict["ids"] = self.list_data_dict[i]["id"] if "id" in self.list_data_dict[i] else "unk"
            return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "ids"))
        multi_input_ids = []
        multi_labels = []
        i_s_pos = []
        for input_id, label in zip(input_ids, labels):
            input_id = input_id[: self.tokenizer.model_max_length - 65] # 미리 자리 만들어놓기
            label = label[: self.tokenizer.model_max_length - 65]
            i_s_pos.append(input_id.shape[0]+1)
            img_id = torch.full((65,), IMAGE_TOKEN_IDX, dtype=input_id.dtype, device=input_id.device)
            img_id[0] = 151665
            input_id = torch.cat([input_id, img_id])
            img_label = torch.full((65,), IMAGE_TOKEN_IDX, dtype=label.dtype, device=label.device)
            img_label[0] = 151665
            label = torch.cat([label, img_label])
            multi_input_ids.append(input_id)
            multi_labels.append(label)

        input_ids = multi_input_ids
        labels = multi_labels

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            print(f"Warning input with length {input_ids.shape[1]} is longer than max length {self.tokenizer.model_max_length}")
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        batch_gen_images = []
        batch_und_images = []
        batch_grid_thw = []

        for instance in instances:
            if "gen_image" in instance:
                batch_gen_images.append(instance["gen_image"])


        if len(batch_gen_images) > 0:
            if all(x is not None and y.shape == batch_gen_images[0][0].shape for x in batch_gen_images for y in x):
                batch["gen_image"] = torch.cat([images for images in batch_gen_images], dim=0)
            else:
                batch["gen_image"] = batch_gen_images
        else:
            batch["gen_image"] = None


        for instance in instances:
            if "und_image" in instance:
                batch_und_images.append(instance["und_image"].unsqueeze(0))  ## 1*1024*1176
                batch_grid_thw.append(instance["grid_thw"])  ## 1*3


        # print(f"batch_und_images {batch_und_images}")
        if len(batch_und_images) > 0:
            batch["und_image"] = torch.cat([images for images in batch_und_images], dim=0)
            batch["grid_thw"] = torch.cat([images for images in batch_grid_thw], dim=0)
        else:
            batch["und_image"] = None
            batch["grid_thw"] = None

        batch["ids"] = ids

        batch["i_s_pos"] = i_s_pos

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:

    if data_args.data_type == "counting":
        train_dataset = CountingGenDataset(tokenizer=tokenizer, data_args=data_args)
    elif data_args.data_type == "ocr_synthetic":
        train_dataset = OCRSyntheticGenDataset(tokenizer=tokenizer, data_args=data_args)
    elif data_args.data_type == "mix":
        train_dataset = LazySupervisedMixDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    else:
        raise ValueError(f"Unknown data type: {data_args.data_type}. Choose from: counting, ocr_synthetic, mix")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def unlock_vit(training_args, model_args, vision_tower):
    for n, p in vision_tower.named_parameters():
        p.requires_grad = True


# ===================== Checkpoint Utilities =====================

def get_latest_checkpoint(output_dir):
    """Find the latest checkpoint directory (epoch{N}_step-{S}) in output_dir."""
    if not os.path.isdir(output_dir):
        return None
    ckpt_dirs = []
    for d in os.listdir(output_dir):
        full_path = os.path.join(output_dir, d)
        if os.path.isdir(full_path):
            m = re.search(r'step-(\d+)', d)
            if m:
                ckpt_dirs.append((int(m.group(1)), full_path))
    if not ckpt_dirs:
        return None
    ckpt_dirs.sort(key=lambda x: x[0])
    return ckpt_dirs[-1][1]


def unwrap_model(model):
    """Unwrap DDP/DeepSpeed/FSDP wrapped model."""
    if hasattr(model, 'module'):
        return unwrap_model(model.module)
    return model


# ===================== Validation Callback =====================

class ValidationCallback(TrainerCallback):
    """Performs validation (understanding + generation) at regular intervals and saves checkpoints."""

    def __init__(self, model_args, data_args, training_args, tokenizer, processor=None):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.processor = processor or AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.trainer = None

        self.validation_interval = training_args.validation_interval
        self.log_freq = training_args.log_freq
        self.validation_samples = training_args.validation_samples
        self.task = data_args.task
        self.mode = data_args.mode

        # Load validation datasets lazily
        self._val_und_ds = None
        self._val_gen_prompts = None
        self._ocr_metrics_loaded = None

    def _get_val_und_dataset(self):
        if self._val_und_ds is None:
            if self.task == "counting":
                try:
                    self._val_und_ds = load_dataset(
                        self.data_args.hf_dataset_path, split="val_und"
                    )
                    rank0_print(f"[Validation] Loaded val_und split: {len(self._val_und_ds)} samples")
                except Exception as e:
                    rank0_print(f"[Validation] Failed to load val_und split: {e}")
                    self._val_und_ds = []
            elif self.task == "ocr_synthetic":
                # Use a subset of training data for OCR validation
                try:
                    raw_ds = load_dataset("agentlans/high-quality-english-sentences", split="train")
                    # Use last N samples as validation
                    n_val = min(self.validation_samples, len(raw_ds))
                    start_idx = max(0, len(raw_ds) - n_val)
                    self._val_und_ds = raw_ds.select(range(start_idx, len(raw_ds)))
                    rank0_print(f"[Validation] Using {len(self._val_und_ds)} OCR validation samples")
                except Exception as e:
                    rank0_print(f"[Validation] Failed to load OCR validation data: {e}")
                    self._val_und_ds = []
        return self._val_und_ds

    def _get_val_gen_prompts(self):
        if self._val_gen_prompts is None:
            if self.task == "counting":
                try:
                    ds = load_dataset(self.data_args.hf_dataset_path, split="val_gen")
                    self._val_gen_prompts = [item['descriptions'] for item in ds.select(range(min(5, len(ds))))]
                except Exception:
                    self._val_gen_prompts = [
                        "A group of 3 red apples on a wooden table",
                        "5 birds flying in the blue sky",
                        "2 cats sitting on a windowsill",
                        "7 colorful balloons floating in the air",
                        "4 books stacked on a shelf",
                    ]
            else:
                self._val_gen_prompts = [
                    "Hello World",
                    "The quick brown fox jumps over the lazy dog",
                    "Machine learning is transforming the world",
                    "Python is a popular programming language",
                    "Deep learning enables many applications",
                ]
        return self._val_gen_prompts

    def _load_ocr_metrics(self):
        if self._ocr_metrics_loaded is None:
            try:
                import evaluate
                self._ocr_metrics_loaded = {
                    "wer": evaluate.load("wer"),
                    "cer": evaluate.load("cer"),
                    "meteor": evaluate.load("meteor"),
                }
            except Exception:
                self._ocr_metrics_loaded = {}
        return self._ocr_metrics_loaded

    def on_train_begin(self, args, state, control, **kwargs):
        # Restore global_step and epoch from custom checkpoint
        if getattr(self, 'resume_global_step', 0) > 0:
            state.global_step = self.resume_global_step
            state.epoch = self.resume_epoch
            rank0_print(f"[Resume] Restored state: global_step={state.global_step}, epoch={state.epoch}")

        if state.is_world_process_zero:
            model = kwargs.get('model', None)
            if model is not None:
                optimized_params = [n for n, p in model.named_parameters() if p.requires_grad]
                param_file = os.path.join(args.output_dir, "optimized_param_names.txt")
                os.makedirs(args.output_dir, exist_ok=True)
                with open(param_file, "w") as f:
                    for name in optimized_params:
                        f.write(f"{name}\n")
                rank0_print(f"[Validation] Saved {len(optimized_params)} trainable param names to {param_file}")
                
            # Run initial validation before training starts
            self._validate_generation(kwargs.get('model', None), state)
            self._validate_understanding(kwargs.get('model', None), state)
            

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return

        # Save checkpoint at save_steps intervals
        if state.global_step > 0 and state.global_step % args.save_steps == 0:
            self._save_checkpoint(args, state, model)

        # Run understanding validation
        if state.global_step > 0 and state.global_step % self.validation_interval == 0:
            self._validate_understanding(model, state)

        # Run generation validation (log images)
        if state.global_step > 0 and state.global_step % self.log_freq == 0:
            self._validate_generation(model, state)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero:
            epoch = int(state.epoch)
            save_dir = os.path.join(args.output_dir, f"epoch{epoch}")
            self._save_checkpoint_to(save_dir, model, state)

    def _save_checkpoint(self, args, state, model):
        save_dir = os.path.join(
            args.output_dir,
            f"epoch{int(state.epoch)}_step-{state.global_step}"
        )
        self._save_checkpoint_to(save_dir, model, state)

    def _save_checkpoint_to(self, save_dir, model, state):
        os.makedirs(save_dir, exist_ok=True)
        raw_model = unwrap_model(model)

        # 1. Save LoRA adapter if using LoRA
        if self.training_args.lora_enable:
            try:
                # For peft-wrapped models, save the adapter
                if hasattr(raw_model, 'save_pretrained'):
                    raw_model.save_pretrained(save_dir)
                    rank0_print(f"[Checkpoint] Saved LoRA adapter to {save_dir}")
            except Exception as e:
                rank0_print(f"[Checkpoint] Failed to save LoRA adapter: {e}")

        # 2. Save non-LoRA trainable components (DiT, down_projector, latent_queries)
        try:
            if self.training_args.lora_enable and hasattr(raw_model, 'base_model'):
                base_model = raw_model.base_model.model
            else:
                base_model = raw_model

            # Access the inner model (Qwen model)
            inner = base_model.model if hasattr(base_model, 'model') else base_model.get_model()

            gen_state = {}
            # if hasattr(inner, 'dit'):
            #     gen_state['dit'] = {k: v.cpu() for k, v in inner.dit.state_dict().items()}
            # if hasattr(inner, 'down_projector'):
            #     gen_state['down_projector'] = {k: v.cpu() for k, v in inner.down_projector.state_dict().items()}
            if hasattr(inner, 'latent_queries'):
                gen_state['latent_queries'] = inner.latent_queries.data.cpu()

            if gen_state:
                torch.save(gen_state, os.path.join(save_dir, "gen_components.pt"))
                rank0_print(f"[Checkpoint] Saved gen_components to {save_dir}")
        except Exception as e:
            rank0_print(f"[Checkpoint] Failed to save gen_components: {e}")

        # # 3. Save tokenizer
        # try:
        #     self.tokenizer.save_pretrained(save_dir)
        # except Exception as e:
        #     rank0_print(f"[Checkpoint] Failed to save tokenizer: {e}")

        rank0_print(f"[Checkpoint] Saved checkpoint at step {state.global_step} to {save_dir}")

    @torch.no_grad()
    def _validate_understanding(self, model, state):
        """Run understanding validation (counting accuracy or OCR metrics)."""
        val_ds = self._get_val_und_dataset()
        if not val_ds or len(val_ds) == 0:
            return

        raw_model = unwrap_model(model)
        raw_model.eval()

        n_samples = min(self.validation_samples, len(val_ds))

        if self.task == "counting":
            self._validate_counting_understanding(raw_model, val_ds, n_samples, state)
        elif self.task == "ocr_synthetic":
            self._validate_ocr_understanding(raw_model, val_ds, n_samples, state)

        raw_model.train()

    def _validate_counting_understanding(self, model, val_ds, n_samples, state):
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
        
        """Validate counting understanding using Qwen2.5-VL generate."""
        correct = 0
        total = 0
        abs_diffs = []

        processor = self.processor

        for idx in range(n_samples):
            item = val_ds[idx]
            image = item['image']
            question = item['question']
            answer = str(item['answer']).strip()

            if not isinstance(image, Image.Image):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            else:
                image = image.convert("RGB")

            # Use Qwen2.5-VL processor for understanding
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ]}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            output_ids = model.generate(**inputs, max_new_tokens=20)
            # Decode only generated tokens
            generated = processor.batch_decode(
                output_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )[0].strip()


            pred_num = extract_number_fixed(generated)
            gt_num = extract_number_fixed(answer)
            
        
            print("="*50)
            print(f"[GT]")
            print(f"{answer} (extracted: {gt_num})")
            print(f"[Prediction]")
            print(f"{generated} (extracted: {pred_num})")

            if pred_num is not None and gt_num is not None:
                if pred_num == gt_num:
                    correct += 1
                abs_diffs.append(abs(pred_num - gt_num))
                total += 1


        if total > 0:
            accuracy = correct / total
            mad = sum(abs_diffs) / len(abs_diffs)
            rank0_print(f"[Validation] Step {state.global_step} | Counting Accuracy: {accuracy:.4f} | MAD: {mad:.2f} | ({total} samples)")

            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "val/accuracy": accuracy,
                        "val/mad": mad,
                        "val/num_samples": total,
                        "global_step": state.global_step,
                    })
            except ImportError:
                pass

    def _validate_ocr_understanding(self, model, val_ds, n_samples, state):
        """Validate OCR understanding using Qwen2.5-VL generate + NLTK metrics."""
        from blip3o.ocr_metrics import calculate_metrics
        from blip3o.ocr_render import generate_image as ocr_generate_image

        predictions = []
        references = []

        processor = self.processor

        for idx in range(n_samples):
            try:
                item = val_ds[idx]
                text = item['text']

                # Render OCR image
                image = ocr_generate_image(
                    '# ' + text, template="clean_light",
                    width=self.data_args.ocr_image_width,
                    height=self.data_args.ocr_image_width,
                    quality=100
                ).convert("RGB")

                question = "Extract all text from the image."
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ]}
                ]
                text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text_input],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                output_ids = model.generate(**inputs, max_new_tokens=150)
                generated = processor.batch_decode(
                    output_ids[:, inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )[0].strip()

                predictions.append(generated)
                references.append(text)

            except Exception as e:
                continue

        if predictions:
            loaded_metrics = self._load_ocr_metrics()
            metrics = calculate_metrics(predictions, references, loaded_metrics=loaded_metrics)
            rank0_print(
                f"[Validation] Step {state.global_step} | OCR Metrics: "
                f"WER={metrics['wer']:.4f} CER={metrics['cer']:.4f} "
                f"METEOR={metrics['meteor']:.4f} BLEU={metrics['bleu']:.4f} "
                f"F1={metrics['f1']:.4f} ({len(predictions)} samples)"
            )

            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "val/ocr_wer": metrics["wer"],
                        "val/ocr_cer": metrics["cer"],
                        "val/ocr_meteor": metrics["meteor"],
                        "val/ocr_bleu": metrics["bleu"],
                        "val/ocr_edit_distance": metrics["edit_distance"],
                        "val/ocr_precision": metrics["precision"],
                        "val/ocr_recall": metrics["recall"],
                        "val/ocr_f1": metrics["f1"],
                        "val/num_samples": len(predictions),
                        "global_step": state.global_step,
                    })
            except ImportError:
                pass

    @torch.no_grad()
    def _validate_generation(self, model, state):
        """Generate images from validation prompts and log to WandB."""
        prompts = self._get_val_gen_prompts()
        if not prompts:
            return

        try:
            import wandb
            if wandb.run is None:
                return
        except ImportError:
            return

        rank0_print(f"[Validation] Step {state.global_step} | Generating {len(prompts)} validation images...")

        from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
        from transformers import CLIPImageProcessor

        raw_model = unwrap_model(model)
        raw_model.eval()

        # Load diffusion pipeline components individually to avoid model_index.json parsing
        # (which triggers ModuleNotFoundError for transformers_modules)
        model_path = self.model_args.model_name_or_path
        diffusion_path = os.path.join(model_path, 'diffusion-decoder')
        dtype = torch.bfloat16 if self.training_args.bf16 else None

        scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_path, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(diffusion_path, subfolder="unet", torch_dtype=dtype, variant="bf16")
        vae = AutoencoderKL.from_pretrained(diffusion_path, subfolder="vae", torch_dtype=dtype, variant="bf16")
        feature_extractor = CLIPImageProcessor.from_pretrained(diffusion_path, subfolder="feature_extractor")

        pipe = EmuVisualGenerationPipeline(
            tokenizer=self.tokenizer,
            multimodal_encoder=raw_model,
            scheduler=scheduler,
            unet=unet,
            vae=vae,
            feature_extractor=feature_extractor,
            safety_checker=None,
        )
        pipe = pipe.to(model.device)

        images_for_wandb = []
        for prompt in prompts:
            # Encode text prompt through the model
            formatted_prompt = f"Please generate image based on: {prompt}"
            result = pipe(formatted_prompt, guidance_scale=3.0, num_inference_steps=50)
            gen_image = result.image
            images_for_wandb.append(wandb.Image(gen_image, caption=prompt))

        if images_for_wandb:
            wandb.log({
                "val/generated_images": images_for_wandb,
                "global_step": state.global_step,
            })
            rank0_print(f"[Validation] Logged {len(images_for_wandb)} generated images to WandB")

        del pipe, unet, vae, scheduler, feature_extractor
        torch.cuda.empty_cache()
        raw_model.train()



# ===================== Main Training Function =====================

def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args, data_args, training_args)
    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    # Route data_type from task argument for new datasets
    if data_args.task in ["counting", "ocr_synthetic"]:
        data_args.data_type = data_args.task

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    ## if there exists vision tower for image understanind, we will load LLaMA LLM, otherwise will load Qwen-VL
    if model_args.vision_tower is not None:
        model = blip3oLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )
    else:
        model = blip3oQwenForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        for (n, p) in model.get_model().named_parameters():
            p.requires_grad = False
        for (n, p) in model.visual.named_parameters():
            p.requires_grad = False
        for (n, p) in model.lm_head.named_parameters():
            p.requires_grad = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    try:
        tokenizer = AutoProcessor.from_pretrained(model_args.model_name_or_path).tokenizer
    except Exception as e:
        tokenizer = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    tokenizer.model_max_length = training_args.model_max_length

    # tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(
                pad_token="<pad>",
                additional_special_tokens=["[IMG]", "[/IMG]", "<image>"],
            ),
            tokenizer=tokenizer,
            model=model,
        )
    elif not "<image>" in tokenizer.get_added_vocab():
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
            tokenizer=tokenizer,
            model=model,
        )
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]
    rank0_print(f"Using conversation format: {conversation_lib.default_conversation.version}")



    # if model_args.vision_tower is not None:
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

    ## generation vision tower
    gen_vision_tower = model.get_gen_vision_tower()
    gen_vision_tower.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device=training_args.device,
    )
    gen_vision_tower.requires_grad_(False)

    data_args.gen_image_processor = gen_vision_tower.image_processor
    data_args.image_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct").image_processor

    data_args.is_multimodal = True
    data_args.n_query = model_args.n_query
    data_args.n_und_query = model_args.n_und_query

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id

    # ===================== Resume: determine checkpoint path =====================
    resume_global_step = 0
    resume_epoch = 0
    resume_path = model_args.resume_ckpt
    if model_args.auto_resume and resume_path is None:
        resume_path = get_latest_checkpoint(training_args.output_dir)
        if resume_path:
            rank0_print(f"[Auto-Resume] Found latest checkpoint: {resume_path}")

    # ===================== LoRA Setup =====================
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, PeftModel, TaskType

        if resume_path and os.path.exists(os.path.join(resume_path, "adapter_config.json")):
            # Resume: load existing LoRA adapter
            rank0_print(f"[Resume] Loading LoRA adapter from {resume_path}")
            model = PeftModel.from_pretrained(
                model, resume_path,
                is_trainable=True,
            )
            # Re-enable gradients for LoRA params
            for name, param in model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
        else:
            # Fresh start: create new LoRA adapter
            target_modules = [m.strip() for m in training_args.lora_target_modules.split(",")]
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=training_args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

    # ===================== Resume: load non-LoRA trainable components =====================
    if resume_path:
        gen_ckpt_path = os.path.join(resume_path, "gen_components.pt")
        if os.path.exists(gen_ckpt_path):
            rank0_print(f"[Resume] Loading gen_components from {gen_ckpt_path}")
            gen_state = torch.load(gen_ckpt_path, map_location="cpu")

            if training_args.lora_enable and hasattr(model, 'base_model'):
                base_model = model.base_model.model
            else:
                base_model = model

            inner = base_model.model if hasattr(base_model, 'model') else base_model.get_model()

            if 'dit' in gen_state and hasattr(inner, 'dit'):
                inner.dit.load_state_dict(gen_state['dit'])
                rank0_print("[Resume] Restored DiT weights")
            if 'down_projector' in gen_state and hasattr(inner, 'down_projector'):
                inner.down_projector.load_state_dict(gen_state['down_projector'])
                rank0_print("[Resume] Restored down_projector weights")
            if 'latent_queries' in gen_state and hasattr(inner, 'latent_queries'):
                inner.latent_queries.data.copy_(gen_state['latent_queries'])
                rank0_print("[Resume] Restored latent_queries")

        # Parse resume step/epoch for logging
        ckpt_name = os.path.basename(resume_path)
        m_step = re.search(r'step-(\d+)', ckpt_name)
        m_epoch = re.search(r'epoch(\d+)', ckpt_name)
        resume_global_step = int(m_step.group(1)) if m_step else 0
        resume_epoch = int(m_epoch.group(1)) if m_epoch else 0
        rank0_print(f"[Resume] Parsed: epoch={resume_epoch}, global_step={resume_global_step}")

    # ===================== Re-enable gen components for LoRA training =====================
    if training_args.lora_enable and training_args.train_gen_components:
        components = [c.strip() for c in training_args.train_gen_components.split(",")]
        if hasattr(model, 'base_model'):
            inner = model.base_model.model.model
        else:
            inner = model.get_model()
        for comp_name in components:
            if comp_name == "latent_queries" and hasattr(inner, 'latent_queries') and inner.latent_queries is not None:
                inner.latent_queries.requires_grad = True
                rank0_print(f"[LoRA+Gen] Re-enabled {comp_name} ({inner.latent_queries.numel():,} params)")
            elif comp_name == "dit" and hasattr(inner, 'dit'):
                for p in inner.dit.parameters():
                    p.requires_grad = True
                n = sum(p.numel() for p in inner.dit.parameters())
                rank0_print(f"[LoRA+Gen] Re-enabled {comp_name} ({n:,} params)")
            elif comp_name == "down_projector" and hasattr(inner, 'down_projector') and inner.down_projector is not None:
                for p in inner.down_projector.parameters():
                    p.requires_grad = True
                n = sum(p.numel() for p in inner.down_projector.parameters())
                rank0_print(f"[LoRA+Gen] Re-enabled {comp_name} ({n:,} params)")
            else:
                rank0_print(f"[LoRA+Gen] Warning: unknown or missing component '{comp_name}'")

    # Calculate total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Create validation callback
    validation_callback = ValidationCallback(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )
    validation_callback.resume_global_step = resume_global_step
    validation_callback.resume_epoch = resume_epoch

    trainer = blip3oTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[validation_callback],
        **data_module,
    )

    # Set trainer reference for callback
    validation_callback.trainer = trainer

    from tabulate import tabulate

    if trainer.is_world_process_zero():
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            stat.append([i, n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))

    # Use HF Trainer's built-in checkpoint resume if available, otherwise fresh start
    hf_resume = None
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        hf_resume = True

    if hf_resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        vision_tower=model_args.vision_tower,
    )


if __name__ == "__main__":
    train()
