import collections
import os
import random
import torch
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import glob
from typing import List, Dict, Any, Optional, Iterator
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from torchvision import transforms
import json
from PIL import Image

class RefinedWebDataset(IterableDataset):
    def __init__(self,
                 data_path,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle=True,
                 repeat=True,
                 buffer_size=1000,
                 max_length=8000,
                 num_workers=1):
        super().__init__()
        self.files = sorted(glob.glob(data_path))  
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.num_workers = num_workers

        self.files = self.files[self.rank::self.world_size]

    def read_parquet_file(self, file_path):
        table = pq.read_table(file_path, columns=["content"])
        df = table.to_pandas()
        for _, row in df.iterrows():
            yield {"content": row["content"]}

    def __iter__(self):
        while True:  
            file_list = self.files
            if self.shuffle:
                random.shuffle(file_list)  

            for file in file_list:
                data_generator = self.read_parquet_file(file)
                buffer = []

                for data in data_generator:
                    text = data["content"].replace("\n", "")
                    if len(text) > self.max_length:
                        start_index = random.randint(0, len(text) - self.max_length - 1)
                        selected_text = text[start_index:start_index + self.max_length]
                    else:
                        selected_text = text

                    buffer.append({"input_ids": selected_text})

                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            random.shuffle(buffer)
                        for item in buffer:
                            yield item
                        buffer = []

                if buffer:
                    if self.shuffle:
                        random.shuffle(buffer)
                    for item in buffer:
                        yield item

            if not self.repeat:
                break  

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)

        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity'):
                batched[k] = torch.stack(v, dim=0)

        return batched

class ChatDataset(IterableDataset):
    def __init__(self,
                 data_path,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle=True,
                 repeat=True,
                 buffer_size=1000,
                 max_length=8000,
                 num_workers=1,
                 tokenizer=None):
        super().__init__()
        self.files = sorted(glob.glob(data_path))  
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer = tokenizer

        self.files = self.files[self.rank::self.world_size]

    def read_parquet_file(self, file_path):
        table = pq.read_table(file_path, columns=["content"])
        df = table.to_pandas()
        for _, row in df.iterrows():
            yield {"content": row["content"]}

    def __iter__(self):
        while True:  
            file_list = self.files
            if self.shuffle:
                random.shuffle(file_list)  

            for file in file_list:
                data_generator = self.read_parquet_file(file)
                buffer = []

                for data in data_generator:
                    text = data["content"]
                    if  self.tokenizer is None:
                        if len(text) > self.max_length:
                            start_index = random.randint(0, len(text) - self.max_length - 1)
                            selected_text = text[start_index:start_index + self.max_length]
                        else:
                            selected_text = text
                    else:
                        if len(self.tokenizer(text)['input_ids']) < self.max_length:
                            selected_text = text
                        else:
                            continue

                    buffer.append({"input_ids": selected_text})

                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            random.shuffle(buffer)
                        for item in buffer:
                            yield item
                        buffer = []

                if buffer:
                    if self.shuffle:
                        random.shuffle(buffer)
                    for item in buffer:
                        yield item

            if not self.repeat:
                break  

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)

        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity'):
                batched[k] = torch.stack(v, dim=0)

        return batched

class R2iDataset(IterableDataset):
    def __init__(self,
                 data_path,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle=True,
                 repeat=True,
                 buffer_size=1000,
                 max_length=8000,
                 num_workers=1,
                 resolution=256,
                 tokenizer=None):
        super().__init__()
        self.data_path = data_path  
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.resolution = resolution

    def __iter__(self):
        while True:  
            subdirs = sorted([d for d in glob.glob(os.path.join(self.data_path, "*")) if os.path.isdir(d)])
            
            if self.shuffle:
                random.shuffle(subdirs)  
            
            subdirs = subdirs[self.rank::self.world_size]

            subdirs = ['/data_storage/lbw/datasets/laion-aesthetics-12m-images-2/00000']
            
            for subdir in subdirs:
                all_files = glob.glob(os.path.join(subdir, "*.*"))
                base_names = set()
                
                for file_path in all_files:
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    base_names.add(base_name)
                
                base_names = list(base_names)
                if self.shuffle:
                    random.shuffle(base_names)
                
                buffer = []
                
                for base_name in base_names:
                    jpg_path = os.path.join(subdir, f"{base_name}.jpg")
                    caption_path = os.path.join(subdir, f"{base_name}.caption")
                    shortcaption_path = os.path.join(subdir, f"{base_name}.shortcaption")
                    
                    if not os.path.exists(jpg_path):
                        continue
                    
                    try:
                        image = Image.open(jpg_path).convert("RGB")
                        
                        caption = ""
                        if os.path.exists(caption_path):
                            with open(caption_path, "r", encoding="utf-8") as f:
                                caption = f.read().strip()
                        
                        short_caption = ""
                        if os.path.exists(shortcaption_path):
                            with open(shortcaption_path, "r", encoding="utf-8") as f:
                                short_caption = f.read().strip()
                        
                        transformed_image = image_transform_clip({"images": image}, resolution=self.resolution)["images"]
                        
                        if self.tokenizer is not None:
                            if len(self.tokenizer(caption)['input_ids']) > self.max_length - 2:
                                continue
                        
                        prompt = (
                            '<|start_header_id|>user<|end_header_id|>\n'
                            "You should first think out a more detailed version of the description and then provide the user with the image. The detailed description is enclosed within <think> </think> tags, i.e. <think> detailed description here </think> image here\n"
                            f"{short_caption}"
                            '<eot_id><|start_header_id|>assistant<|end_header_id|>\n'
                            f"<think>{caption}</think>"
                        )

                        sample = {
                            "images": transformed_image,
                            "input_ids": prompt,
                        }
                        
                        buffer.append(sample)
                        
                        if len(buffer) >= self.buffer_size:
                            if self.shuffle:
                                random.shuffle(buffer)
                            for item in buffer:
                                yield item
                            buffer = []
                    
                    except Exception as e:
                        print(f"Error processing {jpg_path}: {e}")
                        continue
                
                if buffer:
                    if self.shuffle:
                        random.shuffle(buffer)
                    for item in buffer:
                        yield item
            
            if not self.repeat:
                break  

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)

        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity'):
                batched[k] = torch.stack(v, dim=0)

        return batched

class VQADataset(IterableDataset):
    def __init__(self,
                 json_path: str,
                 image_root: str,
                 tokenizer = None,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle: bool = True,
                 repeat: bool = True,
                 buffer_size: int = 100,
                 resolution: int = 256,
                 max_length: int = 8000,
                 num_workers: int = 1,
                 image_transform_method: str = "squash"):
        super().__init__()
        self.json_path = json_path
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.resolution = resolution
        self.max_length = max_length 
        self.num_workers = num_workers
        self.image_transform_method = image_transform_method
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.json_path}")
            self.list_data_dict = []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.json_path}")
            self.list_data_dict = []
        else:
            self.list_data_dict = [item for item in raw_data if 'image' in item and 'conversations' in item]
        self.list_data_dict = self.list_data_dict[self.rank::self.world_size]
        self._sot_token = '<|startoftext|>'
        self._trailing_asst_header = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        self._assistant_header = '<|start_header_id|>assistant<|end_header_id|>'

    def __iter__(self):
        while True:
            current_data_list = list(self.list_data_dict) 
            if self.shuffle:
                random.shuffle(current_data_list)
            buffer = []
            for item in current_data_list:
                image_relative_path = item.get('image')
                conversations = item.get('conversations', [])
                if not image_relative_path or not conversations or len(conversations) < 2:
                    continue
                num_total_messages = len(conversations)
                if num_total_messages % 2 != 0:
                     conversations = conversations[:-1]
                     num_total_messages -= 1
                     if num_total_messages < 2: continue 
                num_turns = num_total_messages // 2
                if num_turns == 0:
                    continue
                selected_num_turns = random.randint(1, num_turns)
                selected_conversations = conversations[:selected_num_turns * 2]
                image_path = os.path.join(self.image_root, image_relative_path)
                try:
                    image = Image.open(image_path).convert("RGB")
                    if self.image_transform_method == "squash":
                        transformed_image = image_transform_squash({"images": image}, resolution=self.resolution)["images"]
                    elif self.image_transform_method == "pad":
                        transformed_image = image_transform_pad({"images": image}, resolution=self.resolution)["images"]
                    else:
                        transformed_image = image_transform_clip({"images": image}, resolution=self.resolution)["images"]
                    first_human_message = selected_conversations[0]['value']
                    processed_message = first_human_message.replace('<image>\n', '').replace('\n<image>', '')
                    current_selection_messages = list(selected_conversations)
                    current_selection_messages[0] = dict(current_selection_messages[0]) 
                    current_selection_messages[0]['value'] = processed_message
                    messages = []
                    for turn in current_selection_messages:
                        role = "user" if turn["from"] == "human" else "assistant"
                        messages.append({"role": role, "content": turn["value"]})
                    formatted_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    if formatted_text.startswith(self._sot_token):
                         formatted_text = formatted_text[len(self._sot_token):]
                    # Trailing 빈 assistant generation prompt 제거 (개행 개수에 robust)
                    idx = formatted_text.rfind(self._trailing_asst_header)
                    if idx != -1 and formatted_text[idx + len(self._trailing_asst_header):].strip() == '':
                        formatted_text = formatted_text[:idx]
                    # Sanity: answer 앞의 진짜 assistant header 가 살아있고 그 뒤 정답 텍스트가 있어야 함
                    if self._assistant_header not in formatted_text:
                        continue
                    if not formatted_text.rsplit(self._assistant_header, 1)[-1].strip():
                        continue
                    token_ids = self.tokenizer(formatted_text)['input_ids']
                    if len(token_ids) > self.max_length:
                        continue 
                    sample = {
                        "images": transformed_image,
                        "input_ids": formatted_text,
                    }
                    buffer.append(sample)
                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            random.shuffle(buffer)
                        for buf_item in buffer:
                            yield buf_item
                        buffer = []
                except FileNotFoundError:
                    print(f"Warning: Image file not found at {image_path}, skipping item.")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing item with image {image_path}: {e}, skipping.")
                    continue
            if buffer:
                if self.shuffle:
                    random.shuffle(buffer)
                for buf_item in buffer:
                    yield buf_item
            if not self.repeat:
                break
    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity'):
                batched[k] = torch.stack(v, dim=0)
        return batched

class CountingDataset(IterableDataset):
    """HuggingFace PixMo counting dataset for object counting understanding task."""

    def __init__(self,
                 rank: int = 0,
                 world_size: int = 1,
                 tokenizer=None,
                 resolution: int = 512,
                 max_length: int = 8000,
                 num_workers: int = 1,
                 count_lower_limit: int = 0,
                 count_upper_limit: int = 20,
                 shuffle: bool = True,
                 repeat: bool = True,
                 buffer_size: int = 100,
                 answer_format: str = "sentence"
                 ):
        super().__init__()
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("heez/pixmo-point-count-gen-und", split="train")
        print(f"Original dataset size: {len(ds)}, filtering counts between {count_lower_limit} and {count_upper_limit}...")
        # ds = ds.filter(lambda x: count_lower_limit <= x['count'] <= count_upper_limit)
        ds = ds.filter(
            lambda count: count is not None and count_lower_limit <= count <= count_upper_limit,
            input_columns=["count"],
            num_proc=32,
        )
        # Lumina parity (train_unified.py:1203): understanding subset is rows with
        # descriptions == None or "" (the gen split has non-empty descriptions).
        if 'descriptions' in ds.column_names:
            print(f"Filtering descriptions...")
            ds = ds.filter(
                lambda descriptions: descriptions is None or descriptions == '',
                input_columns=['descriptions'],
                num_proc=32,
            )
            print(f"Dataset size after filtering descriptions: {len(ds)}")
        all_indices = list(range(len(ds)))
        self.indices = all_indices[rank::world_size]
        self.ds = ds
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.max_length = max_length
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.answer_format = answer_format
        print(f"[INFO] Using answer format: {self.answer_format.upper()}")
        self._sot_token = '<|startoftext|>'
        self._trailing_asst_header = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        self._assistant_header = '<|start_header_id|>assistant<|end_header_id|>'

    def __iter__(self):
        # Worker-level sharding on top of rank-level sharding
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_indices = self.indices[worker_info.id::worker_info.num_workers]
        else:
            worker_indices = self.indices

        while True:
            index_list = list(worker_indices)
            if self.shuffle:
                random.shuffle(index_list)
            buffer = []
            for idx in index_list:
                try:
                    item = self.ds[idx]
                    image = item['image']
                    if not isinstance(image, Image.Image):
                        image = Image.fromarray(image).convert('RGB')
                    else:
                        image = image.convert('RGB')

                    question = item.get('question_count', item.get('question', ''))
                    # Lumina parity: use answer_count verbatim. Wrap as **N** ONLY when
                    # falling back to the raw integer `count` field.
                    raw_answer = item.get('answer_count', None)
                    if raw_answer is None or str(raw_answer).strip() == '':
                        raw_answer = item.get('label', None)
                    if raw_answer is None or str(raw_answer).strip() == '':
                        count_val = item.get('count', None)
                        if count_val is None:
                            continue
                        raw_answer = f'**{int(count_val)}**'
                    else:
                        raw_answer = str(raw_answer).strip()

                    transformed_image = image_transform_squash(
                        {'images': image}, resolution=self.resolution
                    )['images']

                    if self.answer_format == "number":
                        count_val = item.get('count', None)
                        if count_val is None:
                            # Warning: count value missing, skipping this sample
                            print(f"Warning: count value missing for index {idx}, skipping sample.")
                            continue
                        raw_answer = str(count_val)
                        # question을 "Response Example" 전까지 자른다
                        target_substring = "Response Example"
                        if target_substring in question:
                            question = question.split(target_substring)[0].strip()
                        else:
                            # Warning: "Response Example:" not found in question, using full question text
                            print(f"Warning: 'Response Example:' not found in question for index {idx}")
                            continue

                    messages = [
                        {'role': 'user', 'content': question},
                        {'role': 'assistant', 'content': raw_answer},
                    ]
                    formatted_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if formatted_text.startswith(self._sot_token):
                        formatted_text = formatted_text[len(self._sot_token):]
                    # Trailing 빈 assistant generation prompt 제거 (개행 개수에 robust)
                    hidx = formatted_text.rfind(self._trailing_asst_header)
                    if hidx != -1 and formatted_text[hidx + len(self._trailing_asst_header):].strip() == '':
                        formatted_text = formatted_text[:hidx]
                    # Sanity: answer 앞 assistant header + 그 뒤 정답 텍스트가 있어야 함
                    if self._assistant_header not in formatted_text:
                        continue
                    if not formatted_text.rsplit(self._assistant_header, 1)[-1].strip():
                        continue

                    if len(self.tokenizer(formatted_text)['input_ids']) > self.max_length:
                        continue

                    sample = {
                        'images': transformed_image,
                        'input_ids': formatted_text,
                        'answer': str(item.get('count', '')),
                    }
                    buffer.append(sample)
                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            random.shuffle(buffer)
                        for buf_item in buffer:
                            yield buf_item
                        buffer = []
                except Exception as e:
                    print(f'Warning: CountingDataset error on index {idx}: {e}')
                    continue

            if buffer:
                if self.shuffle:
                    random.shuffle(buffer)
                for buf_item in buffer:
                    yield buf_item

            if not self.repeat:
                break

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity', 'answer'):
                batched[k] = torch.stack(v, dim=0)
        return batched


class RelPositionDataset(IterableDataset):
    """HuggingFace heez/relative-position-new dataset for relative-position VQA.

    Schema (verified): image (PIL 512x512), question (str with `Response format:` line),
    answer (full sentence containing one position keyword), position (str: one of
    top-left/top-right/bottom-left/bottom-right). 4 classes ~uniform.
    """

    def __init__(self,
                 rank: int = 0,
                 world_size: int = 1,
                 tokenizer=None,
                 resolution: int = 512,
                 max_length: int = 8000,
                 num_workers: int = 1,
                 shuffle: bool = True,
                 repeat: bool = True,
                 buffer_size: int = 100):
        super().__init__()
        from datasets import load_dataset as hf_load_dataset
        ds = hf_load_dataset("heez/relative-position-new", split="train")
        print(f"RelPositionDataset: loaded {len(ds)} samples from heez/relative-position-new[train]")
        all_indices = list(range(len(ds)))
        self.indices = all_indices[rank::world_size]
        self.ds = ds
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.max_length = max_length
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self._sot_token = '<|startoftext|>'
        self._trailing_asst_header = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        self._assistant_header = '<|start_header_id|>assistant<|end_header_id|>'

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_indices = self.indices[worker_info.id::worker_info.num_workers]
        else:
            worker_indices = self.indices

        while True:
            index_list = list(worker_indices)
            if self.shuffle:
                random.shuffle(index_list)
            buffer = []
            for idx in index_list:
                try:
                    item = self.ds[idx]
                    image = item['image']
                    if not isinstance(image, Image.Image):
                        image = Image.fromarray(image).convert('RGB')
                    else:
                        image = image.convert('RGB')

                    question = item['question']
                    raw_answer = str(item['answer']).strip()
                    if not raw_answer:
                        continue

                    transformed_image = image_transform_squash(
                        {'images': image}, resolution=self.resolution
                    )['images']

                    messages = [
                        {'role': 'user', 'content': question},
                        {'role': 'assistant', 'content': raw_answer},
                    ]
                    formatted_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if formatted_text.startswith(self._sot_token):
                        formatted_text = formatted_text[len(self._sot_token):]
                    # Trailing 빈 assistant generation prompt 제거 (개행 개수에 robust)
                    hidx = formatted_text.rfind(self._trailing_asst_header)
                    if hidx != -1 and formatted_text[hidx + len(self._trailing_asst_header):].strip() == '':
                        formatted_text = formatted_text[:hidx]
                    # Sanity: answer 앞 assistant header + 그 뒤 정답 텍스트가 있어야 함
                    if self._assistant_header not in formatted_text:
                        continue
                    if not formatted_text.rsplit(self._assistant_header, 1)[-1].strip():
                        continue

                    if len(self.tokenizer(formatted_text)['input_ids']) > self.max_length:
                        continue

                    sample = {
                        'images': transformed_image,
                        'input_ids': formatted_text,
                        'answer': str(item.get('position', '')),
                    }
                    buffer.append(sample)
                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            random.shuffle(buffer)
                        for buf_item in buffer:
                            yield buf_item
                        buffer = []
                except Exception as e:
                    print(f'Warning: RelPositionDataset error on index {idx}: {e}')
                    continue

            if buffer:
                if self.shuffle:
                    random.shuffle(buffer)
                for buf_item in buffer:
                    yield buf_item

            if not self.repeat:
                break

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity', 'answer'):
                batched[k] = torch.stack(v, dim=0)
        return batched


def image_transform_clip(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    sample["images"] = image
    return sample

def image_transform_squash(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(image)
    sample["images"] = image
    return sample

def image_transform_pad(sample, resolution=256, fill_color=(255, 255, 255)):
    image = sample["images"]
    w, h = image.size
    if w == h:
        padded_image = image
    elif w < h:
        padding_needed = h - w
        padding_left = padding_needed // 2
        padding_right = padding_needed - padding_left
        pad_transform = transforms.Pad((padding_left, 0, padding_right, 0), fill=fill_color, padding_mode='constant')
        padded_image = pad_transform(image)
    else:
        padding_needed = w - h
        padding_top = padding_needed // 2
        padding_bottom = padding_needed - padding_top
        pad_transform = transforms.Pad((0, padding_top, 0, padding_bottom), fill=fill_color, padding_mode='constant')
        padded_image = pad_transform(image)
    image_resized = transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC)(padded_image)
    image_tensor = transforms.ToTensor()(image_resized)
    image_normalized = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image_tensor)
    sample["images"] = image_normalized
    return sample

if __name__ == '__main__':
    data_path = "/data_storage/shared/datasets/falcon-refinedweb/data/data/*.parquet"
    dataset = RefinedWebDataset(
        data_path=data_path,
        max_length=8000,
        buffer_size=0,
    )
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        dataset, 
        batch_size=1,
        sampler=None, 
        collate_fn=dataset.collate_fn,
        num_workers=0
    )
    
    print("Starting data loading test...")
    for i, batch in enumerate(train_dataloader):
        if i == 0:
            print(batch)
            print(f"Batch size: {len(batch['input_ids'])}")
            print(f"First sample length: {len(batch['input_ids'][0])}")
        if i >= 5:
            break
    print("Data loading test complete")
