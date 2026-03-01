"""
HuggingFace Dataset classes for counting and OCR synthetic tasks.

Produces the same dict format as LazySupervisedDataset._get_item() so the
existing DataCollator classes can be reused without modification.
"""

import logging
import time
import torch
from typing import Dict
from torch.utils.data import Dataset

from datasets import load_dataset

from .data_processor import IGNORE_INDEX, update_processor_pixels, rank0_print
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

logger = logging.getLogger(__name__)


def _get_rope_fn(model_type: str):
    if model_type == "qwen3vl":
        return get_rope_index_3
    elif model_type == "qwen2.5vl":
        return get_rope_index_25
    elif model_type == "qwen2vl":
        return get_rope_index_2
    else:
        raise ValueError(f"model_type: {model_type} not supported")


def _tokenize_and_label(messages, processor):
    """Tokenize messages and apply label masking (same logic as preprocess_qwen_visual)."""
    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        # 77091 = assistant marker token, 151645 = end token
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


def _compute_position_ids(data_dict, get_rope_index, merge_size):
    """Compute RoPE position_ids and attention_mask matching _get_item() format."""
    seq_len = data_dict["input_ids"][0].size(0)

    if "image_grid_thw" in data_dict:
        grid_thw = data_dict.get("image_grid_thw")
        if not isinstance(grid_thw, (list, tuple)):
            grid_thw = [grid_thw]
    else:
        grid_thw = None

    position_ids, _ = get_rope_index(
        merge_size,
        data_dict["input_ids"],
        image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
        video_grid_thw=None,
        second_per_grid_ts=None,
    )

    data_dict["position_ids"] = position_ids
    data_dict["attention_mask"] = [seq_len]
    return data_dict


class HFCountingDataset(Dataset):
    """Dataset for counting task using heez/pixmo-point-count-gen-und from HuggingFace."""

    def __init__(self, processor, data_args):
        super().__init__()

        rank0_print("Loading counting dataset: heez/pixmo-point-count-gen-und")
        hf_path = data_args.hf_data_path if data_args.hf_data_path else "heez/pixmo-point-count-gen-und"
        ds = load_dataset(hf_path, split="train")

        # Filter understanding-only samples (batched for speed; avoid decoding images)
        def is_understanding_batched(question_count, answer_count, descriptions):
            results = []
            for q, a, d in zip(question_count, answer_count, descriptions):
                has_q = q is not None and q != ""
                has_a = a is not None and a != ""
                no_desc = not d
                results.append(has_q and has_a and no_desc)
            return results

        self.dataset = ds.filter(
            is_understanding_batched,
            batched=True,
            batch_size=1000,
            input_columns=["question_count", "answer_count", "descriptions"],
        )
        rank0_print(f"Counting dataset: {len(self.dataset)} understanding samples")

        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.get_rope_index = _get_rope_fn(data_args.model_type)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        for attempt in range(3):
            try:
                return self._get_item(i)
            except Exception as e:
                logger.warning(f"[Counting Try #{attempt}] Failed sample {i}: {e}")
                time.sleep(1)

        # Final attempt — let it raise
        return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sample = self.dataset[i]

        pil_image = sample["image"]
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        question = str(sample["question_count"])
        answer = str(sample["answer_count"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]

        data_dict = _tokenize_and_label(messages, self.processor)
        data_dict = _compute_position_ids(data_dict, self.get_rope_index, self.merge_size)
        return data_dict


class HFOCRSyntheticDataset(Dataset):
    """Dataset for OCR task using agentlans/high-quality-english-sentences from HuggingFace."""

    def __init__(self, processor, data_args):
        super().__init__()

        rank0_print("Loading OCR dataset: agentlans/high-quality-english-sentences")
        hf_path = data_args.hf_data_path if data_args.hf_data_path else "agentlans/high-quality-english-sentences"
        ds = load_dataset(hf_path, split="train")

        num_samples = min(data_args.ocr_num_samples, len(ds))
        self.dataset = ds.select(range(num_samples))
        rank0_print(f"OCR dataset: {len(self.dataset)} samples")

        self.image_width = data_args.ocr_image_width

        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.get_rope_index = _get_rope_fn(data_args.model_type)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        for attempt in range(3):
            try:
                return self._get_item(i)
            except Exception as e:
                logger.warning(f"[OCR Try #{attempt}] Failed sample {i}: {e}")
                time.sleep(1)

        return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sample = self.dataset[i]

        text = sample["text"]
        # Render image on-the-fly (lazy import to avoid requiring markdown for non-OCR tasks)
        from .ocr_render import generate_image
        pil_image = generate_image(
            "# " + text,
            template="random",
            width=self.image_width,
            height=0,
            quality=100,
        )
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        question = "Extract all text from the image."
        answer = text.replace("\n", " ").strip()[:120]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]

        data_dict = _tokenize_and_label(messages, self.processor)
        data_dict = _compute_position_ids(data_dict, self.get_rope_index, self.merge_size)
        return data_dict


class HFCelebDataset(Dataset):
    """Dataset for celebrity recognition task using heez/celeb-recognition from HuggingFace."""

    def __init__(self, processor, data_args):
        super().__init__()

        rank0_print("Loading celeb dataset: heez/celeb-recognition")
        hf_path = data_args.hf_data_path if data_args.hf_data_path else "heez/celeb-recognition"
        self.dataset = load_dataset(hf_path, split="train")
        rank0_print(f"Celeb dataset: {len(self.dataset)} samples")

        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.get_rope_index = _get_rope_fn(data_args.model_type)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        for attempt in range(3):
            try:
                return self._get_item(i)
            except Exception as e:
                logger.warning(f"[Celeb Try #{attempt}] Failed sample {i}: {e}")
                time.sleep(1)

        return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sample = self.dataset[i]

        pil_image = sample["image"]
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        question = str(sample["question"])
        answer = str(sample["answer"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]

        data_dict = _tokenize_and_label(messages, self.processor)
        data_dict = _compute_position_ids(data_dict, self.get_rope_index, self.merge_size)
        return data_dict
