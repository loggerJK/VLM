"""HuggingFace dataset loaders for counting, OCR, and synthetic-OCR tasks.

Each dataset class inherits from DistributedIterableDataset and yields samples
in BAGEL's sequence_plan format (matching SftJSONLIterableDataset / T2IIterableDataset).

Understanding (und) format:
    vit_image(loss=0) → text(question, loss=0) → text(answer, loss=1)

Generation (gen) format:
    text(caption, loss=0, enable_cfg=1) → vae_image(loss=1)
"""

import random
import traceback

from datasets import load_dataset
from PIL import Image

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset


class HFDatasetBase(DistributedIterableDataset):
    """Base class for HF-backed datasets.

    Overrides ``set_epoch`` to handle integer index lists (the parent class
    only supports str/tuple paths used by parquet/jsonl datasets).
    """

    def set_epoch(self, seed=42):
        if self.data_paths is None:
            return
        data_paths = sorted(self.data_paths)
        self.rng.seed(seed)
        self.rng.shuffle(data_paths)

        num_files_per_rank = len(data_paths) // max(self.world_size, 1)
        local_start = self.local_rank * num_files_per_rank
        local_end = (self.local_rank + 1) * num_files_per_rank
        self.num_files_per_rank = num_files_per_rank
        self.data_paths_per_rank = data_paths[local_start:local_end]

    def get_data_paths(self, *args, **kwargs):
        return self.data_paths


# ---------------------------------------------------------------------------
# Counting datasets  (HF: heez/pixmo-point-count-gen-und)
# ---------------------------------------------------------------------------

class HFCountingUndDataset(HFDatasetBase):
    """Counting understanding: image + question → answer (CE loss)."""

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        data_dir_list=None,
        num_used_data=None,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        **kwargs,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status

        if local_rank == 0:
            print(f"[{dataset_name}] Loading HF dataset: heez/pixmo-point-count-gen-und ...")
        self.hf_dataset = load_dataset(
            "heez/pixmo-point-count-gen-und", split="train"
        )
        
        # # Filter samples : 'descriptions' is None
        # self.hf_dataset = self.hf_dataset.filter(
        #     lambda descriptions: descriptions is None,
        #     input_columns = ["descriptions"] ,
        #     num_proc=64,
        # )
        
        # All samples have images; no filter needed
        if local_rank == 0:
            print(f"[{dataset_name}] Loaded {len(self.hf_dataset)} samples")

        # Create pseudo data_paths for distributed sharding
        n = len(self.hf_dataset)
        self.data_paths = list(range(n))
        self.set_epoch()

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if data_paths_per_worker is None:
            data_paths_per_worker = list(range(len(self.hf_dataset)))
            worker_id = 0

        if self.data_status is not None and worker_id in self.data_status:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        transform_stride = self.transform.stride
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            indices = data_paths_per_worker[row_start_id:]
            for pos, idx in enumerate(indices, start=row_start_id):
                try:
                    item = self.hf_dataset[idx]
                    image = pil_img2rgb(item["image"])
                    question = item.get("question_count") or item.get("question") or "How many objects are there?"
                    answer = str(item.get("answer_count") or item.get("answer") or "")
                    if not question or not answer:
                        continue

                    image_tensor = self.transform(image)
                    height, width = image_tensor.shape[1:]
                    num_img_tokens = width * height // (transform_stride ** 2)

                    question_ids = self.tokenizer.encode(question)
                    answer_ids = self.tokenizer.encode(answer)
                    num_tokens = len(question_ids) + num_img_tokens + len(answer_ids)

                    sequence_plan = [
                        {"type": "vit_image", "enable_cfg": 0, "loss": 0, "special_token_loss": 0, "special_token_label": None},
                        {"type": "text", "enable_cfg": 0, "loss": 0, "special_token_loss": 0, "special_token_label": None},
                        {"type": "text", "enable_cfg": 0, "loss": 1, "special_token_loss": 0, "special_token_label": None},
                    ]

                    yield dict(
                        image_tensor_list=[image_tensor],
                        text_ids_list=[question_ids, answer_ids],
                        sequence_plan=sequence_plan,
                        num_tokens=num_tokens,
                        data_indexes={
                            "data_indexes": pos,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        },
                    )
                except Exception:
                    traceback.print_exc()
                    continue

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class HFCountingGenDataset(HFDatasetBase):
    """Counting generation: caption → image (MSE loss via VAE)."""

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        data_dir_list=None,
        num_used_data=None,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        **kwargs,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status

        if local_rank == 0:
            print(f"[{dataset_name}] Loading HF dataset: heez/pixmo-point-count-gen-und ...")
        self.hf_dataset = load_dataset(
            "heez/pixmo-point-count-gen-und", split="train"
        )
        # All samples have images; no filter needed
        if local_rank == 0:
            print(f"[{dataset_name}] Loaded {len(self.hf_dataset)} samples")

        n = len(self.hf_dataset)
        self.data_paths = list(range(n))
        self.set_epoch()

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if data_paths_per_worker is None:
            data_paths_per_worker = list(range(len(self.hf_dataset)))
            worker_id = 0

        if self.data_status is not None and worker_id in self.data_status:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        transform_stride = self.transform.stride
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            indices = data_paths_per_worker[row_start_id:]
            for pos, idx in enumerate(indices, start=row_start_id):
                try:
                    item = self.hf_dataset[idx]
                    image = pil_img2rgb(item["image"])
                    # descriptions is List[str]; pick a random one as caption
                    descriptions = item.get("descriptions") or []
                    if isinstance(descriptions, list) and len(descriptions) > 0:
                        caption = random.choice(descriptions)
                    elif isinstance(descriptions, str) and descriptions:
                        caption = descriptions
                    else:
                        question = item.get("question_count") or item.get("question") or ""
                        answer = str(item.get("answer_count") or item.get("answer") or "")
                        caption = f"{question} {answer}"
                    if not caption.strip():
                        continue

                    image_tensor = self.transform(image)
                    height, width = image_tensor.shape[1:]
                    print(f"Image height: {height}, width: {width}")
                    num_img_tokens = width * height // (transform_stride ** 2)

                    caption_ids = self.tokenizer.encode(caption)
                    num_tokens = len(caption_ids) + num_img_tokens
                    print(f"Caption length (tokens): {len(caption_ids)}, Image tokens: {num_img_tokens}, Total: {num_tokens}")

                    sequence_plan = [
                        {"type": "text", "enable_cfg": 1, "loss": 0, "special_token_loss": 0, "special_token_label": None},
                        {"type": "vae_image", "enable_cfg": 0, "loss": 1, "special_token_loss": 0, "special_token_label": self.tokenizer.convert_tokens_to_ids("<|endofimage|>")}, 
                    ]

                    yield dict(
                        image_tensor_list=[image_tensor],
                        text_ids_list=[caption_ids],
                        sequence_plan=sequence_plan,
                        num_tokens=num_tokens,
                        data_indexes={
                            "data_indexes": pos,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        },
                    )
                except Exception:
                    traceback.print_exc()
                    continue

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


# ---------------------------------------------------------------------------
# OCR datasets  (user-specified HF dataset via hf_dataset_path)
# ---------------------------------------------------------------------------

class HFOCRUndDataset(HFDatasetBase):
    """OCR understanding: image + "Extract all text" → answer (CE loss)."""

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        data_dir_list=None,
        num_used_data=None,
        hf_dataset_path=None,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        **kwargs,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status

        assert hf_dataset_path is not None, "hf_dataset_path is required for OCR datasets"
        if local_rank == 0:
            print(f"[{dataset_name}] Loading HF dataset: {hf_dataset_path} ...")
        self.hf_dataset = load_dataset(hf_dataset_path, split="train")
        if local_rank == 0:
            print(f"[{dataset_name}] Loaded {len(self.hf_dataset)} samples")

        n = len(self.hf_dataset)
        self.data_paths = list(range(n))
        self.set_epoch()

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if data_paths_per_worker is None:
            data_paths_per_worker = list(range(len(self.hf_dataset)))
            worker_id = 0

        if self.data_status is not None and worker_id in self.data_status:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        transform_stride = self.transform.stride
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            indices = data_paths_per_worker[row_start_id:]
            for pos, idx in enumerate(indices, start=row_start_id):
                try:
                    item = self.hf_dataset[idx]
                    image = pil_img2rgb(item["image"])
                    question = "Extract all text from the image."
                    answer = item.get("text", item.get("answer", item.get("ground_truth", "")))

                    image_tensor = self.transform(image)
                    height, width = image_tensor.shape[1:]
                    num_img_tokens = width * height // (transform_stride ** 2)

                    question_ids = self.tokenizer.encode(question)
                    answer_ids = self.tokenizer.encode(answer)
                    num_tokens = len(question_ids) + num_img_tokens + len(answer_ids)

                    sequence_plan = [
                        {"type": "vit_image", "enable_cfg": 0, "loss": 0, "special_token_loss": 0, "special_token_label": None},
                        {"type": "text", "enable_cfg": 0, "loss": 0, "special_token_loss": 0, "special_token_label": None},
                        {"type": "text", "enable_cfg": 0, "loss": 1, "special_token_loss": 0, "special_token_label": None},
                    ]

                    yield dict(
                        image_tensor_list=[image_tensor],
                        text_ids_list=[question_ids, answer_ids],
                        sequence_plan=sequence_plan,
                        num_tokens=num_tokens,
                        data_indexes={
                            "data_indexes": pos,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        },
                    )
                except Exception:
                    traceback.print_exc()
                    continue

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class HFOCRGenDataset(HFDatasetBase):
    """OCR generation: caption → image (MSE loss via VAE)."""

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        data_dir_list=None,
        num_used_data=None,
        hf_dataset_path=None,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        **kwargs,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status

        assert hf_dataset_path is not None, "hf_dataset_path is required for OCR datasets"
        if local_rank == 0:
            print(f"[{dataset_name}] Loading HF dataset: {hf_dataset_path} ...")
        self.hf_dataset = load_dataset(hf_dataset_path, split="train")
        if local_rank == 0:
            print(f"[{dataset_name}] Loaded {len(self.hf_dataset)} samples")

        n = len(self.hf_dataset)
        self.data_paths = list(range(n))
        self.set_epoch()

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if data_paths_per_worker is None:
            data_paths_per_worker = list(range(len(self.hf_dataset)))
            worker_id = 0

        if self.data_status is not None and worker_id in self.data_status:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        transform_stride = self.transform.stride
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            indices = data_paths_per_worker[row_start_id:]
            for pos, idx in enumerate(indices, start=row_start_id):
                try:
                    item = self.hf_dataset[idx]
                    image = pil_img2rgb(item["image"])
                    text = item.get("text", item.get("answer", item.get("ground_truth", "")))
                    caption = (
                        f"A Mathpix Markdown format with sharp, legible black text. "
                        f"High-resolution typography, top-down view. "
                        f"The text is rendered in natural left-to-right, top-to-bottom reading order. "
                        f"The text reads:\n{text}"
                    )

                    image_tensor = self.transform(image)
                    height, width = image_tensor.shape[1:]
                    num_img_tokens = width * height // (transform_stride ** 2)

                    caption_ids = self.tokenizer.encode(caption)
                    num_tokens = len(caption_ids) + num_img_tokens

                    sequence_plan = [
                        {"type": "text", "enable_cfg": 1, "loss": 0, "special_token_loss": 0, "special_token_label": None},
                        {"type": "vae_image", "enable_cfg": 0, "loss": 1, "special_token_loss": 0, "special_token_label": None},
                    ]

                    yield dict(
                        image_tensor_list=[image_tensor],
                        text_ids_list=[caption_ids],
                        sequence_plan=sequence_plan,
                        num_tokens=num_tokens,
                        data_indexes={
                            "data_indexes": pos,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        },
                    )
                except Exception:
                    traceback.print_exc()
                    continue

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


# ---------------------------------------------------------------------------
# Synthetic OCR datasets  (HF: agentlans/high-quality-english-sentences)
# ---------------------------------------------------------------------------

def _generate_ocr_image_safe(text, width=512, height=512):
    """Generate a synthetic OCR image, falling back to a blank image on error."""
    try:
        from .ocr_render import generate_image
        img = generate_image(
            "# " + text,
            template="random",
            width=width,
            height=height,
            quality=100,
        )
        return pil_img2rgb(img)
    except Exception:
        traceback.print_exc()
        # Fallback: white image with no text
        return Image.new("RGB", (width, height), (255, 255, 255))


class HFSyntheticOCRUndDataset(HFDatasetBase):
    """Synthetic OCR understanding: rendered text image + question → answer (CE loss)."""

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        data_dir_list=None,
        num_used_data=None,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        ocr_image_size=512,
        num_samples=200000,
        **kwargs,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.ocr_image_size = ocr_image_size

        if local_rank == 0:
            print(f"[{dataset_name}] Loading HF dataset: agentlans/high-quality-english-sentences ...")
        self.hf_dataset = load_dataset(
            "agentlans/high-quality-english-sentences", split="train"
        )
        # Use first N samples
        n = min(num_samples, len(self.hf_dataset))
        self.hf_dataset = self.hf_dataset.select(range(n))
        if local_rank == 0:
            print(f"[{dataset_name}] Using {len(self.hf_dataset)} samples")

        self.data_paths = list(range(len(self.hf_dataset)))
        self.set_epoch()

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if data_paths_per_worker is None:
            data_paths_per_worker = list(range(len(self.hf_dataset)))
            worker_id = 0

        if self.data_status is not None and worker_id in self.data_status:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        transform_stride = self.transform.stride
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            indices = data_paths_per_worker[row_start_id:]
            for pos, idx in enumerate(indices, start=row_start_id):
                try:
                    item = self.hf_dataset[idx]
                    raw_text = item["text"]
                    answer = raw_text.replace("\n", " ").strip()[:120]

                    # Generate synthetic OCR image from text
                    image = _generate_ocr_image_safe(
                        answer, width=self.ocr_image_size, height=self.ocr_image_size
                    )

                    question = "Extract all text from the image."

                    image_tensor = self.transform(image)
                    height, width = image_tensor.shape[1:]
                    num_img_tokens = width * height // (transform_stride ** 2)

                    question_ids = self.tokenizer.encode(question)
                    answer_ids = self.tokenizer.encode(answer)
                    num_tokens = len(question_ids) + num_img_tokens + len(answer_ids)

                    sequence_plan = [
                        {"type": "vit_image", "enable_cfg": 0, "loss": 0, "special_token_loss": 0, "special_token_label": None},
                        {"type": "text", "enable_cfg": 0, "loss": 0, "special_token_loss": 0, "special_token_label": None},
                        {"type": "text", "enable_cfg": 0, "loss": 1, "special_token_loss": 0, "special_token_label": None},
                    ]

                    yield dict(
                        image_tensor_list=[image_tensor],
                        text_ids_list=[question_ids, answer_ids],
                        sequence_plan=sequence_plan,
                        num_tokens=num_tokens,
                        data_indexes={
                            "data_indexes": pos,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        },
                    )
                except Exception:
                    traceback.print_exc()
                    continue

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")


class HFSyntheticOCRGenDataset(HFDatasetBase):
    """Synthetic OCR generation: caption → rendered text image (MSE loss via VAE)."""

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        data_dir_list=None,
        num_used_data=None,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        ocr_image_size=512,
        num_samples=200000,
        **kwargs,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.ocr_image_size = ocr_image_size

        if local_rank == 0:
            print(f"[{dataset_name}] Loading HF dataset: agentlans/high-quality-english-sentences ...")
        self.hf_dataset = load_dataset(
            "agentlans/high-quality-english-sentences", split="train"
        )
        n = min(num_samples, len(self.hf_dataset))
        self.hf_dataset = self.hf_dataset.select(range(n))
        if local_rank == 0:
            print(f"[{dataset_name}] Using {len(self.hf_dataset)} samples")

        self.data_paths = list(range(len(self.hf_dataset)))
        self.set_epoch()

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if data_paths_per_worker is None:
            data_paths_per_worker = list(range(len(self.hf_dataset)))
            worker_id = 0

        if self.data_status is not None and worker_id in self.data_status:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        transform_stride = self.transform.stride
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at row#{row_start_id}"
        )

        while True:
            indices = data_paths_per_worker[row_start_id:]
            for pos, idx in enumerate(indices, start=row_start_id):
                try:
                    item = self.hf_dataset[idx]
                    raw_text = item["text"]

                    # Generate synthetic OCR image from text
                    image = _generate_ocr_image_safe(
                        raw_text, width=self.ocr_image_size, height=self.ocr_image_size
                    )

                    caption = (
                        f"A Mathpix Markdown format with sharp, legible black text. "
                        f"High-resolution typography, top-down view. "
                        f"The text is rendered in natural left-to-right, top-to-bottom reading order. "
                        f"The text reads:\n{raw_text}"
                    )

                    image_tensor = self.transform(image)
                    height, width = image_tensor.shape[1:]
                    num_img_tokens = width * height // (transform_stride ** 2)

                    caption_ids = self.tokenizer.encode(caption)
                    num_tokens = len(caption_ids) + num_img_tokens

                    sequence_plan = [
                        {"type": "text", "enable_cfg": 1, "loss": 0, "special_token_loss": 0, "special_token_label": None},
                        {"type": "vae_image", "enable_cfg": 0, "loss": 1, "special_token_loss": 0, "special_token_label": None},
                    ]

                    yield dict(
                        image_tensor_list=[image_tensor],
                        text_ids_list=[caption_ids],
                        sequence_plan=sequence_plan,
                        num_tokens=num_tokens,
                        data_indexes={
                            "data_indexes": pos,
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        },
                    )
                except Exception:
                    traceback.print_exc()
                    continue

            row_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
