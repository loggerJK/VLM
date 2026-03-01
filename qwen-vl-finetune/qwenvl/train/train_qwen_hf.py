# Training entry point for HuggingFace dataset-based tasks (counting, OCR).
#
# Key differences from train_qwen.py:
# - Loads datasets from HuggingFace (not JSON annotation files)
# - LoRA with resume support (Janus convention)
# - No DeepSpeed (plain DDP, compatible with bnb 8-bit AdamW)
# - Validation callback for periodic evaluation and checkpoint saving

import os
import logging
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
print(f"Adding project root to sys.path: {project_root}")

# project_root = Path(__file__).parent.parent.parent.parent
# print(f"Adding project root to sys.path: {project_root}")
# sys.path.append(str(project_root))


from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from qwenvl.data.hf_dataset import HFCountingDataset, HFOCRSyntheticDataset, HFCelebDataset
from qwenvl.data.data_processor import (
    DataCollatorForSupervisedDataset,
    FlattenedDataCollatorForSupervisedDataset,
)
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.train.callbacks import QwenValidationCallback
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # --- Load model ---
    if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"

    rank0_print(f"Loaded model: {model_args.model_name_or_path} ({model.__class__.__name__})")

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # --- LoRA setup ---
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType

    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    if training_args.resume_checkpoint:
        rank0_print(f"Resuming LoRA from: {training_args.resume_checkpoint}")
        model = PeftModel.from_pretrained(model, training_args.resume_checkpoint)
        # Re-enable requires_grad for LoRA params
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
    else:
        rank0_print("Initializing new LoRA adapter")
        lora_config = LoraConfig(
            r=training_args.lora_r or 16,
            lora_alpha=training_args.lora_alpha or 32,
            lora_dropout=training_args.lora_dropout or 0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "down_proj", "gate_proj",
            ],
            bias="none",
            exclude_modules=".*visual.*", 
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    # Trainable Parameter 목록 직접 출력 (LoRA 파라미터만 출력)
    rank0_print("Trainable parameters:")
    for n, p in model.named_parameters():
        if p.requires_grad:
            rank0_print(n)

    # --- Build dataset ---
    task = data_args.task
    if task == "counting":
        train_dataset = HFCountingDataset(processor, data_args)
    elif task == "ocr":
        train_dataset = HFOCRSyntheticDataset(processor, data_args)
    elif task == "celeb":
        train_dataset = HFCelebDataset(processor, data_args)
    else:
        raise ValueError(f"Unknown task: {task!r}. Must be 'counting', 'ocr', or 'celeb'.")

    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
    else:
        data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)

    # --- Validation callback ---
    val_callback = QwenValidationCallback(
        processor=processor,
        task=task,
        training_args=training_args,
        log_freq=training_args.val_log_freq,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        callbacks=[val_callback],
    )
    val_callback.trainer = trainer

    # --- Train ---
    trainer.train()
    trainer.save_state()

    # --- Final save ---
    model.config.use_cache = True
    final_path = os.path.join(training_args.output_dir, "final_model")
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    rank0_print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    train(attn_implementation="sdpa")
