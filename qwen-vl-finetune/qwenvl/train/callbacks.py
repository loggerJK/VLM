"""
QwenValidationCallback — Janus-style validation and checkpoint saving for Qwen2.5-VL.

Provides:
- Periodic LoRA checkpoint saving with Janus naming: epoch{E}_step-{S}/, epoch{E}/, final_model/
- Periodic validation with counting accuracy or OCR metrics
- wandb logging of validation results
"""

import os
import re
import logging
import torch
import torch.distributed as dist
from transformers import TrainerCallback
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)


def _is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _save_lora_checkpoint(model, processor, save_path):
    """Save LoRA adapter and processor to save_path."""
    os.makedirs(save_path, exist_ok=True)
    # unwrap DDP if needed
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    logger.info(f"Saved checkpoint to {save_path}")


class QwenValidationCallback(TrainerCallback):
    """Callback for periodic validation and Janus-style checkpoint saving."""

    def __init__(self, processor, task, training_args, log_freq=500):
        self.processor = processor
        self.task = task
        self.training_args = training_args
        self.log_freq = log_freq
        self.save_steps = training_args.save_steps_callback
        self.trainer = None  # set externally after Trainer construction

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not _is_main_process():
            return

        # Log optimized param names
        param_names = [n for n, p in model.named_parameters() if p.requires_grad]
        param_file = os.path.join(args.output_dir, "trainable_params.txt")
        with open(param_file, "w") as f:
            for name in param_names:
                f.write(name + "\n")
        logger.info(f"Logged {len(param_names)} trainable params to {param_file}")

        try:
            import wandb
            if wandb.run is not None:
                wandb.save(param_file)
        except ImportError:
            pass
        
        print("[INFO] Trying to run initial validation...")
        self._run_validation(model, 0) # Initial validation at step 0

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if not _is_main_process():
            return

        step = state.global_step

        # Checkpoint saving
        if self.save_steps > 0 and step > 0 and step % self.save_steps == 0:
            epoch = int(state.epoch) if state.epoch is not None else 0
            save_path = os.path.join(args.output_dir, f"epoch{epoch}_step-{step}")
            _save_lora_checkpoint(model, self.processor, save_path)

        # Validation
        if self.log_freq > 0 and step > 0 and step % self.log_freq == 0:
            self._run_validation(model, step)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not _is_main_process():
            return

        epoch = int(state.epoch) if state.epoch is not None else 0
        save_path = os.path.join(args.output_dir, f"epoch{epoch}")
        _save_lora_checkpoint(model, self.processor, save_path)

    def _run_validation(self, model, step):
        model.eval()
        try:
            if self.task == "counting":
                self._validate_counting(model, step)
            elif self.task == "ocr":
                self._validate_ocr(model, step)
        except Exception as e:
            logger.warning(f"Validation failed at step {step}: {e}")
        finally:
            model.train()
            
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

    @torch.no_grad()
    def _validate_counting(self, model, step):
        from datasets import load_dataset

        ds = load_dataset(
            "heez/pixmo-point-count-gen-und", split="val_und", streaming=True
        )

        correct = 0
        total = 0
        predictions = []
        device = next(model.parameters()).device
        unwrapped = model.module if hasattr(model, "module") else model

        for sample in ds:
            if total >= 100:
                break

            pil_image = sample["image"]
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            question = str(sample.get("question", ""))
            gt_answer = str(sample.get("answer", ""))

            if not question or not gt_answer:
                continue

            # Qwen inference pattern
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, return_tensors="pt"
            ).to(device)

            output_ids = unwrapped.generate(**inputs, max_new_tokens=50, do_sample=False)
            # Trim input tokens
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            pred_text = self.processor.decode(generated, skip_special_tokens=True).strip()

            # # Extract number from **N** pattern or plain number
            # match = re.search(r"\*\*(\d+)\*\*", pred_text)
            # if match:
            #     pred_num = match.group(1)
            # else:
            #     nums = re.findall(r"\d+", pred_text)
            #     pred_num = nums[0] if nums else pred_text

            # gt_match = re.search(r"\*\*(\d+)\*\*", gt_answer)
            # if gt_match:
            #     gt_num = gt_match.group(1)
            # else:
            #     gt_nums = re.findall(r"\d+", gt_answer)
            #     gt_num = gt_nums[0] if gt_nums else gt_answer

            pred_num = self.extract_number_fixed(pred_text)
            gt_num = self.extract_number_fixed(gt_answer)

            is_correct = pred_num == gt_num
            if is_correct:
                correct += 1
            total += 1
            
            print("="*50)
            print(f"[GT]")
            print(f"{gt_answer} (extracted: {gt_num})")
            print(f"[Prediction]")
            print(f"{pred_text} (extracted: {pred_num})")

            predictions.append({
                "question": question,
                "gt": gt_answer,
                "pred": pred_text,
                "correct": is_correct,
            })

        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"[Step {step}] Counting val accuracy: {accuracy:.4f} ({correct}/{total})")

        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"val/accuracy": accuracy, "val/step": step}, step=step)
                table = wandb.Table(
                    columns=["question", "gt", "pred", "correct"],
                    data=[[p["question"], p["gt"], p["pred"], p["correct"]] for p in predictions[:20]],
                )
                wandb.log({"val/predictions": table}, step=step)
        except ImportError:
            pass

    @torch.no_grad()
    def _validate_ocr(self, model, step):
        from datasets import load_dataset
        from .ocr_render import generate_image

        ds = load_dataset(
            "agentlans/high-quality-english-sentences", split="test"
        )

        exact_match = 0
        char_correct = 0
        char_total = 0
        total = 0
        predictions = []
        device = next(model.parameters()).device
        unwrapped = model.module if hasattr(model, "module") else model

        for i in range(min(100, len(ds))):
            sample = ds[i]
            text = sample["text"]
            gt_answer = text.replace("\n", " ").strip()[:120]

            try:
                pil_image = generate_image(
                    "# " + text,
                    template="clean_light",
                    width=512,
                    height=0,
                    quality=100,
                )
            except Exception:
                continue

            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            question = "Extract all text from the image."

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text_prompt], images=image_inputs, return_tensors="pt"
            ).to(device)

            output_ids = unwrapped.generate(**inputs, max_new_tokens=150, do_sample=False)
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            pred_text = self.processor.decode(generated, skip_special_tokens=True).strip()

            # Exact match
            if pred_text == gt_answer:
                exact_match += 1

            # Character-level accuracy
            min_len = min(len(pred_text), len(gt_answer))
            chars_match = sum(1 for a, b in zip(pred_text[:min_len], gt_answer[:min_len]) if a == b)
            char_correct += chars_match
            char_total += max(len(pred_text), len(gt_answer))
            total += 1

            predictions.append({
                "gt": gt_answer[:50],
                "pred": pred_text[:50],
                "exact": pred_text == gt_answer,
            })

        em_acc = exact_match / total if total > 0 else 0.0
        char_acc = char_correct / char_total if char_total > 0 else 0.0
        logger.info(f"[Step {step}] OCR val — EM: {em_acc:.4f}, Char: {char_acc:.4f} ({total} samples)")

        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "val/exact_match": em_acc,
                    "val/char_accuracy": char_acc,
                    "val/step": step,
                }, step=step)
                table = wandb.Table(
                    columns=["gt", "pred", "exact"],
                    data=[[p["gt"], p["pred"], p["exact"]] for p in predictions[:20]],
                )
                wandb.log({"val/predictions": table}, step=step)
        except ImportError:
            pass
