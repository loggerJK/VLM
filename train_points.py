# from multiprocess.managers import dispatch
# from cupy._manipulation.split import split
import os
import torch
import re
import argparse
import requests
from io import BytesIO
from datasets import load_dataset, load_from_disk
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoConfig
from transformers import TrainerCallback
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from accelerate.utils import gather_object
from scipy.optimize import linear_sum_assignment


class EnhancedMultiModalModel(MultiModalityCausalLM):
    """
    Enhanced MultiModalityCausalLM that supports custom forward logic.
    Adds a forward method to MultiModalityCausalLM for training.
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        image_token_masks: torch.BoolTensor = None,
        labels: torch.LongTensor = None,
        **kwargs,
    ):
        """
        Forward method supporting joint processing of images and text.
        """
        if pixel_values is not None and image_token_masks is not None:
            # Process multimodal inputs (replace part of text embeddings with image embeddings)
            inputs_embeds = self._process_multimodal_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_masks=image_token_masks,
            )
        else:
            # Process text-only inputs
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Remove input_ids as we already computed inputs_embeds
        kwargs.pop("inputs_embeds", None)
        
        # Call the actual language model
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return outputs

    def _process_multimodal_inputs(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_token_masks: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Extracts visual features, aligns them, and inserts them into the appropriate positions within the text embedding sequence.
        """
        bs, n = pixel_values.shape[0:2] # (bs, n, 3, h, w)
        images = pixel_values.view(bs * n, *pixel_values.shape[2:]) # (bs * n, 3, h, w)
        
        # Pass through vision encoder and aligner
        image_features = self.vision_model(images) 
        aligned_features = self.aligner(image_features) # (bs * n, n_image_tokens, D)

        # Reshape batch dimensions
        aligned_features = aligned_features.view(bs, n, *aligned_features.shape[1:]) # (bs, n, n_image_tokens, D)
        aligned_features = aligned_features.flatten(1, 2) # (bs, n * n_image_tokens, D)

        # Generate base text embeddings
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Insert aligned image features into image token positions using the mask
        for i in range(bs):
            num_image_tokens = image_token_masks[i].sum().item()
            num_aligned_tokens = aligned_features[i].shape[0]
            if num_image_tokens != num_aligned_tokens:
                raise ValueError(
                    f"Mismatch at sample {i}: "
                    f"image_token_masks has {num_image_tokens} tokens, "
                    f"but aligned_features has {num_aligned_tokens} tokens!"
                )
            text_embeds[i][image_token_masks[i]] = aligned_features[i]

        return text_embeds

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

def extract_number(text):
    """Extract number from model output. Expected format: **<number>**"""
    match = re.search(r"\*\*(\d+)\*\*", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return -1

def calculate_metrics(pred_str, gt_points):
    # Parse pred_str to list of (x, y)
    matches = re.findall(r'x\d+="([\d\.]+)"\s+y\d+="([\d\.]+)"', pred_str)
    pred_points = []
    for x, y in matches:
        try:
            pred_points.append((float(x), float(y)))
        except:
            pass
            
    # gt_points is a list of dicts [{'x':.., 'y':..}] or list of lists
    gt_coords = []
    if gt_points:
        for p in gt_points:
            if isinstance(p, dict):
                gt_coords.append((float(p['x']), float(p['y'])))
            else:
                # Assuming list/tuple
                gt_coords.append((float(p[0]), float(p[1])))
            
    count_dev = abs(len(pred_points) - len(gt_coords))
    
    if len(gt_coords) == 0:
        if len(pred_points) == 0: return 0.0, 1.0, 0.0
        return 100.0, 0.0, float(count_dev) # Max deviation penalty
        
    if len(pred_points) == 0:
        return 100.0, 0.0, float(count_dev) # Max deviation penalty
        
    # Cost matrix for Hungarian matching
    cost_matrix = np.zeros((len(gt_coords), len(pred_points)))
    for i, (gx, gy) in enumerate(gt_coords):
        for j, (px, py) in enumerate(pred_points):
            dist = np.sqrt((gx-px)**2 + (gy-py)**2)
            cost_matrix[i, j] = dist
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_dists = cost_matrix[row_ind, col_ind]
    mean_coord_dev = matched_dists.mean()
    
    # Accuracy (threshold 5.0 units out of 100)
    correct = (matched_dists < 5.0).sum()
    accuracy = correct / len(gt_coords)
    
    return mean_coord_dev, accuracy, float(count_dev)

def collate_fn(batch, processor):
    prepare_list = []
    
    for item in batch:
        image = item.get('image')
        if image is None: continue

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{item['question']}",
                "images": [image]
            },
            {
                "role": "<|Assistant|>",
                "content": item['answer']
            }
        ]
        
        try:
            prepare = processor(
                conversations=conversation,
                images=[image],
                force_batchify=False
            )
            prepare_list.append(prepare)
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    if not prepare_list:
        return {}

    inputs = processor.batchify(prepare_list)
    inputs_dict = dict(inputs)
    
    input_ids = inputs_dict['input_ids']
    labels = input_ids.clone() # (bs, seq_len)
    
    # Get Assistant role token ID
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|Assistant|>")

    for i in range(input_ids.shape[0]):
        # Find Assistant token
        assistant_indices = (input_ids[i] == assistant_token_id).nonzero(as_tuple=True)[0]
        
        if assistant_indices.numel() > 0:
            split_point = assistant_indices[0]
            # Mask everything before assistant response
            labels[i, :split_point + 1] = -100

    # Mask image tokens
    if "images_seq_mask" in inputs_dict:
        image_mask = inputs_dict["images_seq_mask"]
        labels[image_mask] = -100
        inputs_dict["image_token_masks"] = image_mask
    
    inputs_dict["labels"] = labels
    

    # Remove non-tensor fields
    keys_to_remove = [k for k, v in inputs_dict.items() if not isinstance(v, torch.Tensor)]
    for k in keys_to_remove:
        del inputs_dict[k]

    return inputs_dict

class ValidationCallback(TrainerCallback):
    def __init__(self, processor, args, log_freq=500):
        self.processor = processor
        self.log_freq = log_freq
        self.args = args
        self.trainer = None
        self.val_table = None
        self.val_count_table = None
        self.accelerator = Accelerator()
        
    def on_train_begin(self, args, state, control, **kwargs):
        optimized_param_names = [n for n, p in self.trainer.model.named_parameters() if p.requires_grad]
        if self.accelerator.is_main_process:
            print(f"Starting training..., Number of Optimized parameters: {len(optimized_param_names)}")
            if wandb.run is not None:
                wandb.config.update({"optimized_param_names": optimized_param_names})

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.args.save_steps == 0 and self.accelerator.is_main_process:
            checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if self.args.tuning_mode == "lora":
                print(f"\n[Step {state.global_step}] Saving checkpoint to {checkpoint_dir}")
                peft_model = self.trainer.model.module.language_model if hasattr(self.trainer.model, 'module') else self.trainer.model.language_model
                peft_model.save_pretrained(checkpoint_dir)
                self.processor.save_pretrained(checkpoint_dir)
            else:
                self.trainer.save_model(checkpoint_dir)
                self.processor.save_pretrained(checkpoint_dir)
                    
        if state.global_step % self.args.log_freq == 0:
            self.validate(model, state)

    def validate(self, model, state):
        if hasattr(model, "module"):
            unwrap_model = model.module
        else:
            unwrap_model = model
            
        device = next(unwrap_model.parameters()).device
        
        # Clear cache before validation to prevent OOM
        torch.cuda.empty_cache()
        
        model.eval()
        
        # 1. Validation on Counting (Jiwon-Kang/pixmo-count-filtered-imgContained)
        self.validate_counting(unwrap_model, state, device)
        
        # 2. Validation on Points (Multi-GPU enabled)
        self.validate_points(unwrap_model, state, device)
        
        model.train()

    def get_qa_prompt(self, label):
        response_example = f'<points x1="<number of x1>" y1="<number of y1>" x2="<number of x2>" y2="<number of y2>" ... x_n="<number of x_n>" y_n="<number of y_n>" alt={label}>{label}</points>. Total number of {label} is **<count>**.'
        return f"Locate all {label} and count the total number of {label}. Response Example : {response_example}"

    def validate_counting(self, model, state, device):
        if self.accelerator.is_main_process:
            print(f"\n[Step {state.global_step}] Running Validation 1: Counting Accuracy...")
        
        eval_limit = 100 
        
        try:
            val_dataset_stream = load_dataset("Jiwon-Kang/pixmo-count-filtered-imgContained", split="validation", streaming=True)
            val_dataset = list(val_dataset_stream.take(eval_limit))
        except Exception as e:
            if self.accelerator.is_main_process:
                print(f"[WARN] Failed to load counting validation dataset: {e}")
            return
        
        shard_size = len(val_dataset) // self.accelerator.num_processes
        start_idx = self.accelerator.process_index * shard_size
        end_idx = start_idx + shard_size if self.accelerator.process_index != self.accelerator.num_processes - 1 else len(val_dataset)
        my_shard = val_dataset[start_idx:end_idx]

        local_correct = 0
        local_total = 0
        local_details = []

        with torch.no_grad():
            for item in tqdm(my_shard, desc="Val: Counting", disable=not self.accelerator.is_main_process):
                image = item.get('image')
                if image is None: continue
                gt_count = item.get('count')
                if gt_count is None: continue
                
                label = item.get('label', '<object>')
                question_prompt = self.get_qa_prompt(label)
                
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{question_prompt}",
                        "images": [image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                
                prepare_inputs = self.processor(
                    conversations=conversation, images=[image], force_batchify=True
                ).to(device)
                
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    max_new_tokens=20,
                    do_sample=False,
                    use_cache=True
                )
                
                answer = self.processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                pred_count = extract_number(answer)
                
                gt_count_val = int(gt_count)
                pred_count_val = int(pred_count)
                is_correct = bool(pred_count_val == gt_count_val)
                
                if is_correct:
                    local_correct += 1
                local_total += 1
                
                res_str = f"GT: {gt_count_val} | Pred: {pred_count_val} | Correct: {is_correct} | Ans: {answer}"
                local_details.append(res_str)
                
                # Debug Logging to Terminal
                if self.accelerator.is_main_process and len(local_details) <= 3:
                    print(f"\n[Counting Rank 0 Debug] Q: {question_prompt[:50]}...")
                    print(f"Ans: {answer} | Parsed: {pred_count_val} | GT: {gt_count_val}")

        # Gather results
        all_correct_sum = self.accelerator.gather(torch.tensor([local_correct], device=device)).sum().item()
        all_total_sum = self.accelerator.gather(torch.tensor([local_total], device=device)).sum().item()
        
        # gather_object returns list of objects from all ranks
        all_details_list = gather_object(local_details)
        # If input was list, gather_object returns list of lists
        if len(all_details_list) > 0 and isinstance(all_details_list[0], list):
            flattened_details = [item for sublist in all_details_list for item in sublist]
        else:
            flattened_details = all_details_list

        if self.accelerator.is_main_process:
            accuracy = all_correct_sum / all_total_sum if all_total_sum > 0 else 0
            print(f"Validation Accuracy (Counting): {accuracy:.4f} ({all_correct_sum}/{all_total_sum})")
            
            if wandb.run is not None:
                if self.val_count_table is None:
                    self.val_count_table = wandb.Table(columns=["Step", "Accuracy", "Details"], log_mode='MUTABLE')
                
                all_details_str = "\n".join(flattened_details)
                self.val_count_table.add_data(state.global_step, accuracy, all_details_str)
                wandb.log({
                    "val/accuracy": accuracy, 
                    "val/counting_predictions": self.val_count_table
                })

    def validate_points(self, model, state, device):
        if self.accelerator.is_main_process:
            print(f"\n[Step {state.global_step}] Running Validation 2: Points Qualitative...")
        eval_limit = 100
        
        if os.path.exists(self.args.data_path):
            try:
                full_dataset = load_from_disk(self.args.data_path)
                val_dataset = full_dataset.train_test_split(test_size=1000, seed=42)['test']
                val_dataset = val_dataset.select(range(min(len(val_dataset), eval_limit)))
                val_list = [item for item in val_dataset]
            except Exception as e:
                if self.accelerator.is_main_process: print(f"[WARN] Failed to load local dataset: {e}")
                return
        else:
            try:
                val_dataset = load_dataset(self.args.data_path, split="validation", streaming=True).take(eval_limit)
                val_list = list(val_dataset)
            except Exception:
                val_dataset = load_dataset(self.args.data_path, split="train", streaming=True).take(eval_limit)
                val_list = list(val_dataset)
        
        shard_size = len(val_list) // self.accelerator.num_processes
        start_idx = self.accelerator.process_index * shard_size
        end_idx = start_idx + shard_size if self.accelerator.process_index != self.accelerator.num_processes - 1 else len(val_list)
        my_shard = val_list[start_idx:end_idx]

        local_results = []
        
        # Metrics aggregators
        local_mean_coord_dev_sum = 0.0
        local_accuracy_sum = 0.0
        local_count_dev_sum = 0.0
        local_sample_count = 0

        with torch.no_grad():
            for item in tqdm(my_shard, desc="Val: Points", disable=not self.accelerator.is_main_process):
                image = item.get('image')
                answer_gt = item.get('answer')
                label = item.get('label', 'object')
                gt_points = item.get('points', [])
                
                question = self.get_qa_prompt(label)
                
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{question}",
                        "images": [image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                
                prepare_inputs = self.processor(
                    conversations=conversation, images=[image], force_batchify=True
                ).to(device)
                
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True
                )
                
                generated_text = self.processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                
                # Calculate Metrics
                m_dev, m_acc, c_dev = calculate_metrics(generated_text, gt_points)
                local_mean_coord_dev_sum += m_dev
                local_accuracy_sum += m_acc
                local_count_dev_sum += c_dev
                local_sample_count += 1

                # Debug Logging
                if self.accelerator.is_main_process and local_sample_count <= 3:
                    print(f"\n[Points Rank 0 Debug] Q: {question[:50]}...")
                    print(f"Gen: {generated_text[:100]}...")
                    print(f"Metrics: Dev={m_dev:.2f}, Acc={m_acc:.2f}, C_Dev={c_dev}")

                local_results.append({
                    "step": state.global_step,
                    "question": question,
                    "generated": generated_text,
                    "gt": answer_gt,
                    "image": image,
                    "metrics": (m_dev, m_acc, c_dev)
                })

        # Gather Metrics
        total_samples = self.accelerator.gather(torch.tensor([local_sample_count], device=device)).sum().item()
        total_mean_coord_dev = self.accelerator.gather(torch.tensor([local_mean_coord_dev_sum], device=device)).sum().item()
        total_accuracy = self.accelerator.gather(torch.tensor([local_accuracy_sum], device=device)).sum().item()
        total_count_dev = self.accelerator.gather(torch.tensor([local_count_dev_sum], device=device)).sum().item()
        
        # Calculate Averages
        avg_mean_coord_dev = total_mean_coord_dev / total_samples if total_samples > 0 else 0
        avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
        avg_count_dev = total_count_dev / total_samples if total_samples > 0 else 0

        # Gather Results for Table
        all_results_gathered = gather_object(local_results)
        if len(all_results_gathered) > 0 and isinstance(all_results_gathered[0], list):
            all_results = [item for sublist in all_results_gathered for item in sublist]
        else:
            all_results = all_results_gathered

        if self.accelerator.is_main_process:
            print(f"Points Validation Results: Mean Dev={avg_mean_coord_dev:.2f}, Accuracy={avg_accuracy:.2f}, Count Dev={avg_count_dev:.2f}")
            
            if self.val_table is None and wandb.run is not None:
                self.val_table = wandb.Table(columns=["Step", "Question", "Generated", "GT", "Image", "Metrics (Dev, Acc, CDev)"], log_mode='MUTABLE')

            for i, res in enumerate(all_results):
                if wandb.run is not None:
                    try:
                        self.val_table.add_data(
                            res['step'], 
                            res['question'], 
                            res['generated'], 
                            res['gt'], 
                            wandb.Image(res['image']),
                            str(res['metrics'])
                        )
                    except Exception as e:
                        print(f"Error adding to wandb: {e}")
            
            if wandb.run is not None:
                wandb.log({
                    "val/qualitative_points": self.val_table,
                    "val/points_mean_coord_dev": avg_mean_coord_dev,
                    "val/points_accuracy": avg_accuracy,
                    "val/points_mean_count_dev": avg_count_dev
                })

def main():
    parser = argparse.ArgumentParser(description="Train Janus model for pointing task")
    parser.add_argument("--tuning_mode", type=str, default="transformer_ONLY", help="Tuning mode")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B", help="Path to pretrained model")
    parser.add_argument("--data_path", type=str, default="allenai/pixmo-points", help="Dataset path (Hub or Local)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/janus-points", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64, help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", type=int, default=1, help="Enable gradient checkpointing")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every n steps")
    parser.add_argument("--log_freq", type=int, default=100, help="Log validation every n steps")
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = "janus-points-finetune"
    accelerator = Accelerator()
    
    # Load Dataset Logic (Local vs Hub)
    if os.path.exists(args.data_path):
        if accelerator.is_main_process: print(f"Loading dataset from disk: {args.data_path}")
        full_dataset = load_from_disk(args.data_path)
        print(f"Splitting dataset into train/validation... (Rank {accelerator.process_index})")
        split_ds = full_dataset.train_test_split(test_size=1000, seed=42)
        train_dataset = split_ds['train']
    else:
        if accelerator.is_main_process: print(f"Loading dataset from Hub: {args.data_path}")
        train_dataset = load_dataset(args.data_path, split="train", streaming=True)
    
    if accelerator.is_main_process: print(f"Loading model from {args.model_path}...")
    processor = VLChatProcessor.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    model = EnhancedMultiModalModel.from_pretrained(
        args.model_path,
        language_config=language_config,
        trust_remote_code=True,
        torch_dtype=torch_dtype
    )
    
    if args.tuning_mode == "transformer_lora":
        if accelerator.is_main_process: print("[INFO] Transformer LoRA tuning...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        model.language_model.print_trainable_parameters()

    elif args.tuning_mode == "transformer_ONLY":
        if accelerator.is_main_process: print("[INFO] Transformer ONLY full fine-tuning...")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.language_model.model.parameters():
            param.requires_grad = True
        
    elif args.tuning_mode == "full":
        if accelerator.is_main_process: print("[INFO] Full model fine-tuning...")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.language_model.parameters():
            param.requires_grad = True
        for param in model.vision_model.parameters():
            param.requires_grad = True
        for param in model.aligner.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported tuning mode: {args.tuning_mode}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="wandb",
        remove_unused_columns=False,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        ddp_find_unused_parameters=False if args.gradient_checkpointing else None,
        dataloader_num_workers=args.num_workers, 
        max_steps=10000, 
        accelerator_config = {
            "dispatch_batches": False
        }
    )
    
    def data_collator(batch):
        return collate_fn(batch, processor)
    
    val_callback = ValidationCallback(processor, args, log_freq=args.log_freq) 

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[val_callback]
    )
    val_callback.trainer = trainer
    
    if accelerator.is_main_process: print("Starting training...")
    trainer.train()
    
    final_output_dir = os.path.join(args.output_dir, "final_model")
    if accelerator.is_main_process:
        os.makedirs(final_output_dir, exist_ok=True)
        print(f"Saving model to {final_output_dir}")
        if args.tuning_mode == "lora":
            model.language_model.save_pretrained(final_output_dir)
        else:
            trainer.save_model(final_output_dir)
        processor.save_pretrained(final_output_dir)

if __name__ == "__main__":
    main()