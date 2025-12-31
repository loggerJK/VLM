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


def extract_number(text):
    """모델 출력에서 **숫자** 패턴 추출"""
    match = re.search(r"\*\*(\d+)\*\*", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    return -1

class EnhancedMultiModalModel(MultiModalityCausalLM):
    """
    향상된 멀티모달 인과 언어 모델로, 사용자 정의 순전파(forward) 로직을 지원합니다.
    학습을 위해 MultiModalityCausalLM에 forward 메서드를 추가한 클래스입니다.
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
        이미지와 텍스트의联合 처리를 지원하는 forward 메서드.
        """
        if pixel_values is not None and image_token_masks is not None:
            # 멀티모달 입력 처리 (텍스트 임베딩 일부를 이미지 임베딩으로 교체)
            inputs_embeds = self._process_multimodal_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_masks=image_token_masks,
            )
        else:
            # 텍스트 전용 입력 처리
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 이미 inputs_embeds를 계산했으므로 input_ids는 제거 (중복 방지)
        kwargs.pop("inputs_embeds", None)
        
        # 실제 언어 모델 호출
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
        시각적 특징을 추출하고 정렬하여 텍스트 임베딩 시퀀스 내의 적절한 위치에 삽입합니다.
        """
        bs, n = pixel_values.shape[0:2] # (bs, n, 3, h, w)
        images = pixel_values.view(bs * n, *pixel_values.shape[2:]) # (bs * n, 3, h, w)
        
        # 비전 인코더 및 얼라이너 통과
        image_features = self.vision_model(images) 
        aligned_features = self.aligner(image_features) # (bs * n, n_image_tokens, D)

        # 배치 차원 재구성
        aligned_features = aligned_features.view(bs, n, *aligned_features.shape[1:]) # (bs, n, n_image_tokens, D)
        aligned_features = aligned_features.flatten(1, 2) # (bs, n * n_image_tokens, D)

        # 기본 텍스트 임베딩 생성
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 마스크를 사용하여 이미지 토큰 자리에 정렬된 이미지 특징 삽입
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

class StreamingDatasetWrapper(torch.utils.data.IterableDataset):
    """
    URL만 있는 데이터셋을 받아 실시간으로 이미지를 다운로드하여 제공하는 Wrapper.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_samples = len(dataset)
        tmp_sample = self.dataset[0]
        if 'image' in tmp_sample.keys():
            print("[INFO] Dataset already contains 'image' column. Using existing images.")
        else:
            print("[INFO] Dataset does not contain 'image' column. Images will be downloaded on-the-fly.")

    def fetch_image(self, url):
        if not url: return None
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                return img
        except Exception:
            pass
        return None

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            indices = range(self.num_samples)
        else:
            per_worker = int(np.ceil(self.num_samples / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, self.num_samples)
            indices = range(start, end)

        for idx in indices:
            item = self.dataset[idx]
            
            # 만약 'image' column이 없다면
            if 'image' in item.keys():
                image = item['image']
            else :
                image = self.fetch_image(item.get('image_url'))
                
                while image is None:
                    random_idx = np.random.randint(0, self.num_samples)
                    item = self.dataset[random_idx]
                    image = self.fetch_image(item.get('image_url'))
                
            image = image.convert("RGB")
            # image.save(f"./debug.png")  # 디버깅용 저장
            
            output_item = item.copy()
            output_item['image'] = image
            yield output_item

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
    
    # Assistant 역할 토큰 ID를 가져옵니다.
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|Assistant|>")

    for i in range(input_ids.shape[0]):
        # Assistant 토큰의 위치를 찾습니다.
        assistant_indices = (input_ids[i] == assistant_token_id).nonzero(as_tuple=True)[0]
        
        if assistant_indices.numel() > 0:
            # 첫 번째 등장 위치를 어시스턴트 턴의 시작으로 간주합니다.
            split_point = assistant_indices[0]
            
            # 어시스턴트 토큰을 포함하여 그 이전의 모든 레이블을 -100으로 마스킹합니다.
            # 모델이 어시스턴트 토큰 이후부터 답변을 생성하도록 유도합니다.
            labels[i, :split_point + 1] = -100

    # 이미지 토큰도 손실 계산에서 제외합니다.
    if "images_seq_mask" in inputs_dict:
        image_mask = inputs_dict["images_seq_mask"]
        # 이미지 마스크가 True인 위치의 레이블을 -100으로 설정합니다.
        labels[image_mask] = -100
        inputs_dict["image_token_masks"] = image_mask
    
    inputs_dict["labels"] = labels
    

    # 텐서가 아닌 필드 제거
    keys_to_remove = [k for k, v in inputs_dict.items() if not isinstance(v, torch.Tensor)]
    for k in keys_to_remove:
        del inputs_dict[k]

    return inputs_dict

class ValidationCallback(TrainerCallback):
    def __init__(self, processor, eval_dataset, args, log_freq=500):
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.log_freq = log_freq
        self.args = args
        self.trainer = None
        self.val_table = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        optimized_param_names = [n for n, p in self.trainer.model.named_parameters() if p.requires_grad]
        print(f"Starting training..., Number of Optimized parameters: {len(optimized_param_names)}")
        with open(os.path.join(args.output_dir, "optimized_param_names.txt"), "w") as f:
            for name in optimized_param_names:
                f.write(f"{name}\n")
        wandb.config.update({"optimized_param_names": optimized_param_names})    
        

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.args.save_steps == 0 and state.is_world_process_zero:
            checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if self.args.tuning_mode == "lora":
                if self.trainer.is_world_process_zero:
                    print(f"\n[Step {state.global_step}] Saving checkpoint to {checkpoint_dir}")
                    peft_model = self.trainer.model.module.language_model if hasattr(self.trainer.model, 'module') else self.trainer.model.language_model
                    peft_model.save_pretrained(checkpoint_dir)
                    self.processor.save_pretrained(checkpoint_dir)
            else:
                self.trainer.save_model(checkpoint_dir)
                if self.trainer.is_world_process_zero:
                    self.processor.save_pretrained(checkpoint_dir)
                    
        if state.global_step % self.log_freq == 0 and state.is_world_process_zero:
            self.validate(model, state)

    def validate(self, model, state):
        print(f"\n[Step {state.global_step}] Running Validation on CountBenchQA...")
        model.eval()
        correct = 0
        total = 0
        eval_limit = 100 # len(self.eval_dataset)
        
        val_dataset_stream = load_dataset("Jiwon-Kang/pixmo-count-filtered-imgContained", split="validation", streaming=True)
        eval_dataset = iter(val_dataset_stream)  
        
        if hasattr(model, "module"):
            unwrap_model = model.module
        else:
            unwrap_model = model
            
        device = next(unwrap_model.parameters()).device

        import wandb
        
        # 테이블 초기화 (최초 1회)
        if self.val_table is None and wandb.run is not None:
            self.val_table = wandb.Table(columns=["Step", "Accuracy", "Details"], log_mode='MUTABLE')

        details_buffer = []

        with torch.no_grad():
            count = 0
            # len() 호출 없이 순차적으로 접근
            progress  = tqdm(range(eval_limit), desc="Validation", unit="sample")
            for item in eval_dataset:
                if count >= eval_limit:
                    break
                
                
                image = item.get('image')
                question = item.get('question')
                if image is None: continue
                gt_count = item.get('count')
                if gt_count is None: continue
                
                label = item.get('label', '<object>')

                question_prompt = f"{question}? Response Example : There are **<number>** of {label} in the image."
                
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{question_prompt}",
                        "images": [image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                
                pil_images = [image]
                prepare_inputs = self.processor(
                    conversations=conversation, images=pil_images, force_batchify=True
                ).to(device)
                
                inputs_embeds = unwrap_model.prepare_inputs_embeds(**prepare_inputs)
                
                outputs = unwrap_model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    max_new_tokens=20,
                    do_sample=False,
                    use_cache=True
                )
                
                answer = self.processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                print("=========================================")
                print(f"answer : {answer}, gt: {gt_count}")
                pred_count = extract_number(answer)
                
                # 명시적 타입 변환
                gt_count_val = int(gt_count)
                pred_count_val = int(pred_count)
                is_correct = bool(pred_count_val == gt_count_val)
                
                if is_correct:
                    correct += 1
                total += 1
                count += 1
                
                # 결과 문자열 포맷팅
                res_str = f"[{count}] GT: {gt_count_val} | Pred: {pred_count_val} | Correct: {is_correct} | Ans: {answer}"
                details_buffer.append(res_str)
                
                progress.update(1)
            progress.close()

        
        accuracy = correct / total if total > 0 else 0
        print(f"Validation Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        if wandb.run is not None and self.val_table is not None:
            # 모든 상세 내용을 하나의 문자열로 합침
            all_details = "\n".join(details_buffer)
            
            # 하나의 Row 추가 (Step, Accuracy, Details)
            self.val_table.add_data(state.global_step, accuracy, all_details)
            
            wandb.log({
                "val/accuracy": accuracy, 
                "global_step": state.global_step,
                "val/predictions": self.val_table
            })
        
        model.train()

def main():
    parser = argparse.ArgumentParser(description="Train Janus model for counting task")
    parser.add_argument("--tuning_mode", type=str, default="lora", help="Tuning mode: 'full' or 'lora'")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/Janus-Pro-7B", help="Path to pretrained model")
    parser.add_argument("--data_path", type=str, default="./data/pixmo_counting_filtered", help="Path to prepared dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/janus-counting", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", type=int, default=1, help="Enable gradient checkpointing (1=True, 0=False)")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of dataloader workers")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every n steps")
    parser.add_argument("--log_freq", type=int, default=500, help="Log validation every n steps")
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = "janus-counting-finetune"
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found at {args.data_path}. Please run prepare_dataset.py first!")
    
    raw_dataset = load_from_disk(args.data_path)
    train_dataset = StreamingDatasetWrapper(raw_dataset)
    
    print(f"Loading model from {args.model_path}...")
    processor = VLChatProcessor.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'
    
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    # EnhancedMultiModalModel 클래스 사용 (forward 구현 포함)
    model = EnhancedMultiModalModel.from_pretrained(
        args.model_path,
        language_config=language_config,
        trust_remote_code=True,
        torch_dtype=torch_dtype
    )
    
    if args.tuning_mode == "transformer_lora":
        print("[INFO] Transformer LoRA tuning...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        # model = get_peft_model(model, lora_config)
        model.language_model = get_peft_model(model.language_model, lora_config)
        model.language_model.print_trainable_parameters()
    elif args.tuning_mode == "transformer":
        print("[INFO] Transformer fine-tuning...")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.language_model.parameters():
            param.requires_grad = True
    elif args.tuning_mode == "transformer_ONLY":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.language_model.model.parameters():
            param.requires_grad = True
        
    elif args.tuning_mode == "full":
        print("[INFO] Full model fine-tuning...")
        # 모든 파라미터가 기본적으로 requires_grad=True이므로 별도 설정 불필요
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


    
    
    if args.gradient_checkpointing:
        print("Enabling Gradient Checkpointing...")
        model.gradient_checkpointing_enable()
        
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="no",
        report_to="wandb",
        remove_unused_columns=False,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        ddp_find_unused_parameters=False if args.gradient_checkpointing else None,
        dataloader_num_workers=args.num_workers, 
        # split_batches=True,
        # dispatch_batches=False
        accelerator_config = {
            # "split_batches": True,
            "dispatch_batches": False
        }
    )
    
    def data_collator(batch):
        return collate_fn(batch, processor)
    
    val_callback = ValidationCallback(processor, None, args, log_freq=args.log_freq) 

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[val_callback]
    )
    val_callback.trainer = trainer
    
    print("Starting training...")
    trainer.train()
    
    final_output_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Saving model to {final_output_dir}")
    
    if args.tuning_mode == "lora":
        # Save only the LoRA adapter
        model.language_model.save_pretrained(final_output_dir)
    else:
        # Save the full model
        trainer.save_model(final_output_dir)
        
        
    processor.save_pretrained(final_output_dir)

if __name__ == "__main__":
    main()
