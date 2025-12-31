# multimodal_trainer.py

import os
from typing import List, Tuple, Dict, Any
import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW, SGD
from transformers import Trainer, TrainingArguments, get_scheduler
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images


class EnhancedMultiModalModel(MultiModalityCausalLM):
    """
    향상된 멀티모달 인과 언어 모델로, 사용자 정의 순전파(forward) 로직을 지원합니다.
    """

    def forward(
        self,
        input_ids:torch.LongTensor,
        attention_mask=None,
        pixel_values=None,
        image_token_masks=None,
        labels=None,
        **kwargs,
    ):
        """
        사용자 정의 순전파(forward) 메서드로, 이미지와 텍스트의 공동 처리를 지원합니다.
        """
        if pixel_values is not None and image_token_masks is not None:
            inputs_embeds = self._process_multimodal_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_masks=image_token_masks,
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        kwargs.pop("inputs_embeds", None)
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
        bs, n = pixel_values.shape[0:2] # (bs, n, 3, h, w)
        images = pixel_values.view(bs * n, *pixel_values.shape[2:]) # (bs * n, 3, h, w)
        image_features = self.vision_model(images) 
        aligned_features = self.aligner(image_features) # (bs * n, n_image_tokens, D)

        aligned_features = aligned_features.view(bs, n, *aligned_features.shape[1:]) # (bs, n, n_image_tokens, D)
        aligned_features = aligned_features.flatten(1, 2) # (bs, n * n_image_tokens, D)

        text_embeds = self.language_model.get_input_embeddings()(input_ids)

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


class EnhancedMultiModalTrainer:
    def __init__(self,
                 data_dir: str,
                 pretrained_model_path: str,
                 output_dir: str,
                 batch_size: int = 1,
                 max_epochs: int = 10,
                 lr: float = 3e-4,
                 user_question: str = "这张照片讲了什么？", # “이 사진은 무엇을 말하고 있나요?”
                 optimizer_name: str = "AdamW",
                 lora_config: dict|None = None,
                 training_args: dict|None = None):
        """
        향상된 멀티모달 트레이너 초기화.
        """
        self.data_dir = data_dir
        self.pretrained_model_path = pretrained_model_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.user_question = user_question
        self.optimizer_name = optimizer_name
        self.lora_config = lora_config if lora_config else {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        self.training_args = training_args if training_args else {
            "fp16": True,
            "max_grad_norm": 1.0,
            "save_strategy": "epoch",
            "evaluation_strategy": "no",
            "logging_steps": 50,
            "save_total_limit": 2,
            "remove_unused_columns": False,
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_data(self) -> List[Tuple[str, str]]:
        """이미지와 텍스트 쌍 데이터 로드."""
        pairs = []
        img_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        for root, _, files in os.walk(self.data_dir):
            for fname in files:
                base, ext = os.path.splitext(fname)
                full_path = os.path.join(root, fname)
                if ext.lower() in img_exts: # 만약 이미지 파일이라면
                    txt_path = os.path.join(root, f"{base}.txt") # 동일한 이름의 .txt 파일 찾기
                    if os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8") as f:
                            text_content = f.read().strip()
                        pairs.append((full_path, text_content))
        if not pairs:
            raise ValueError(f"No matching image-text pairs found in {self.data_dir}.")
        print(f"Loaded {len(pairs)} image-text pairs from {self.data_dir}")
        return pairs

    def _prepare_model(self):
        """사전 훈련된 모델 로드 및 LoRA 미세 조정 구성."""
        print(f"Loading pretrained model from {self.pretrained_model_path}")
        self.processor = VLChatProcessor.from_pretrained(
            self.pretrained_model_path,
            slow_image_processor_class="AutoImageProcessor"
        )
        self.model = EnhancedMultiModalModel.from_pretrained(
            self.pretrained_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # LoRA 구성
        lora_config = LoraConfig(**self.lora_config)
        self.model = get_peft_model(self.model, lora_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable_params} / {total_params}")

    def _prepare_optimizer_and_scheduler(self, dataset_size: int):
        """옵티마이저 및 학습률 스케줄러 준비."""
        if self.optimizer_name == "AdamW":
            optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        num_training_steps = dataset_size * self.max_epochs // self.batch_size
        num_warmup_steps = int(0.1 * num_training_steps)

        scheduler_name = self.training_args.get("lr_scheduler_type", "cosine")
        if scheduler_name not in ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]:
            raise ValueError(f"Unsupported learning rate scheduler: {scheduler_name}")

        lr_scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """사용자 정의 데이터 수집 함수."""
        conversations = [] # List[Dict[str, Any]]
        for item in batch:
            conversations.extend(self._generate_conversation(item["image_path"], item["text"]))
        
        pil_images_list = load_pil_images(conversations)
        encoded = self.processor(
            conversations=conversations,
            images=pil_images_list,
            return_tensors="pt",
            force_batchify=True,
        )
        encoded["labels"] = encoded["input_ids"].clone()

        image_placeholder_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image_placeholder>")
        input_ids = encoded["input_ids"]
        image_token_masks = (input_ids == image_placeholder_token_id)

        if not image_token_masks.any():
            raise ValueError("No <image_placeholder> tokens found in the input!")

        encoded["image_token_masks"] = image_token_masks
        return dict(encoded)

    def _generate_conversation(self, image_path: str, assistant_text: str) -> List[Dict[str, Any]]:  
        """대화 템플릿 생성."""  
        # assistant_text가 EOS 토큰으로 끝나는지 확인  
        eos_token = self.processor.tokenizer.eos_token  
        if not assistant_text.endswith(eos_token):  
            assistant_text = assistant_text + eos_token  
          
        return [  
            {"role": "<|User|>", "content": f"<image_placeholder>\n{self.user_question}", "images": [image_path]},
            {"role": "<|Assistant|>", "content": assistant_text},
        ]

    def train(self):
        """메인 훈련 프로세스."""
        pairs = self._load_data()
        dataset = [{"image_path": img, "text": txt} for img, txt in pairs]

        self._prepare_model()
        self._prepare_optimizer_and_scheduler(len(dataset))

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.max_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.lr,
            **self.training_args, 
        )

        trainer = Trainer(
            model=self.model, 
            args=training_args,
            train_dataset=dataset,
            data_collator=self._collate_fn,
            optimizers=(self.optimizer, self.lr_scheduler),
        )

        print("[Start!] Start training!!!!!---------->>>>>>>")
        trainer.train()

        print("[on Progress] Merging LoRA weights...")
        self.model = self.model.merge_and_unload()

        print("[on Progress] Saving...")
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        print(f"[Done!] Fine-tuned model have saved to {self.output_dir}")