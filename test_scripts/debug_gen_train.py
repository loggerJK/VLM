"""
Generation 모드 디버깅 스크립트.
train_counting.py의 generation forward를 최소 단위로 테스트.
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F

def encode_image_to_vq_tokens(vq_model, image, img_size=384):
    """PIL Image -> VQ token IDs (LongTensor)"""
    image = image.resize((img_size, img_size))
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor * 2 - 1
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    with torch.no_grad():
        _, _, (_, _, indices) = vq_model.encode(image_tensor.to(device=device, dtype=dtype))
    return indices.view(-1)


def main():
    device = "cuda:0"
    model_path = "deepseek-ai/Janus-Pro-7B"
    img_size = 384
    num_vq_tokens = (img_size // 16) ** 2  # 576 for 384

    print(f"=== Generation Debug ===")
    print(f"img_size={img_size}, num_vq_tokens={num_vq_tokens}")

    # Load model
    print("Loading model...")
    processor = VLChatProcessor.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'

    model = MultiModalityCausalLM.from_pretrained(
        model_path,
        language_config=language_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    print(f"Model loaded. Device: {device}")

    # Create dummy test image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

    # === Step 1: VQ encode ===
    print("\n[Step 1] VQ Encoding...")
    vq_model = model.gen_vision_model
    vq_tokens = encode_image_to_vq_tokens(vq_model, dummy_image, img_size=img_size)
    print(f"  vq_tokens.shape: {vq_tokens.shape}, range: [{vq_tokens.min()}, {vq_tokens.max()}]")

    # === Step 2: Build generation input sequence ===
    print("\n[Step 2] Building input sequence...")
    boi_id = processor.image_start_id
    eoi_id = processor.image_end_id
    print(f"  boi_id={boi_id}, eoi_id={eoi_id}")

    text = "A cute cat sitting on a windowsill."
    conversation = [
        {"role": "<|User|>", "content": text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    text_ids = processor.tokenizer.encode(sft_format)
    print(f"  text_ids length: {len(text_ids)}")
    print(f"  sft_format: {sft_format[:100]}...")

    full_ids = text_ids + [boi_id] + vq_tokens.tolist() + [eoi_id]
    full_ids = torch.LongTensor(full_ids).unsqueeze(0).to(device)
    print(f"  full_ids.shape: {full_ids.shape}")

    # Labels: mask prompt, only predict image tokens + eoi
    labels = torch.full_like(full_ids, -100)
    img_start = len(text_ids) + 1  # after boi
    labels[0, img_start:] = full_ids[0, img_start:]
    print(f"  labels non-masked range: [{img_start}, {full_ids.shape[1]})")

    # Gen token mask
    gen_mask = torch.zeros(full_ids.shape, dtype=torch.bool, device=device)
    gen_mask[0, img_start:img_start + num_vq_tokens] = True
    print(f"  gen_mask True count: {gen_mask.sum().item()}")

    attention_mask = torch.ones_like(full_ids)

    # === Step 3: Test _forward_generation logic ===
    print("\n[Step 3] Testing generation forward...")
    model.eval()

    with torch.no_grad():
        # 3a) Base text embedding
        inputs_embeds = model.language_model.get_input_embeddings()(full_ids)
        print(f"  inputs_embeds.shape: {inputs_embeds.shape}")

        # 3b) Replace gen token positions with gen embeddings (teacher forcing)
        image_token_ids = full_ids[gen_mask]
        print(f"  image_token_ids.shape: {image_token_ids.shape}")
        print(f"  image_token_ids range: [{image_token_ids.min()}, {image_token_ids.max()}]")

        image_embeds = model.prepare_gen_img_embeds(image_token_ids)
        print(f"  image_embeds.shape: {image_embeds.shape}")
        print(f"  image_embeds has NaN: {torch.isnan(image_embeds).any()}")

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[gen_mask] = image_embeds.to(inputs_embeds.dtype)

        # 3c) LLM forward (hidden states only)
        lm_model = model.language_model.model if hasattr(model.language_model, 'model') else model.language_model.base_model
        print(f"  lm_model type: {type(lm_model).__name__}")

        outputs = lm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        print(f"  hidden_states.shape: {hidden_states.shape}")
        print(f"  hidden_states has NaN: {torch.isnan(hidden_states).any()}")

        # 3d) gen_head -> logits
        gen_logits = model.gen_head(hidden_states)
        print(f"  gen_logits.shape: {gen_logits.shape}")
        print(f"  gen_logits has NaN: {torch.isnan(gen_logits).any()}")

        # 3e) Shifted cross-entropy loss
        shift_logits = gen_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        print(f"\n  Loss: {loss.item():.4f}")
        print(f"  Loss is NaN: {torch.isnan(loss).any()}")

    # === Step 4: Test backward (gradient flow) ===
    print("\n[Step 4] Testing backward pass...")
    model.train()

    # Enable grad for gen components only (simulate training config)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.gen_head.parameters():
        param.requires_grad = True
    for param in model.gen_embed.parameters():
        param.requires_grad = True
    for param in model.gen_aligner.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    # Forward with grad
    inputs_embeds = model.language_model.get_input_embeddings()(full_ids)
    image_token_ids = full_ids[gen_mask]
    image_embeds = model.prepare_gen_img_embeds(image_token_ids)
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[gen_mask] = image_embeds.to(inputs_embeds.dtype)

    outputs = lm_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )
    hidden_states = outputs.last_hidden_state
    gen_logits = model.gen_head(hidden_states)

    shift_logits = gen_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    print(f"  Loss (with grad): {loss.item():.4f}")

    loss.backward()

    # Check gradients
    for name in ['gen_head', 'gen_embed', 'gen_aligner']:
        module = getattr(model, name)
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if grads:
            grad_norm = torch.sqrt(sum(g.norm()**2 for g in grads))
            print(f"  {name} grad norm: {grad_norm.item():.6f}")
        else:
            print(f"  {name}: NO GRADIENTS!")

    print("\n=== All checks done ===")


if __name__ == "__main__":
    main()
