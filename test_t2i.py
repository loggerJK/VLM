import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from tqdm import tqdm
from diffusers.utils import make_image_grid


# specify the path to the model
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# t2i_prompt = "A dog sitting in the sidecar of a motorcycle."
# t2i_prompt = "A man flying through the air while riding a snowboard."
# t2i_prompt = "an image of plate with food on it"

t2i_prompt_list = [
#     "A dog sitting in the sidecar of a motorcycle.",
#     "A man flying through the air while riding a snowboard.",
#     "an image of plate with food on it",
#     "A young girl in a red dress is smiling.",
#     "A couple of elephants standing next to each other.",
    "A man is standing in a park with a ’Run for Rights’ banner in the background. He is wearing a white t-shirt with the number 28 on it, grey shorts, and grey socks with black shoes. The park is filled with people, some sitting on benches, and there is a bicycle leaning against a tree."
]

for idx, t2i_prompt in enumerate(t2i_prompt_list):

    save_dir = os.path.join("generated_samples", t2i_prompt.replace(" ", "_")[:20])
    os.makedirs(save_dir, exist_ok=True)

    conversation = [
        {
            "role": "<|User|>",
            "content": f"{t2i_prompt}",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag


    @torch.inference_mode()
    def generate(
        mmgpt: MultiModalityCausalLM,
        vl_chat_processor: VLChatProcessor,
        prompt: str,
        temperature: float = 1,
        parallel_size: int = 16,
        cfg_weight: float = 4,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
    ):
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = vl_chat_processor.pad_id

        inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in tqdm(range(image_token_num_per_image), desc="Generating images", total=image_token_num_per_image):
            outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            
            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)


        dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        os.makedirs(save_dir, exist_ok=True)
        for i in range(parallel_size):
            save_path = os.path.join(save_dir, "img_{}.jpg".format(i))
            PIL.Image.fromarray(visual_img[i]).save(save_path)
        
        grid = make_image_grid([PIL.Image.fromarray(visual_img[i]) for i in range(parallel_size)], rows=int(np.sqrt(parallel_size)), cols=int(np.sqrt(parallel_size)))
        grid.save(os.path.join(save_dir, "grid.jpg"))


    generate(
        vl_gpt,
        vl_chat_processor,
        prompt,
    )