import os
import hashlib
from PIL import Image
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

# 업로더의 원래 멀티모달 모델 폴더 주소입니다. 자신의 폴더 경로를 입력하세요.
model_path = "/root/autodl-tmp/deepseek-janus-pro-lora/Janus-Pro-7B"
config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'
vl_gpt = AutoModelForCausalLM.from_pretrained(
    model_path,
    language_config=language_config,
    trust_remote_code=True,
    torch_dtype=torch.float16  # BFloat16 문제를 방지하기 위해 float16 사용
)
if torch.cuda.is_available():
    vl_gpt = vl_gpt.cuda()

vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.inference_mode()
def multimodal_understanding(image, question, seed=42, top_p=0.95, temperature=0.1):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.float16)  # float16 사용
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

def generate_hash_name(filename):
    """
    파일에 대한 해시 코드 이름을 생성합니다.
    """
    hash_object = hashlib.md5(filename.encode())
    return hash_object.hexdigest()

def convert_images_to_jpeg_and_rename(folder_path):
    """
    폴더 내의 이미지를 JPEG 형식으로 변환하고 해시 코드 이름으로 이름을 변경한 다음, 마지막으로 1부터 시작하는 숫자로 통일하여 처리합니다.
    
    :param folder_path: 이미지 폴더 경로
    :return: 새 이미지 파일 경로 목록 반환
    """
    temp_image_paths = []
    new_image_paths = []
    count = 1  # 1부터 시작하여 이름 증가

    # 폴더 내의 모든 이미지 순회
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            
            # 이미지를 열고 RGB 모드로 변환
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                
                # 해시 코드 이름 생성
                hash_name = generate_hash_name(filename)
                temp_filename = f"{hash_name}.jpeg"
                temp_image_path = os.path.join(folder_path, temp_filename)
                
                # JPEG 형식으로 저장
                img.save(temp_image_path, "JPEG")
                
                # 새 파일이 저장되었는지 확인
                if os.path.exists(temp_image_path):
                    temp_image_paths.append(temp_image_path)
                    os.remove(image_path)  # 원본 파일 삭제
                    print(f"Converted {filename} -> {temp_filename}")
                else:
                    print(f"Failed to save {temp_filename}, skipping deletion of {filename}")
    
    # 파일을 1부터 시작하는 숫자로 이름 변경
    for temp_image_path in sorted(temp_image_paths):
        new_filename = f"{count}.jpeg"
        new_image_path = os.path.join(folder_path, new_filename)
        os.rename(temp_image_path, new_image_path)
        new_image_paths.append(new_image_path)
        print(f"Renamed {temp_image_path} -> {new_filename}")
        count += 1

    return new_image_paths

def process_images_in_folder(folder_path, output_folder, question="Describe this picture. Please note that the person in this picture named Trump."):
    """
    폴더 내의 모든 이미지를 처리하고 해당 인식 결과를 생성합니다.
    
    :param folder_path: 이미지 폴더 경로
    :param output_folder: 결과 출력 폴더 경로
    :param question: 각 이미지에 대한 질문
    """
    # 출력 폴더가 존재하는지 확인
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 이미지를 JPEG 형식으로 변환하고 이름 변경
    new_image_paths = convert_images_to_jpeg_and_rename(folder_path)
    
    # 새 이미지 경로 목록 출력
    print("New image paths:", new_image_paths)
    
    # 새 이미지 경로 순회
    for image_path in new_image_paths:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}, skipping...")
            continue
        
        filename = os.path.basename(image_path)
        output_txt_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
        
        # 이미지 로드
        image = np.array(Image.open(image_path).convert("RGB"))
        
        # 멀티모달 모델을 호출하여 결과 생성
        result = multimodal_understanding(image, question)
        
        # 결과 수정: "person"이 나타나면 뒤에 "called Trump" 추가
        # result = result.replace("人", "叫做川普的人")
        
        # 결과를 txt 파일로 저장
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(result)
        
        print(f"Processed {filename} -> {output_txt_path}")

# 입력 폴더 및 출력 폴더 설정
input_folder = "/root/autodl-tmp/deepseek-janus-pro-lora/trump"  # 이미지 폴더 경로로 교체
output_folder = "/root/autodl-tmp/deepseek-janus-pro-lora/trump"    # 결과 출력 폴더 경로로 교체

# 이미지 처리
# 用中文描述这张照片,照片里的人的名字叫川普: “이 사진을 중국어로 설명해 주세요. 사진 속 인물의 이름은 트럼프입니다.”
process_images_in_folder(input_folder, output_folder, question="用中文描述这张照片,照片里的人的名字叫川普")