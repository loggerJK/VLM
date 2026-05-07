python inference/inference_t2i_ocr.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --height 1024 \
    --width 1024 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --seed 65513 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output/ocr_t2i \
    --prompt_files ./ocr_sentences.txt \
    --no-save-attention-maps --manual-attention 
    
python inference/inference_t2i_ocr.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --height 1024 \
    --width 1024 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --seed 65513 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output/ocr_t2i \
    --prompt_files ./ocr_sentences.txt \
    --no-save-attention-maps --manual-attention 