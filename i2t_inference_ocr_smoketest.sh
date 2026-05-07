python inference/i2t_inference_ocr.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output2/ocr_i2t_baseline \
    --dataset_path Jiwon-Kang/OCR-Synthetic-Rendered-200K \
    --num_samples 5 \
    --save-attention-maps \
    --manual-attention \
    --attention-save-step-interval 8 \
    "$@"

python inference/i2t_inference_ocr.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --output_dir output2/ocr_i2t_baseline \
    --dataset_path Jiwon-Kang/OCR-Synthetic-Rendered-200K \
    --num_samples 5 \
    --save-attention-maps \
    --manual-attention \
    --attention-save-step-interval 8 \
    --lora_ckpt_path /data/mm-llm-backbone_890/personal/sirius/audio_ablation/lumina_checkpoints/lumina_lora128_ocr_synthetic_wohead_gen_epoch0_iter16799_step525 \
    "$@"
