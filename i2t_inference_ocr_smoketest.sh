python inference/i2t_inference_ocr.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output2/ocr_i2t_smoketest \
    --dataset_path Jiwon-Kang/OCR-Synthetic-Rendered-200K \
    --num_samples 20 \
    --no-save-attention-maps \
    --manual-attention \
    "$@"
