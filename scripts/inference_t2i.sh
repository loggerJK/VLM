
export CUDA_VISIBLE_DEVICES=7

# PROMPT=$(cat <<'EOF'
# Generate an image of a clean academic document page.

# IMPORTANT: The text content MUST be rendered EXACTLY as provided below,
# character-by-character, with no omissions, no paraphrasing, no rewording,
# no spelling changes, and no added or removed text.

# Render the following text verbatim:

# """
# Why Do We Need OCRBench v2?

# Limitations of Existing Benchmarks.
# Recent evaluations of LMMs’ OCR capabilities have made significant progress,
# yet most existing benchmarks exhibit limitations.
# """
# EOF
# )


python inference/inference_t2i_official.py\
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --prompt "a photo of beautiful scenery with mountains and a lake" \
    --height 1024 \
    --width 1024 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --seed 65513 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output/results_text_to_image