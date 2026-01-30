OUTPUT_DIR="./evaluation_results/lora128_spatial_relation_wohead_und-20260126-144357/cvbench/step3500"
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=0 python evaluate_cvbench_spatial.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir "$OUTPUT_DIR" \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 \
    --lora_ckpt_path /home/work/.project/heeji/VLM/output/lora128_spatial_relation_wohead_und-20260126-144357/epoch2-iter20863-step3500 \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

OUTPUT_DIR="./evaluation_results/lora128_spatial_relation_wohead_und-20260126-144357/cvbench/step4000"
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=0 python evaluate_cvbench_spatial.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir "$OUTPUT_DIR" \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 \
    --lora_ckpt_path /home/work/.project/heeji/VLM/output/lora128_spatial_relation_wohead_und-20260126-144357/epoch2-iter36863-step4000 \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

OUTPUT_DIR="./evaluation_results/lora128_spatial_relation_wohead_und-20260126-144357/cvbench/step4500"
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=0 python evaluate_cvbench_spatial.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir "$OUTPUT_DIR" \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 \
    --lora_ckpt_path /home/work/.project/heeji/VLM/output/lora128_spatial_relation_wohead_und-20260126-144357/epoch3-iter7295-step4500 \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

OUTPUT_DIR="./evaluation_results/lora128_spatial_relation_wohead_und-20260126-144357/cvbench/step5000"
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=0 python evaluate_cvbench_spatial.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir "$OUTPUT_DIR" \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 \
    --lora_ckpt_path /home/work/.project/heeji/VLM/output/lora128_spatial_relation_wohead_und-20260126-144357/epoch3-iter23295-step5000 \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

OUTPUT_DIR="./evaluation_results/lora128_spatial_relation_wohead_und-20260126-144357/cvbench/step5500"
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=0 python evaluate_cvbench_spatial.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir "$OUTPUT_DIR" \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 \
    --lora_ckpt_path /home/work/.project/heeji/VLM/output/lora128_spatial_relation_wohead_und-20260126-144357/epoch3-iter39295-step5500 \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

OUTPUT_DIR="./evaluation_results/lora128_spatial_relation_wohead_und-20260126-144357/cvbench/step6000"
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=0 python evaluate_cvbench_spatial.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir "$OUTPUT_DIR" \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 \
    --lora_ckpt_path /home/work/.project/heeji/VLM/output/lora128_spatial_relation_wohead_und-20260126-144357/epoch4-iter9727-step6000 \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

OUTPUT_DIR="./evaluation_results/lora128_spatial_relation_wohead_und-20260126-144357/cvbench/step6500"
mkdir -p "$OUTPUT_DIR"
CUDA_VISIBLE_DEVICES=0 python evaluate_cvbench_spatial.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir "$OUTPUT_DIR" \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 \
    --lora_ckpt_path /home/work/.project/heeji/VLM/output/lora128_spatial_relation_wohead_und-20260126-144357/epoch4-iter25727-step6500 \
    2>&1 | tee "$OUTPUT_DIR/eval.log"