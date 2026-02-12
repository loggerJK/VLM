
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

python inference/inference_mmu.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --prompt "Can you extract all visible text from the document here in reading order?" \
    --image_path examples/ocr2.png \
    --steps 1 \
    --gen_length 128 \
    --block_length 128 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output/outputs_text_understanding