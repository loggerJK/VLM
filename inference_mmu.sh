export CUDA_VISIBLE_DEVICES=7
python -m pdb inference/inference_mmu.py \
    --checkpoint "/mnt/data1/jiwon/Lumina-DiMOO/output/Lumina-DiMOO-counting-full/epoch3-iter38911-step5900" \
    --prompt "How many cars are there in the image? Response Example : There are **<number>** cars in the image.." \
    --image_path /mnt/data1/jiwon/Lumina-DiMOO/image.png \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir output/outputs_text_understanding