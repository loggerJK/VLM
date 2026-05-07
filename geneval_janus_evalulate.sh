CUDA_VISIBLE_DEVICES=0 python geneval_janus_evalulate.py \
    "/mnt/data1/jiwon/geneval/prompts/evaluation_metadata.jsonl" \
    --model "/mnt/data1/jiwon/deepseek-janus-pro-lora/checkpoints/train[full]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-600" \
    --outdir "results/train[full]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-600"