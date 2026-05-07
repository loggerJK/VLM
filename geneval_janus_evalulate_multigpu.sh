# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 geneval_janus_evalulate_multigpu.py \
#     "/mnt/data1/jiwon/geneval/prompts/evaluation_metadata.jsonl" \
#     --model "/mnt/data1/jiwon/deepseek-janus-pro-lora/checkpoints/train[full]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-600" \
#     --outdir "results/train[full]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-600"

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 geneval_janus_evalulate_multigpu.py \
#     "/mnt/data1/jiwon/geneval/prompts/evaluation_metadata.jsonl" \
#     --model "/mnt/data1/jiwon/deepseek-janus-pro-lora/checkpoints/train[transformer]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-600" \
#     --outdir "results/train[transformer]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-600"

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:29510 geneval_janus_evalulate_multigpu.py \
    "/mnt/data1/jiwon/geneval/prompts/evaluation_metadata.jsonl" \
    --model "/mnt/data1/jiwon/Janus/checkpoints/train[transformer_ONLY]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-4200" \
    --outdir "results/train[transformer_ONLY]_dset[pixmo-count]_ngpu1_bs1_accum128_lr4e-5_ep3_full/checkpoint-4200"