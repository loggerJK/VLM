# export CUDA_VISIBLE_DEVICES=7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MASTER_ADDR=localhost
export MASTER_PORT=25001
export CUDA_VISIBLE_DEVICES=4,5,6,7 # Set this according to your available
ngpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')
# base_dir="/mnt/cvlab22_data1"
base_dir="/mnt/data1"

# ---------------------------------------------------------------------------- #
#                        내부 체크포인트들 / Understanding Only                        #
# ---------------------------------------------------------------------------- #


# checkpoint_path="jiwon/Lumina-DiMOO/output"
# ckpt_list=(
#     "lora128_counting_wohead/epoch4"
#     "lora128_counting_wohead_resumeEpoch12_lr3e-6/epoch14"
# )

# for ckpt in "${ckpt_list[@]}"; do
#     output_dir_name=$(echo "$ckpt" | tr '/' '_')
#     echo "Running inference for checkpoint: $ckpt"
#     echo "Output directory name: $output_dir_name"
#     torchrun --nproc_per_node ${ngpus} inference/inference_t2i_multigpu.py\
#         --checkpoint Alpha-VLLM/Lumina-DiMOO \
#         --height 1024 \
#         --width 1024 \
#         --timesteps 64 \
#         --cfg_scale 4.0 \
#         --seed 65513 \
#         --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#         --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
#         --output_dir ${base_dir}/dvlm/counting/generation/geneval/lumina_${output_dir_name} \
#         --prompt_files ${base_dir}/jiwon/geneval/prompts/evaluation_metadata_count.jsonl
# done


# ---------------------------------------------------------------------------- #
#                                   외부 체크포인트들                                  #
# ---------------------------------------------------------------------------- #

# /mnt/data1/dvlm/checkpoints
checkpoint_path="dvlm/checkpoints"
ckpt_list=(
    #### Generation 
    # "lumina_train_generation_wohead_1024/epoch0"
    # "lumina_train_generation_wohead_1024/epoch1"
    # "lumina_train_generation_wohead_1024/epoch2"
    # "lumina_train_generation_wohead_1024/epoch3"
    # "lumina_train_generation_wohead_1024/epoch4"
    # "lumina_train_generation_wohead_1024/epoch5"
    # "lumina_train_generation_wohead_1024/epoch6"
    # "lumina_train_generation_wohead_1024/epoch7"
    # "lumina_train_generation_wohead_1024/epoch8"
    # "lumina_train_generation_wohead_1024/epoch9"
    # "lumina_train_generation_wohead_1024/epoch10"
    # "lumina_train_generation_wohead_1024/epoch11"
    # "lumina_train_generation_wohead_1024/epoch12"

    #### Generation + Understanding
    # "lora128_counting_wohead_both/epoch0"
    # "lora128_counting_wohead_both/epoch1"
    # "lora128_counting_wohead_both/epoch2"
    # "lora128_counting_wohead_both/epoch3"
    # "lora128_counting_wohead_both/epoch4"
    # "lora128_counting_wohead_both/epoch5"
    # "lora128_counting_wohead_both/epoch6"
    # "lora128_counting_wohead_both/epoch7"
    # "lora128_counting_wohead_both/epoch8"
    # "lora128_counting_wohead_both/epoch9"
    # "lora128_counting_wohead_both/epoch10"
    # "lora128_counting_wohead_both/epoch11"
    # "lora128_counting_wohead_both/epoch12"
    # "lora128_counting_wohead_both/epoch13"
    # "lora128_counting_wohead_both/epoch14"
    # "lora128_counting_wohead_both/epoch15"
    # "lora128_counting_wohead_both/epoch16"
    # "lora128_counting_wohead_both/epoch17"
    # "lora128_counting_wohead_both/epoch18"
    # "lora128_counting_wohead_both/epoch19"
    # "lora128_counting_wohead_both/epoch20"
    # "lora128_counting_wohead_both/epoch21"
    # "lora128_counting_wohead_both/epoch22"
    # "lora128_counting_wohead_both/epoch23"
    # "lora128_counting_wohead_both/epoch24"
    # "lora128_counting_wohead_both/epoch25"
    # "lora128_counting_wohead_both/epoch26"
    # "lora128_counting_wohead_both/epoch27"
    # "lora128_counting_wohead_both/epoch28"
    # "lora128_counting_wohead_both/epoch29"
    # "lora128_counting_wohead_both/epoch30"
    # "lora128_counting_wohead_both/epoch31"
    # "lora128_counting_wohead_both/epoch32"
    # "lora128_counting_wohead_both/epoch33"
    # "lora128_counting_wohead_both/epoch34"
    # "lora128_counting_wohead_both/epoch35"
    # "lora128_counting_wohead_both/epoch36"
)

for ckpt in "${ckpt_list[@]}"; do # 순방향 인퍼런스
    output_dir_name=$(echo "$ckpt" | tr '/' '_')
    echo "Running inference for checkpoint: $ckpt"
    echo "Output directory name: $output_dir_name"
    torchrun --nproc_per_node ${ngpus} inference/inference_t2i_multigpu.py\
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --height 1024 \
        --width 1024 \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --seed 65513 \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/counting/generation/geneval/lumina_${output_dir_name} \
        --prompt_files ${base_dir}/jiwon/geneval/prompts/evaluation_metadata_count.jsonl
done


# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

output_dir_name="BASELINE"
# echo "Running inference for checkpoint: $ckpt"
echo "Output directory name: $output_dir_name"
torchrun --nproc_per_node ${ngpus} inference/inference_t2i_multigpu.py\
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --height 1024 \
    --width 1024 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --seed 65513 \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir ${base_dir}/dvlm/counting/generation/geneval/lumina_${output_dir_name} \
    --prompt_files ${base_dir}/jiwon/geneval/prompts/evaluation_metadata_count.jsonl
