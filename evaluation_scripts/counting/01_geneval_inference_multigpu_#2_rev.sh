# export CUDA_VISIBLE_DEVICES=7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MASTER_ADDR=localhost
export MASTER_PORT=25010
export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7 # Set this according to your available
ngpus=6
# base_dir="/mnt/cvlab22_data1"
base_dir="/mnt/data1"

checkpoint_path="jiwon/Lumina-DiMOO/output"
ckpt_list=(
    "lora128_counting_wohead/epoch0"
    "lora128_counting_wohead/epoch2"
    "lora128_counting_wohead/epoch4"
    "lora128_counting_wohead/epoch6"
    "lora128_counting_wohead/epoch8"
    "lora128_counting_wohead/epoch10"
    "lora128_counting_wohead/epoch12"
    "lora128_counting_wohead/epoch14"

    "lora128_counting_wohead/epoch1"
    "lora128_counting_wohead/epoch3"
    "lora128_counting_wohead/epoch5"
    "lora128_counting_wohead/epoch7"
    "lora128_counting_wohead/epoch9"
    "lora128_counting_wohead/epoch11"
    "lora128_counting_wohead/epoch13"
    "lora128_counting_wohead/epoch15"
    
    "lora128_counting_wohead_resumeEpoch12_lr3e-6/epoch11"
    "lora128_counting_wohead_resumeEpoch12_lr3e-6/epoch12"
    "lora128_counting_wohead_resumeEpoch12_lr3e-6/epoch13"
    "lora128_counting_wohead_resumeEpoch12_lr3e-6/epoch14"
    "lora128_counting_wohead_resumeEpoch12_lr3e-6/epoch15"

    "lora128_counting_wohead_range0-7_resumeEpoch10_lr3e-6/epoch13"
    "lora128_counting_wohead_range0-7_resumeEpoch10_lr3e-6/epoch14"
    "lora128_counting_wohead_range0-7_resumeEpoch10_lr3e-6/epoch15"
    "lora128_counting_wohead_range0-7_resumeEpoch10_lr3e-6/epoch16"
    "lora128_counting_wohead_range0-7_resumeEpoch10_lr3e-6/epoch17"
    "lora128_counting_wohead_range0-7_resumeEpoch10_lr3e-6/epoch18"
)

for ((i=${#ckpt_list[@]}-1; i>=0; i--)); do
    ckpt="${ckpt_list[i]}"
    output_dir_name=$(echo "$ckpt" | tr '/' '_')
    echo "Running inference for checkpoint: $ckpt"
    echo "Output directory name: $output_dir_name"
    torchrun --rdzv_endpoint=localhost:25010 --nproc_per_node ${ngpus} inference/inference_t2i_multigpu.py\
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --height 1024 \
        --width 1024 \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --seed 65513 \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/generation_eval/lumina_${output_dir_name} \
        --prompt_files ${base_dir}/jiwon/geneval/prompts/evaluation_metadata_count.jsonl
done

# /mnt/data1/dvlm/checkpoints
checkpoint_path="dvlm/checkpoints"
ckpt_list=(
    "lumina_train_generation_wohead_1024/epoch0"
    "lumina_train_generation_wohead_1024/epoch1"
    "lumina_train_generation_wohead_1024/epoch2"
    "lumina_train_generation_wohead_1024/epoch3"
    "lumina_train_generation_wohead_1024/epoch4"
    "lumina_train_generation_wohead_1024/epoch5"
    "lumina_train_generation_wohead_1024/epoch6"
    "lumina_train_generation_wohead_1024/epoch7"
    "lumina_train_generation_wohead_1024/epoch8"
    "lumina_train_generation_wohead_1024/epoch9"
    "lumina_train_generation_wohead_1024/epoch10"
    
    "lora128_counting_wohead_both/epoch0"
    "lora128_counting_wohead_both/epoch1"
    "lora128_counting_wohead_both/epoch2"
    "lora128_counting_wohead_both/epoch3"
    "lora128_counting_wohead_both/epoch4"
    "lora128_counting_wohead_both/epoch5"
    "lora128_counting_wohead_both/epoch6"
    "lora128_counting_wohead_both/epoch7"
    "lora128_counting_wohead_both/epoch8"
    "lora128_counting_wohead_both/epoch9"
    "lora128_counting_wohead_both/epoch10"
)

for ((i=${#ckpt_list[@]}-1; i>=0; i--)); do
    ckpt="${ckpt_list[i]}"
    output_dir_name=$(echo "$ckpt" | tr '/' '_')
    echo "Running inference for checkpoint: $ckpt"
    echo "Output directory name: $output_dir_name"
    torchrun --rdzv_endpoint=localhost:25010 --nproc_per_node ${ngpus} inference/inference_t2i_multigpu.py\
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --height 1024 \
        --width 1024 \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --seed 65513 \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/generation_eval/lumina_${output_dir_name} \
        --prompt_files ${base_dir}/jiwon/geneval/prompts/evaluation_metadata_count.jsonl
done
