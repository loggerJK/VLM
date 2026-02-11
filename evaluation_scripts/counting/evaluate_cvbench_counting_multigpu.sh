export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=25001
ngpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')



# base_dir="/mnt/data1"
base_dir="/mnt/cvlab22_data1"
checkpoint_path="jiwon/Lumina-DiMOO/output"

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

echo "========================================"
ckpt="BASELINE"
output_dir_name="BASELINE"
echo "Running evaluation for checkpoint: $ckpt"
echo "Output directory name: $output_dir_name"
echo "========================================"

PYTHONPATH=${base_dir}/heeji/VLM torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${base_dir}/heeji/VLM/evaluate_cvbench_counting_multigpu.py \
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir ${base_dir}/dvlm/understanding_eval_cvbench/${output_dir_name} \
    --steps 20 \
    --gen_length 20 \
    --block_length 20 


ckpt_list=(
    # Understanding finetuing
    "lora128_counting_wohead/epoch0"
    "lora128_counting_wohead/epoch2"
    "lora128_counting_wohead/epoch4"
    "lora128_counting_wohead/epoch6"
    "lora128_counting_wohead/epoch8"
    "lora128_counting_wohead/epoch10"
    "lora128_counting_wohead/epoch12"
    "lora128_counting_wohead/epoch14"
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

    "lora128_counting_wohead/epoch1"
    "lora128_counting_wohead/epoch3"
    "lora128_counting_wohead/epoch5"
    "lora128_counting_wohead/epoch7"
    "lora128_counting_wohead/epoch9"
    "lora128_counting_wohead/epoch11"
    "lora128_counting_wohead/epoch13"
)

for ckpt in "${ckpt_list[@]}"; do
    output_dir_name=$(echo "$ckpt" | tr '/' '_') # Replace '/' with '_'
    echo "========================================"
    echo "Running evaluation for checkpoint: $ckpt"
    echo "Output directory name: $output_dir_name"
    echo "========================================"
    
    PYTHONPATH=${base_dir}/heeji/VLM torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${base_dir}/heeji/VLM/evaluate_cvbench_counting_multigpu.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/understanding_eval_cvbench/${output_dir_name} \
        --steps 20 \
        --gen_length 20 \
        --block_length 20 
done

base_dir="/mnt/data1"
checkpoint_path="dvlm/checkpoints"
ckpt_list=(
    #### Generation 
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
    "lumina_train_generation_wohead_1024/epoch11"
    "lumina_train_generation_wohead_1024/epoch12"

    #### Generation + Understanding
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
    "lora128_counting_wohead_both/epoch11"
    "lora128_counting_wohead_both/epoch12"
    "lora128_counting_wohead_both/epoch13"
    "lora128_counting_wohead_both/epoch14"
    "lora128_counting_wohead_both/epoch15"
    "lora128_counting_wohead_both/epoch16"
    "lora128_counting_wohead_both/epoch17"
    "lora128_counting_wohead_both/epoch18"
    "lora128_counting_wohead_both/epoch19"
    "lora128_counting_wohead_both/epoch20"
    "lora128_counting_wohead_both/epoch21"
    "lora128_counting_wohead_both/epoch22"
    "lora128_counting_wohead_both/epoch23"
    "lora128_counting_wohead_both/epoch24"
    "lora128_counting_wohead_both/epoch25"
    "lora128_counting_wohead_both/epoch26"
    "lora128_counting_wohead_both/epoch27"
    "lora128_counting_wohead_both/epoch28"
    "lora128_counting_wohead_both/epoch29"
    "lora128_counting_wohead_both/epoch30"
    "lora128_counting_wohead_both/epoch31"
    "lora128_counting_wohead_both/epoch32"
    "lora128_counting_wohead_both/epoch33"
    "lora128_counting_wohead_both/epoch34"
    "lora128_counting_wohead_both/epoch35"
    "lora128_counting_wohead_both/epoch36"
)

for ckpt in "${ckpt_list[@]}"; do
    output_dir_name=$(echo "$ckpt" | tr '/' '_')
    echo "========================================"
    echo "Running evaluation for checkpoint: $ckpt"
    echo "Output directory name: $output_dir_name"
    echo "========================================"
    
    PYTHONPATH=${base_dir}/heeji/VLM torchrun --nproc_per_node=$ngpus --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${base_dir}/heeji/VLM/evaluate_cvbench_counting_multigpu.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/understanding_eval_cvbench/${output_dir_name} \
        --steps 20 \
        --gen_length 20 \
        --block_length 20 
done
