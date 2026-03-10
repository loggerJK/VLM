# export CUDA_VISIBLE_DEVICES=7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MASTER_ADDR=localhost
export MASTER_PORT=25001
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # Set this according to your available
export CUDA_VISIBLE_DEVICES=2,3 # Set this according to your available
ngpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')
# base_dir="/mnt/cvlab22_data1"
base_dir="/mnt/data1"
task="rel_position_v2"
seed="42"
PROMPT_FILES_PATH="evaluation_scripts/rel_position/generation/evaluation_metadata_diag_rel_positions_v2.jsonl"
NUM_SAMPLES=20



# ---------------------------------------------------------------------------- #
#                                   외부 체크포인트들                                  #
# ---------------------------------------------------------------------------- #

# /mnt/data1/dvlm/checkpoints
checkpoint_path="dvlm/lumina/checkpoints"
ckpt_list=(
    # "lora128_rel_position_wohead_gen/epoch0"
    "lora128_rel_position_wohead_gen/epoch1"
    # "lora128_rel_position_wohead_gen/epoch2"
    # "lora128_rel_position_wohead_gen/epoch3"
    # "lora128_rel_position_wohead_gen/epoch4"

)

for ckpt in "${ckpt_list[@]}"; do # 순방향 인퍼런스
    echo "Running inference for checkpoint: $ckpt"
    echo "Output directory name: $ckpt"
    torchrun --nproc_per_node ${ngpus} --rdzv_endpoint 127.0.0.1:$MASTER_PORT  inference/inference_t2i_multigpu.py\
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --height 1024 \
        --width 1024 \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --seed ${seed} \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/lumina/${task}/generation/${ckpt} \
        --prompt_files ${PROMPT_FILES_PATH} \
        --num_samples ${NUM_SAMPLES}

done

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

output_dir_name="baseline"
# echo "Running inference for checkpoint: $ckpt"
echo "Output directory name: $output_dir_name"
torchrun --nproc_per_node ${ngpus} --rdzv_endpoint 127.0.0.1:$MASTER_PORT inference/inference_t2i_multigpu.py\
    --checkpoint Alpha-VLLM/Lumina-DiMOO \
    --height 1024 \
    --width 1024 \
    --timesteps 64 \
    --cfg_scale 4.0 \
    --seed ${seed} \
    --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
    --output_dir ${base_dir}/dvlm/lumina/${task}/generation/${output_dir_name} \
    --prompt_files ${PROMPT_FILES_PATH} \
    --num_samples ${NUM_SAMPLES}



# ---------------------------------------------------------------------------- #
#                        내부 체크포인트들 / Understanding Only                        #
# ---------------------------------------------------------------------------- #


checkpoint_path="jiwon/Lumina-DiMOO/output"
ckpt_list=(

    # "lora128_rel_position_wohead_und/epoch0"
    "lora128_rel_position_wohead_und/epoch1"
    # "lora128_rel_position_wohead_und/epoch2"
    # "lora128_rel_position_wohead_und/epoch3"
    # "lora128_rel_position_wohead_und/epoch4"


)


for ckpt in "${ckpt_list[@]}"; do
    echo "Running inference for checkpoint: $ckpt"
    echo "Output directory name: $ckpt"
    torchrun --nproc_per_node ${ngpus} --rdzv_endpoint 127.0.0.1:$MASTER_PORT  inference/inference_t2i_multigpu.py\
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --height 1024 \
        --width 1024 \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --seed ${seed} \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/lumina/${task}/generation/${ckpt} \
        --prompt_files ${PROMPT_FILES_PATH} \
        --num_samples ${NUM_SAMPLES}

done

