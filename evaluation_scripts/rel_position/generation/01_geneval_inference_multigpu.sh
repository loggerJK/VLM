# export CUDA_VISIBLE_DEVICES=7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MASTER_ADDR=localhost
export MASTER_PORT=25001
export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7 # Set this according to your available
# export CUDA_VISIBLE_DEVICES=6,7 # Set this according to your available
ngpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')
# base_dir="/mnt/cvlab22_data1"
base_dir="/mnt/data1"
task="rel_position"

# ---------------------------------------------------------------------------- #
#                                   Baseline                                   #
# ---------------------------------------------------------------------------- #

output_dir_name="baseline"
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
    --output_dir ${base_dir}/dvlm/lumina/${task}/generation/${output_dir_name} \
    --prompt_files evaluation_scripts/rel_position/generation/evaluation_metadata_diag_rel_positions.jsonl


# ---------------------------------------------------------------------------- #
#                        내부 체크포인트들 / Understanding Only                        #
# ---------------------------------------------------------------------------- #


checkpoint_path="jiwon/Lumina-DiMOO/output"
ckpt_list=(

    "lora128_rel_position_wohead_und/epoch0"
    "lora128_rel_position_wohead_und/epoch1"
    "lora128_rel_position_wohead_und/epoch2"
    "lora128_rel_position_wohead_und/epoch3"
    "lora128_rel_position_wohead_und/epoch4"


    "lora128_rel_position_wohead_und/epoch0-iter7999-step250"
    "lora128_rel_position_wohead_und/epoch0-iter15999-step500"
    "lora128_rel_position_wohead_und/epoch0-iter23999-step750"
    "lora128_rel_position_wohead_und/epoch0-iter31999-step1000"
    "lora128_rel_position_wohead_und/epoch0-iter39999-step1250"
    "lora128_rel_position_wohead_und/epoch0-iter47999-step1500"
    "lora128_rel_position_wohead_und/epoch1-iter6175-step1750"
    "lora128_rel_position_wohead_und/epoch1-iter14175-step2000"
    "lora128_rel_position_wohead_und/epoch1-iter22175-step2250"
    "lora128_rel_position_wohead_und/epoch1-iter30175-step2500"
    "lora128_rel_position_wohead_und/epoch1-iter38175-step2750"
    "lora128_rel_position_wohead_und/epoch1-iter46175-step3000"
    "lora128_rel_position_wohead_und/epoch2-iter4351-step3250"
    "lora128_rel_position_wohead_und/epoch2-iter12351-step3500"
    "lora128_rel_position_wohead_und/epoch2-iter20351-step3750"
    "lora128_rel_position_wohead_und/epoch2-iter28351-step4000"
    "lora128_rel_position_wohead_und/epoch2-iter36351-step4250"
    "lora128_rel_position_wohead_und/epoch2-iter44351-step4500"
    "lora128_rel_position_wohead_und/epoch3-iter2527-step4750"
    "lora128_rel_position_wohead_und/epoch3-iter10527-step5000"
    "lora128_rel_position_wohead_und/epoch3-iter18527-step5250"
    "lora128_rel_position_wohead_und/epoch3-iter26527-step5500"
    "lora128_rel_position_wohead_und/epoch3-iter34527-step5750"
    "lora128_rel_position_wohead_und/epoch3-iter42527-step6000"
    "lora128_rel_position_wohead_und/epoch4-iter703-step6250"
    "lora128_rel_position_wohead_und/epoch4-iter8703-step6500"
    "lora128_rel_position_wohead_und/epoch4-iter16703-step6750"
    "lora128_rel_position_wohead_und/epoch4-iter24703-step7000"
    "lora128_rel_position_wohead_und/epoch4-iter32703-step7250"
    "lora128_rel_position_wohead_und/epoch4-iter40703-step7500"
    "lora128_rel_position_wohead_und/epoch4-iter48703-step7750"
    "lora128_rel_position_wohead_und/epoch5-iter6879-step8000"
)


for ckpt in "${ckpt_list[@]}"; do
    echo "Running inference for checkpoint: $ckpt"
    echo "Output directory name: $ckpt"
    torchrun --nproc_per_node ${ngpus} inference/inference_t2i_multigpu.py\
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --height 1024 \
        --width 1024 \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --seed 65513 \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/lumina/${task}/generation/${ckpt} \
        --prompt_files evaluation_scripts/rel_position/generation/evaluation_metadata_diag_rel_positions.jsonl
done


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
    echo "Running inference for checkpoint: $ckpt"
    echo "Output directory name: $ckpt"
    torchrun --nproc_per_node ${ngpus} inference/inference_t2i_multigpu.py\
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --height 1024 \
        --width 1024 \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --seed 65513 \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/lumina/${task}/generation/${ckpt} \
        --prompt_files evaluation_metadata_diag_rel_positions.jsonl
done
