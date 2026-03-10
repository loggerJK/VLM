#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

# based on CUDA
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

base_dir="/mnt/data1"
checkpoint_path="jiwon/Lumina-DiMOO/output"

STEPS=128
GEN_LENGTH=128
BLOCK_LENGTH=128
NUM_SAMPLES=250
MASTER_PORT=29505

# ---------------------------------------------------------------------------- #
#  Optional modes (uncomment ONE of the flags below to change evaluation mode):
#    --mqa                    : Multiple-choice (a/b/c/d) instead of open-ended
#    --predict_position_only  : Mask only position tokens in GT answer
#  These two flags are mutually exclusive.
# ---------------------------------------------------------------------------- #
EXTRA_FLAGS="--mqa"
# EXTRA_FLAGS="--predict_position_only"
# EXTRA_FLAGS=""

if [[ "$EXTRA_FLAGS" == *"--mqa"* ]]; then
    MODE_SUFFIX="_mqa"
elif [[ "$EXTRA_FLAGS" == *"--predict_position_only"* ]]; then
    MODE_SUFFIX="_position_only"
else
    MODE_SUFFIX=""
fi

output_dir_label="understanding_block128${MODE_SUFFIX}"

# ---------------------------------------------------------------------------- #
#                                   baseline                                   #
# ---------------------------------------------------------------------------- #

# output_dir_name="baseline"
# torchrun --nproc_per_node=$NGPUS --master_port=${MASTER_PORT} \
#     evaluation_scripts/rel_position/understanding/evaluate_understanding.py \
#     --checkpoint Alpha-VLLM/Lumina-DiMOO \
#     --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#     --output_dir ${base_dir}/dvlm/lumina/rel_position/${output_dir_label}/${output_dir_name} \
#     --steps ${STEPS} \
#     --gen_length ${GEN_LENGTH} \
#     --block_length ${BLOCK_LENGTH} \
#     --num_samples ${NUM_SAMPLES} \
#     ${EXTRA_FLAGS}


# ---------------------------------------------------------------------------- #
#                          Rel Position Understanding                          #
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
    echo "========================================"
    echo "Running rel_position understanding evaluation for checkpoint: $ckpt"
    echo "Output directory name: $ckpt"
    echo "========================================"

    torchrun --nproc_per_node=$NGPUS --master_port=${MASTER_PORT} \
        evaluation_scripts/rel_position/understanding/evaluate_understanding.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/lumina/rel_position/${output_dir_label}/${ckpt} \
        --steps ${STEPS} \
        --gen_length ${GEN_LENGTH} \
        --block_length ${BLOCK_LENGTH} \
        --num_samples ${NUM_SAMPLES} \
        ${EXTRA_FLAGS}

done

checkpoint_path="dvlm/lumina/checkpoints"
ckpt_list=(
    # "lora128_rel_position_wohead_gen/epoch0"
    # "lora128_rel_position_wohead_gen/epoch1"
    # "lora128_rel_position_wohead_gen/epoch2"
    # "lora128_rel_position_wohead_gen/epoch3"

    # "lora128_rel_position_wohead_gen/epoch0-iter15999-step500"
    # "lora128_rel_position_wohead_gen/epoch0-iter23999-step750"
    # "lora128_rel_position_wohead_gen/epoch0-iter31999-step1000"
    # "lora128_rel_position_wohead_gen/epoch0-iter39999-step1250"
    # "lora128_rel_position_wohead_gen/epoch0-iter47999-step1500"
    # "lora128_rel_position_wohead_gen/epoch0-iter7999-step250"
    # "lora128_rel_position_wohead_gen/epoch1-iter14175-step2000"
    # "lora128_rel_position_wohead_gen/epoch1-iter22175-step2250"
    # "lora128_rel_position_wohead_gen/epoch1-iter30175-step2500"
    # "lora128_rel_position_wohead_gen/epoch1-iter38175-step2750"
    # "lora128_rel_position_wohead_gen/epoch1-iter46175-step3000"
    # "lora128_rel_position_wohead_gen/epoch1-iter6175-step1750"
    # "lora128_rel_position_wohead_gen/epoch2-iter12351-step3500"
    # "lora128_rel_position_wohead_gen/epoch2-iter20351-step3750"
    # "lora128_rel_position_wohead_gen/epoch2-iter28351-step4000"
    # "lora128_rel_position_wohead_gen/epoch2-iter36351-step4250"
    # "lora128_rel_position_wohead_gen/epoch2-iter4351-step3250"
    # "lora128_rel_position_wohead_gen/epoch2-iter44351-step4500"
    # "lora128_rel_position_wohead_gen/epoch3-iter10527-step5000"
    # "lora128_rel_position_wohead_gen/epoch3-iter18527-step5250"
    # "lora128_rel_position_wohead_gen/epoch3-iter2527-step4750"
    # "lora128_rel_position_wohead_gen/epoch3-iter26527-step5500"
    # "lora128_rel_position_wohead_gen/epoch3-iter34527-step5750"
    # "lora128_rel_position_wohead_gen/epoch3-iter42527-step6000"
    # "lora128_rel_position_wohead_gen/epoch4-iter16703-step6750"
    # "lora128_rel_position_wohead_gen/epoch4-iter24703-step7000"
    # "lora128_rel_position_wohead_gen/epoch4-iter32703-step7250"
    # "lora128_rel_position_wohead_gen/epoch4-iter40703-step7500"
    # "lora128_rel_position_wohead_gen/epoch4-iter703-step6250"
    # "lora128_rel_position_wohead_gen/epoch4-iter8703-step6500"
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running rel_position understanding evaluation for checkpoint: $ckpt"
    echo "Output directory name: $ckpt"
    echo "========================================"

    torchrun --nproc_per_node=$NGPUS --master_port=${MASTER_PORT} \
        evaluation_scripts/rel_position/understanding/evaluate_understanding.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/lumina/rel_position/${output_dir_label}/${ckpt} \
        --steps ${STEPS} \
        --gen_length ${GEN_LENGTH} \
        --block_length ${BLOCK_LENGTH} \
        --num_samples ${NUM_SAMPLES} \
        ${EXTRA_FLAGS}

done
