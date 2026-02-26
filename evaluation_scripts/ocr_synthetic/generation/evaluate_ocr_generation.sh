#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

base_dir="/mnt/data1"
checkpoint_path="jiwon/Lumina-DiMOO/output"
NUM_SAMPLES=250
# Torchrun endpoint
MASTER_ADDR="localhost"
MASTER_PORT=29501
# ---------------------------------------------------------------------------- #
#                                   baseline                                   #
# ---------------------------------------------------------------------------- #

# echo "========================================"
# echo "Running baseline OCR generation evaluation for quotes dataset "
# echo "========================================"

# torchrun --nproc_per_node=$NGPUS \
#     evaluation_scripts/ocr_synthetic/generation/evaluate_ocr_generation.py \
#     --checkpoint Alpha-VLLM/Lumina-DiMOO \
#     --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#     --output_dir ${base_dir}/dvlm/lumina/ocr_synthetic/generation/baseline \
#     --timesteps 64 \
#     --cfg_scale 4.0 \
#     --height 1024 \
#     --width 1024 \
#     --ocr_model_path zai-org/GLM-OCR \
#     --num_samples ${NUM_SAMPLES}

# ---------------------------------------------------------------------------- #
#                               OCR_Understanding                              #
# ---------------------------------------------------------------------------- #

ckpt_list=(
    # "lora128_ocr_4_images_wohead/epoch0"
    # "lora128_ocr_4_images_wohead/epoch1"
    # "lora128_ocr_4_images_wohead/epoch2"
    # "lora128_ocr_4_images_wohead/epoch3"
    # "lora128_ocr_4_images_wohead/epoch4"
    # "lora128_ocr_4_images_wohead/epoch5"
    # "lora128_ocr_4_images_wohead/epoch6"
    # "lora128_ocr_4_images_wohead/epoch7"
    # "lora128_ocr_4_images_wohead/epoch8"
    # "lora128_ocr_4_images_wohead/epoch9"
    # "lora128_ocr_4_images_wohead/epoch10"
    # "lora128_ocr_4_images_wohead/epoch11"
    # "lora128_ocr_4_images_wohead/epoch12"
    # "lora128_ocr_4_images_wohead/epoch13"
    # "lora128_ocr_4_images_wohead/epoch14"
    # "lora128_ocr_4_images_wohead/epoch15"
    # "lora128_ocr_4_images_wohead/epoch16"
    # "lora128_ocr_4_images_wohead/epoch17"
    # "lora128_ocr_4_images_wohead/epoch18"
    # Add more checkpoints here...
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running OCR generation evaluation for checkpoint: $ckpt on quotes dataset"
    echo "========================================"

    torchrun --nproc_per_node=$NGPUS \
        evaluation_scripts/ocr_synthetic/generation/evaluate_ocr_generation.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/lumina/ocr_synthetic/generation/${ckpt} \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --height 1024 \
        --width 1024 \
        --ocr_model_path zai-org/GLM-OCR \
        --num_samples ${NUM_SAMPLES}

done

checkpoint_path="dvlm/lumina/checkpoints"
ckpt_list=(
    # ---------------------------------------------------------------------------- #
    #                               OCR Understanding                              #
    # ---------------------------------------------------------------------------- #

    # "lora128_ocr_synthetic_wohead/epoch0"
    # "lora128_ocr_synthetic_wohead/epoch1"
    # "lora128_ocr_synthetic_wohead/epoch2"
    # "lora128_ocr_synthetic_wohead/epoch3"
    # "lora128_ocr_synthetic_wohead/epoch4"
    # "lora128_ocr_synthetic_wohead/epoch5"
    # "lora128_ocr_synthetic_wohead/epoch6"

    # ---------------------------------------------------------------------------- #
    #                                OCR Generation                                #
    # ---------------------------------------------------------------------------- #

    # "lora128_ocr_synthetic_wohead_gen/epoch0-iter15999-step500"
    # "lora128_ocr_synthetic_wohead_gen/epoch0-iter31999-step1000"
    # "lora128_ocr_synthetic_wohead_gen/epoch0-iter47999-step1500"
    # # "lora128_ocr_synthetic_wohead_gen/epoch1"
    # "lora128_ocr_synthetic_wohead_gen/epoch1-iter14015-step2000"
    # "lora128_ocr_synthetic_wohead_gen/epoch1-iter30015-step2500"
    # "lora128_ocr_synthetic_wohead_gen/epoch1-iter46015-step3000"
    # # "lora128_ocr_synthetic_wohead_gen/epoch2"
    # "lora128_ocr_synthetic_wohead_gen/epoch2-iter12031-step3500"
    # "lora128_ocr_synthetic_wohead_gen/epoch2-iter28031-step4000"
    # "lora128_ocr_synthetic_wohead_gen/epoch2-iter44031-step4500"
    # # "lora128_ocr_synthetic_wohead_gen/epoch3"
    # "lora128_ocr_synthetic_wohead_gen/epoch3-iter10047-step5000"
    # "lora128_ocr_synthetic_wohead_gen/epoch3-iter26047-step5500"
    # "lora128_ocr_synthetic_wohead_gen/epoch3-iter42047-step6000"
    # # "lora128_ocr_synthetic_wohead_gen/epoch4"
    # "lora128_ocr_synthetic_wohead_gen/epoch4-iter24063-step7000"
    # "lora128_ocr_synthetic_wohead_gen/epoch4-iter40063-step7500"
    # "lora128_ocr_synthetic_wohead_gen/epoch4-iter8063-step6500"
    # # "lora128_ocr_synthetic_wohead_gen/epoch5"
    # "lora128_ocr_synthetic_wohead_gen/epoch5-iter22079-step8500"
    # "lora128_ocr_synthetic_wohead_gen/epoch5-iter38079-step9000"
    # "lora128_ocr_synthetic_wohead_gen/epoch5-iter6079-step8000"
    # # "lora128_ocr_synthetic_wohead_gen/epoch6"
    # "lora128_ocr_synthetic_wohead_gen/epoch6-iter20095-step10000"
    # "lora128_ocr_synthetic_wohead_gen/epoch6-iter36095-step10500"
    # "lora128_ocr_synthetic_wohead_gen/epoch6-iter4095-step9500"
    # # "lora128_ocr_synthetic_wohead_gen/epoch7"
    # "lora128_ocr_synthetic_wohead_gen/epoch7-iter18111-step11500"
    # "lora128_ocr_synthetic_wohead_gen/epoch7-iter2111-step11000"
    # "lora128_ocr_synthetic_wohead_gen/epoch7-iter34111-step12000"
    # "lora128_ocr_synthetic_wohead_gen/epoch8-iter127-step12500"
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running OCR generation evaluation for checkpoint: $ckpt on quotes dataset"
    echo "========================================"

    torchrun --nproc_per_node=$NGPUS \
        evaluation_scripts/ocr_synthetic/generation/evaluate_ocr_generation.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/lumina/ocr_synthetic/generation/${ckpt} \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --height 1024 \
        --width 1024 \
        --ocr_model_path zai-org/GLM-OCR \
        --num_samples ${NUM_SAMPLES}
done



checkpoint_path="jiwon/Lumina-DiMOO/output"

ckpt_list=(
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter799-step25
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter1599-step50
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter2399-step75
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter3199-step100
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter3999-step125
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter4799-step150
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter5599-step175
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter6399-step200
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter7199-step225
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter7999-step250
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter8799-step275
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter9599-step300
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter10399-step325
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter11199-step350
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter11999-step375
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter12799-step400
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter13599-step425
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter14399-step450
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter15199-step475
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter15999-step500
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter16799-step525
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter17599-step550
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter18399-step575
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter19199-step600
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter19999-step625
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter20799-step650
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter21599-step675
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter22399-step700
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter23199-step725
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter23999-step750
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter24799-step775
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter25599-step800
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter26399-step825
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter27199-step850
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter27999-step875
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter28799-step900
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter29599-step925
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter30399-step950
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter31199-step975
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter31999-step1000
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter32799-step1025
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter33599-step1050
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter34399-step1075
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter35199-step1100
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter35999-step1125
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter36799-step1150
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter37599-step1175
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter38399-step1200
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter39199-step1225
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter39999-step1250
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter40799-step1275
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter41599-step1300
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter42399-step1325
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter43199-step1350
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter43999-step1375
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter44799-step1400
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter45599-step1425
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter46399-step1450
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter47199-step1475
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter47999-step1500
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter48799-step1525
    lora128_ocr_synthetic_wohead_gen_#2/epoch0-iter49599-step1550
)

for ckpt in "${ckpt_list[@]}"; do
    echo "========================================"
    echo "Running OCR generation evaluation for checkpoint: $ckpt on quotes dataset"
    echo "========================================"

    torchrun --nproc_per_node=$NGPUS --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        evaluation_scripts/ocr_synthetic/generation/evaluate_ocr_generation.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${base_dir}/dvlm/lumina/ocr_synthetic/generation/${ckpt} \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --height 1024 \
        --width 1024 \
        --ocr_model_path zai-org/GLM-OCR \
        --num_samples ${NUM_SAMPLES}
done
