export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MASTER_ADDR=localhost
export MASTER_PORT=25001
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ngpus=$(echo ${CUDA_VISIBLE_DEVICES} | awk -F',' '{print NF}')
base_dir="/mnt/data1"
expected_images=1000  # must match --max_prompts (default 1000)

# Resume helper: skip if output_dir already has >= expected_images .png files
check_done() {
    local out_dir="$1"
    if [ -d "$out_dir" ]; then
        local count=$(find "$out_dir" -maxdepth 1 -name '*.png' | wc -l)
        if [ "$count" -ge "$expected_images" ]; then
            echo "[SKIP] $out_dir already has $count images (>= $expected_images). Skipping."
            return 0  # done
        else
            echo "[RESUME] $out_dir has $count/$expected_images images. Resuming."
            return 1  # not done
        fi
    fi
    return 1  # dir doesn't exist => not done
}

# ---------------------------------------------------------------------------- #
#                  Counting Understanding LoRA                                  #
# ---------------------------------------------------------------------------- #

checkpoint_path="jiwon/Lumina-DiMOO/output"
ckpt_list=(
    # "lora128_counting_wohead/epoch5"
    # "lora128_counting_wohead/epoch12"
    # "lora128_rel_position_wohead_und/epoch1"

)

for ckpt in "${ckpt_list[@]}"; do
    out_dir="${base_dir}/dvlm/lumina/fid/generation_eval/${ckpt}"
    if check_done "$out_dir"; then
        continue
    fi
    echo "Running FID inference for checkpoint: $ckpt"
    torchrun --nproc_per_node ${ngpus} evaluation_scripts/fid/fid_inference.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --height 1024 \
        --width 1024 \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --seed 65513 \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${out_dir} \
        --prompt_files captions_val2014.json \
        --num_samples 1 \
        --max_prompts ${expected_images}
done

# ---------------------------------------------------------------------------- #
#                                Baseline                                       #
# ---------------------------------------------------------------------------- #

# output_dir_name="BASELINE"
# out_dir="${base_dir}/dvlm/lumina/fid/generation_eval/${output_dir_name}"
# if ! check_done "$out_dir"; then
#     echo "Running FID inference for: $output_dir_name"
#     torchrun --nproc_per_node ${ngpus} evaluation_scripts/fid/fid_inference.py \
#         --checkpoint Alpha-VLLM/Lumina-DiMOO \
#         --height 1024 \
#         --width 1024 \
#         --timesteps 64 \
#         --cfg_scale 4.0 \
#         --seed 65513 \
#         --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
#         --output_dir ${out_dir} \
#         --prompt_files captions_val2014.json \
#         --num_samples 1 \
#         --max_prompts ${expected_images}
# fi

# ---------------------------------------------------------------------------- #
#                  Counting Generation LoRA                                     #
# ---------------------------------------------------------------------------- #

checkpoint_path="dvlm/lumina/checkpoints"
ckpt_list=(
    # "lumina_train_generation_wohead_1024/epoch5"
    # "lumina_train_generation_wohead_1024/epoch12"
    "lora128_rel_position_wohead_gen/epoch1"

)

for ckpt in "${ckpt_list[@]}"; do
    out_dir="${base_dir}/dvlm/lumina/fid/generation_eval/${ckpt}"
    if check_done "$out_dir"; then
        continue
    fi
    echo "Running FID inference for checkpoint: $ckpt"
    torchrun --nproc_per_node ${ngpus} evaluation_scripts/fid/fid_inference.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --height 1024 \
        --width 1024 \
        --timesteps 64 \
        --cfg_scale 4.0 \
        --seed 65513 \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --lora_ckpt_path ${base_dir}/${checkpoint_path}/${ckpt} \
        --output_dir ${out_dir} \
        --prompt_files captions_val2014.json \
        --num_samples 1 \
        --max_prompts ${expected_images}
done
