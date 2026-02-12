BASE_CKPT_DIR="/mnt/data1/heeji/dVLM_outputs_from_ext_srv/lora128_spatial_relation_wohead_und-20260126-144357"
BASE_OUT_DIR="/mnt/data1/dvlm/spatial/understanding/blink"

for EPOCH in {0..9}; do
    CKPT_PATH="${BASE_CKPT_DIR}/epoch${EPOCH}"
    OUTPUT_DIR="${BASE_OUT_DIR}/epoch${EPOCH}"

    echo "======================================"
    echo "Evaluating epoch${EPOCH}"
    echo "CKPT: ${CKPT_PATH}"
    echo "OUT : ${OUTPUT_DIR}"
    echo "======================================"

    mkdir -p "${OUTPUT_DIR}"

    CUDA_VISIBLE_DEVICES=0 python evaluate_blink_spatial.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --vae_ckpt Alpha-VLLM/Lumina-DiMOO \
        --output_dir "${OUTPUT_DIR}" \
        --steps 20 \
        --gen_length 20 \
        --block_length 20 \
        --lora_ckpt_path "${CKPT_PATH}" \
        2>&1 | tee "${OUTPUT_DIR}/eval.log"
done