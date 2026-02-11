# Lumina-DiMOO Project Context

## Project Overview
Lumina-DiMOO is an omni diffusion large language model designed for seamless multimodal generation and understanding. It features a unified discrete diffusion architecture capable of handling inputs and outputs across various modalities.

**Key Capabilities:**
*   **Text-to-Image Generation:** High-resolution generation from text prompts.
*   **Image-to-Image Generation:** Includes image editing, subject-driven generation, style transfer, and controllable generation (depth, canny, etc.).
*   **Image Inpainting & Extrapolation:** Extending or modifying existing images.
*   **Image Understanding:** Describing and analyzing images.
*   **Efficiency:** Accelerated sampling using a Max Logit-based Cache (ML-Cache).

## Environment Setup

1.  **Create Conda Environment:**
    ```bash
    conda create -n lumina_dimoo python=3.10 -y
    conda activate lumina_dimoo
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Key Scripts & Usage

### 1. Text-to-Image (T2I) Generation
*   **Script:** `inference/inference_t2i.py`
*   **Basic Usage:**
    ```bash
    python inference/inference_t2i.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --prompt "Your prompt here" \
        --height 768 --width 1536 \
        --timesteps 64 --cfg_scale 4.0 \
        --output_dir output/results_t2i
    ```
*   **With Caching (Faster):** Add `--use-cache`, `--cache_ratio`, `--warmup_ratio`, `--refresh_interval`.
*   **DDP Sampling:** Use `inference/inference_t2i_ddp.py` with `torchrun` for multi-GPU support.

### 2. Image-to-Image (I2I) & Editing
*   **Script:** `inference/inference_i2i.py`
*   **Modes:**
    *   **Controllable:** `--edit_type [depth_control|hed_control|openpose_control]`
    *   **Subject Driven:** `--edit_type subject_driven`
    *   **Editing:** `--edit_type [edit_add|edit_remove|edit_replace|edit_background]`
    *   **Style Transfer:** `--edit_type image_ref_transfer`
*   **Example (Editing):**
    ```bash
    python inference/inference_i2i.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --prompt "Edit instruction" \
        --image_path path/to/image.png \
        --edit_type edit_add \
        --output_dir output/results_i2i
    ```

### 3. Image Inpainting & Extrapolation
*   **Script:** `inference/inference_t2i.py` (Same as T2I but with specific arguments)
*   **Arguments:** `--painting_mode [inpainting|outpainting]`, `--painting_image`, `--mask_h_ratio`, `--mask_w_ratio`.

### 4. Image Understanding (MMU)
*   **Script:** `inference/inference_mmu.py`
*   **Usage:**
    ```bash
    python inference/inference_mmu.py \
        --checkpoint Alpha-VLLM/Lumina-DiMOO \
        --prompt "Describe this image." \
        --image_path path/to/image.jpg \
        --output_dir output/outputs_mmu
    ```

### 5. Training & Fine-Tuning
1.  **Pre-tokenization:** `bash pre_tokenizer/run_pre_token.sh` (Extracts discrete codes).
2.  **Training:** `bash train/train.sh` (Runs the training loop).

### 6. Benchmark Evaluation
*   **Location:** `VLMEvalKit/` directory.
*   **Setup:** Requires separate installation (`pip install -r requirements.txt` inside `VLMEvalKit`).
*   **Usage:** `python run.py --data [MMMU_DEV_VAL|POPE|etc] --model Lumina_DiMOO`.
*   **Note:** Requires `OPENAI_API_KEY` in `VLMEvalKit/.env`.

## Project Structure
*   `assets/`: Images and JSON samples for documentation and testing.
*   `configs/`: Configuration files (e.g., `data.yaml`).
*   `examples/`: Sample images for inference testing.
*   `generators/`: Core logic for different generation tasks (T2I, I2I, MMU).
*   `inference/`: Entry scripts for running inference.
*   `model/`: Model architecture definitions (`modeling_llada.py`, `modeling_xllmx_dimoo.py`).
*   `pre_tokenizer/`: Scripts for data preprocessing and tokenization.
*   `train/`: Training scripts.
*   `utils/`: Utility functions for image processing, prompting, etc.
*   `VLMEvalKit/`: Sub-module for benchmarking and evaluation.
*   `xllmx/`: Likely a supporting library or module for the model's backend.
