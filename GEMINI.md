# DeepSeek Janus Pro LoRA - Counting Task

## Project Overview

This project is focused on fine-tuning the **DeepSeek Janus-Pro-7B** multimodal model for a visual counting task. It adapts the model to accurately count objects in images using the **Pixmo** dataset. The project utilizes **LoRA (Low-Rank Adaptation)** for efficient fine-tuning and integrates with **WandB** for experiment tracking.

## Key Technologies

*   **Model:** `deepseek-ai/Janus-Pro-7B`
*   **Training:** PyTorch, Hugging Face Transformers, PEFT (LoRA)
*   **Data:** Hugging Face Datasets, Pillow (PIL)
*   **Logging:** Weights & Biases (WandB)

## Directory Structure & Key Files

*   **`train_counting.py`**: The main entry point for training. It defines the `EnhancedMultiModalModel` (wrapper for custom forward pass), `StreamingDatasetWrapper` (for on-the-fly image loading), and the training loop using Hugging Face `Trainer`.
*   **`process_pixmo_count.py`**: Data preparation script. It downloads images from URLs in the Pixmo dataset, filters valid entries, formats questions/answers, and saves the dataset to disk.
*   **`script/`**: Contains shell scripts for launching training jobs.
    *   `train_counting_0.sh`: Example script to run training with specific hyperparameters (batch size, learning rate, etc.).
*   **`multimodal_trainer.py`**: A trainer class implementation, likely an alternative or modularized version of the training logic.
*   **`data/`**: Directory where processed datasets are stored (e.g., `pixmo-count-filtered-imgContained`).
*   **`janus/`**: Contains the source code for the Janus model architecture (`models/`) and utilities.

## Setup & Usage

### 1. Environment Setup

Ensure you have a Python environment (recommend 3.10) with the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Before training, you must prepare the dataset. This script downloads images and formats the data:

```bash
python process_pixmo_count.py
```

This will save the processed dataset to `data/pixmo-count-filtered-imgContained`.

### 3. Training

Use the provided shell scripts to start training. You may need to adjust environment variables (like `WANDB_API_KEY`) or hyperparameters inside the script.

```bash
# Example: Run training on GPU 0
bash script/train_counting_0.sh
```

**Key Hyperparameters in `train_counting_0.sh`:**
*   `TUNING_MODE`: `transformer`, `lora`, or `full`.
*   `BATCH_SIZE`: Per-device batch size.
*   `GRAD_ACCUM_STEPS`: Gradient accumulation steps.
*   `LR`: Learning rate.

### 4. Validation

The training script includes a `ValidationCallback` that automatically evaluates the model on the validation set at specified intervals (`LOG_FREQ`) and logs the results (accuracy and sample predictions) to WandB.

## Notes

*   **`trump.zip` / `process_image_with_description.py`**: These appear to be legacy or demo files from the original repository and are not central to the current Pixmo counting task.
*   **Hardware**: The project is configured for GPU training (requires ~32GB VRAM for 7B model fine-tuning depending on batch size/LoRA settings).
