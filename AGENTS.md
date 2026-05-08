# Repository Guidelines

## Project Structure & Module Organization
Core model code lives in `models/`, with training entry points and data utilities in `training/`. Dataset loaders backed by parquet files are in `parquet/`. Runtime configuration is split between YAMLs in `configs/` and distributed launch presets in `accelerate_configs/`.

Top-level entry points are `app.py` for the Gradio demo, `generate.py` for text generation, `inference_mmu.py` for multimodal understanding, and `inference_t2i.py` for text-to-image inference. Evaluation code is under `evaluation/`, including `evaluation/VLMEvalKit/` and `evaluation/lm/`. Demo media is in `assets/`; validation inputs are in `validation_prompts/`, `mmu_validation/`, and `lm_chat_validation/`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install the main Python dependencies.
- `python app.py`: launch the local Gradio demo.
- `python generate.py`: run the default text generation path.
- `python3 inference_mmu.py config=configs/mmada_demo.yaml mmu_image_root=./mmu_validation mmu_prompts_file=./mmu_validation/prompts_with_vqa.json`: run MMU inference.
- `python3 inference_t2i.py config=configs/mmada_demo.yaml batch_size=1 validation_prompts_file=validation_prompts/text2image_prompts.txt guidance_scale=3.5 generation_timesteps=15 mode='t2i'`: run text-to-image inference.
- `accelerate launch --config_file accelerate_configs/1_gpu.yaml training/train_mmada.py config=configs/mmada_pretraining_stage1_llada_instruct.yaml`: launch a small training run.

## Coding Style & Naming Conventions
Use Python with 4-space indentation. Follow existing naming: `snake_case` for functions, variables, and config keys; `PascalCase` for model, dataset, and config classes. Keep command-line overrides compatible with the existing OmegaConf `key=value` style. There is no committed formatter configuration, so keep edits PEP 8-oriented and match nearby import ordering, logging patterns, and type hints.

## Testing Guidelines
There is no standalone pytest suite in this checkout. Validate changes with the smallest relevant script or evaluation path. For VLM evaluation, use `cd evaluation/VLMEvalKit && pip install -e .`, then `CUDA_VISIBLE_DEVICES=0 python run.py --data MathVista_MINI --model MMaDA-MixCoT`. For language evaluation, use `cd evaluation/lm && bash eval.sh`. If adding tests, name files `test_*.py`.

## Commit & Pull Request Guidelines
The available local history contains only the clone entry, so use clear, imperative subjects such as `training: fix mixed batch weighting` or `evaluation: add MathVista config`. Pull requests should state the affected task, config file, checkpoint or dataset assumptions, exact command run, GPU count, and any metric or wandb links.

## Security & Configuration Tips
Do not commit checkpoints, downloaded datasets, wandb credentials, or local output directories. Prefer config overrides or YAML edits for paths, and document any required private dataset locations in the PR rather than hard-coding them.
