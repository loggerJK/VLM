# Repository Guidelines

## Project Structure & Module Organization

This repository contains Lumina-DiMOO, a Python multimodal generation and understanding model built around LLaDA. Core model code lives in `model/`, with generation logic in `generators/` and shared helpers in `utils/`. Training entry points and shell wrappers are in `train/`; inference scripts are in `inference/`; benchmark and task evaluations are under `evaluation_scripts/`. Data processors and distributed training utilities are split between `data/` and `xllmx/`. Configuration files live in `configs/` and `lumina.yaml`. Keep sample inputs in `assets/` or `examples/`, and treat `VLMEvalKit/` as a vendored evaluation toolkit unless your change is evaluation-specific.

## Build, Test, and Development Commands

Create the main environment with:

```bash
conda env create -f lumina.yaml
conda activate lumina_dimoo
export PYTHONNOUSERSITE=1
```

For pip-based setup, use `pip install -r requirements.txt`. OCR/document rendering utilities may also need `bash dependency.sh`.

Run single-task inference with `python inference/inference_t2i.py --checkpoint <path> --vae_ckpt Alpha-VLLM/Lumina-DiMOO --output_dir output/`. Use `torchrun --nproc_per_node=<N> inference/inference_t2i_ddp.py ...` for distributed sampling. Training generally uses `python -m torch.distributed.run --nproc_per_node=<N> train/train_unified.py --task <task> --data_config configs/data.yaml ...`; prefer existing `train/*.sh` wrappers when matching an experiment.

## Coding Style & Naming Conventions

Use Python 3.10, 4-space indentation, and descriptive `snake_case` for functions, variables, and script flags. Keep task-specific scripts named by task and mode, for example `train_unified_ocr.sh` or `evaluate_pixmo_multigpu.py`. Follow existing argparse-style CLIs and avoid hardcoded local paths; expose paths as flags or YAML config entries. No repository-wide formatter config is present, so keep edits consistent with surrounding files.

## Testing Guidelines

There is no formal unit-test suite. Validate changes with the narrowest runnable smoke test: `python -m py_compile <changed_files>`, a relevant script in `test_scripts/`, or a small inference/evaluation run. For distributed changes, test both single-process and `torchrun` paths when practical. Store generated outputs under `output/` or an experiment directory, not in source folders.

## Commit & Pull Request Guidelines

Recent history uses short prefixes such as `feat:`, `fix:`, `chore:`, and `add:`. Prefer `type: concise summary`, for example `fix: skip corrupted images in dataset loader`. Pull requests should describe the task, changed scripts/configs, required checkpoints or datasets, and exact validation commands. Include sample outputs or metrics for inference, training, or evaluation changes, and never commit API keys, W&B credentials, checkpoints, or large generated artifacts.
