# Agent Guide for XCube

## Purpose
XCube is a research codebase for large-scale 3D generative modeling using sparse voxel hierarchies. It trains and evaluates hierarchical VAE + diffusion models over sparse voxel grids (fvdb / OpenVDB), with scripts for ShapeNet, Objaverse, and Waymo-style data.

This repo is Linux + NVIDIA GPU focused. Most workflows assume CUDA, PyTorch Lightning, and custom CUDA extensions.

## Quick Orientation
Key entry points
- `train.py`: main training entry point for VAEs and diffusion models.
- `test.py`: evaluation runner (loads checkpoints, runs `trainer.test`).
- `inference/`: sampling scripts for ShapeNet, Objaverse, Waymo.
- `datagen/`: data preparation utilities (e.g., ShapeNet mesh to VDB).

Core packages
- `xcube/`: primary library code
  - `xcube/models/`: model definitions
  - `xcube/modules/`: model building blocks
  - `xcube/data/`: datasets and datamodules
  - `xcube/utils/`: common helpers

Supporting directories
- `configs/`: YAML configs for datasets and training stages
- `ext/`: CUDA/CPP extensions (e.g., `nksr-cuda`) and common C++ utilities
- `assets/`: figures and setup helpers
- `slurm/`: cluster submission scripts
- `checkpoints/`: expected location for pretrained weights
- `results/`: inference outputs

## Environment Setup (Linux + GPU)
The repo uses **uv** with `pyproject.toml` for package management.

Typical setup
- Run `bash scripts/install.sh` to create a venv and install all dependencies.
- This installs PyTorch 2.8.0 + CUDA 12.6, fvdb-core 0.3.0 (built from source), XCube, and nksr CUDA extensions.
- Activate with `source .venv/bin/activate`.

Notes
- fvdb-core requires recent NVIDIA GPUs (Ampere or newer).
- fvdb-core must be built from source (no pre-built cu126 wheels). The install script handles this.
- Custom CUDA extensions (`ext/nksr-cuda`) are built with `--no-build-isolation` (they depend on installed torch/fvdb).
- PyTorch Lightning 2.x is used (migrated from 1.9.4).

## Data & Checkpoints
- Checkpoints are expected under `checkpoints/`.
- Results are written under `results/` by inference scripts.
- ShapeNet data is external; paths are configured in YAML under `configs/shapenet/*/data.yaml`.
- MISC tips for data prep and voxelization live in `MISC.md`.

## Training Workflow
Training uses PyTorch Lightning with configs:
- Configs are read via `exp.ArgumentParserX` with base config `configs/default/param.yaml`.
- Typical invocation: `python train.py <config.yaml> --wname <run-name> --gpus <N> ...`.
- `--wname` is used for experiment naming (WandB by default).

Stages
- Stage 1 (coarse): train VAE then diffusion at lower resolutions.
- Stage 2 (fine): train sparse VAE then diffusion at higher resolutions.

See `README.md` for concrete commands.

## Inference Workflow
Use scripts in `inference/`:
- `sample_shapenet.py`, `sample_waymo.py`, `sample_objaverse.py`.
- Most scripts accept `--ema`, `--use_ddim`, `--ddim_step`, and `--extract_mesh` flags.
- Visualize outputs via `visualize_object.py` or `visualize_scene.py`.

## Evaluation
- Use `python test.py --ckpt <path>` to run evaluation on a checkpoint.
- `test.py` can also load from W&B if configured.

## Logging & Experiment Tracking
- Default logger is WandB. Disable with `--logger_type none` or `--nolog`.
- Checkpoints are saved via PyTorch Lightning callbacks; see `train.py` for details.

## Gotchas & Constraints
- Linux-only support (per README).
- GPU + CUDA toolchain required for fvdb and CUDA extensions.
- This repo uses uv + `pyproject.toml` for packaging. Install via `scripts/install.sh`.
- Scripts sometimes assume large GPU memory and multi-GPU setups; reduce batch sizes and turn on grad accumulation if needed.

## How To Make Changes Safely
- Prefer editing configs rather than hardcoding parameters.
- Keep training/inference scripts CLI-compatible; avoid breaking existing flags.
- When touching CUDA extensions under `ext/`, expect rebuilds.
- If adding new datasets, mirror existing config patterns in `configs/` and dataset classes under `xcube/data/`.

## Where To Look Next
- `README.md`: authoritative setup + training/inference commands.
- `MISC.md`: data prep details and training tips.
- `configs/`: concrete YAMLs for datasets and stages.

