# XCube Build System Modernization Plan

## Context

XCube currently uses conda (`environment.yml`) for dependency management with Python 3.10, PyTorch 2.2.0, CUDA 12.1, and PyTorch Lightning 1.9.4. The goal is to:
- Replace conda with pip/PyPI entirely via a modern `pyproject.toml`
- Upgrade to Python 3.12 + CUDA 12.6
- Upgrade PyTorch Lightning from 1.9.4 to 2.x
- Simplify the fvdb build process (keep fvdb, but document as a clean pre-requisite)
- Drop torchsparse (confirmed unused - not imported anywhere in source code)
- Remove `environment.yml` and `environment_py311.yml`

---

## Step 1: Create root `pyproject.toml`

**File**: `/home/tbm/XCube/pyproject.toml` (new)

```toml
[project]
name = "xcube"
version = "0.1.0"
description = "Large-scale 3D generative modeling with sparse voxel hierarchies"
requires-python = ">=3.12"
dependencies = [
    # Core ML
    "torch>=2.6.0",
    "pytorch-lightning>=2.4,<3",
    "numpy<2.0.0",
    "tensorboard",

    # PyG ecosystem (scatter only - PyG itself not directly imported)
    "torch-scatter",

    # Config & serialization
    "omegaconf",
    "flatten-dict",
    "pyyaml",

    # Logging & monitoring
    "wandb",
    "loguru",
    "pynvml",
    "rich",

    # Data & scientific
    "pandas",
    "scipy",
    "matplotlib",
    "tqdm",

    # 3D / point cloud
    "point_cloud_utils>=0.29.5",
    "trimesh",
    "polyscope",

    # ML utilities
    "einops",
    "transformers",

    # Experiment framework
    "python-pycg",

    # Misc
    "randomname",
    "gdown",
    "gitpython",
    "packaging",
    "linkify-it-py",
    "ipython",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-benchmark",
    "pytest-cov",
]
docs = [
    "sphinx>=7.0.0",
    "sphinx_rtd_theme",
    "myst-parser",
]

[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["xcube*"]
```

**Key decisions**:
- `torch-scatter` requires a find-links URL at install time (see install script, Step 6)
- `point_cloud_utils` pin relaxed to `>=0.29.5` (test if latest works)
- `numpy<2.0.0` constraint kept (fvdb and other native extensions require it)
- fvdb is NOT listed (pre-requisite, installed separately)
- nksr is NOT listed (built separately from ext/nksr-cuda/)
- torchsparse DROPPED (not imported anywhere)
- `torch-geometric` DROPPED (not directly imported; torch-scatter is the only PyG dep actually used)
- `sparsehash` DROPPED (only needed for building torchsparse from source)
- Compiler toolchain (gcc, cmake, ninja, cuda-nvcc) documented as system pre-requisites, not pip deps

---

## Step 2: Upgrade PyTorch Lightning 1.9.4 → 2.x

This requires code changes in 3 files.

### 2a. `train.py` — Major changes

**Lines 33, 35**: Remove deleted imports
```python
# REMOVE:
from pytorch_lightning.plugins.training_type.dp import DataParallelPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException

# REPLACE WITH:
from lightning.pytorch.utilities.exceptions import MisconfigurationException
```

**Lines 40-43**: Remove version-gated Callback import (PL 2.x always has it at top level)
```python
# SIMPLIFY TO:
from pytorch_lightning.callbacks import Callback
```

**Lines 57-81**: Remove `CustomizedDataParallel` and `CustomizedDataParallelPlugin` classes entirely. DataParallel is removed in Lightning 2.x (DDP is the standard). If DP is still needed, implement as a custom Strategy.

**Line 167**: `pl.Trainer.add_argparse_args()` removed in 2.x
- Replace with manual argparse or use LightningCLI
- Recommended: extract Trainer args into explicit argparse arguments

**Lines 216-236**: Remove all `version.parse(pl.__version__)` branching. Simplify to single code path for PL 2.x:
```python
program_args.strategy = 'ddp'
program_args.accelerator = "gpu"
program_args.devices = program_args.gpus
```

**Lines 264-274**: Remove the `accelerator_plugins` block (DP plugin removed). For DDP:
```python
from pytorch_lightning.strategies import DDPStrategy
strategy = DDPStrategy(find_unused_parameters=False)
```

**Line 358**: `pl.Trainer.from_argparse_args()` removed in 2.x
- Replace with explicit `pl.Trainer(...)` constructor, passing args directly

**Line 365**: `plugins=accelerator_plugins` → `strategy=strategy`

**Line 367**: `precision=program_args.model_precision` — In PL 2.x, integer precision values like `32` should become `"32-true"` and `16` should become `"16-mixed"`.

### 2b. `test.py` — Moderate changes

**Line 37**: `pl.Trainer.add_argparse_args()` — same removal as train.py
**Line 74**: `pl.Trainer.from_argparse_args()` — same replacement as train.py

### 2c. `xcube/models/base_model.py` — Targeted changes

**Line 211**: `from pytorch_lightning.utilities.grads import grad_norm` — removed in PL 2.x
- Replace with inline implementation:
```python
# Inline replacement for removed grad_norm utility
norms = {f"grad_{norm_type}_norm/{name}": p.grad.data.norm(norm_type)
         for name, p in self.named_parameters() if p.grad is not None}
```

**Lines 256-259**: `from pytorch_lightning.utilities.logger import (_convert_params, _flatten_dict)` — removed in PL 2.x
- These are used only for TensorBoard hparams logging. Replace with manual dict flattening or use `flatten-dict` package (already a dependency).

**Line 242**: `on_before_optimizer_step(self, optimizer, optimizer_idx)` → remove `optimizer_idx` parameter (PL 2.x signature is `on_before_optimizer_step(self, optimizer)`).

**Line 415**: `on_test_batch_start(self, batch, batch_idx, dataloader_idx)` → remove `dataloader_idx` (PL 2.x drops it).

**Line 418**: `on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx)` → remove `dataloader_idx`.

---

## Step 3: Simplify fvdb installation

**Keep**: `assets/setup.py` (the fvdb build script) — this stays as-is since it's a pip-installable source build.

**Create**: `scripts/install_fvdb.sh` (new helper script)
```bash
#!/bin/bash
# Simplified fvdb installation for pip-based environments
set -e

FVDB_DIR="${FVDB_DIR:-/tmp/openvdb}"

if [ ! -d "$FVDB_DIR" ]; then
    git clone https://github.com/AcademySoftwareFoundation/openvdb.git "$FVDB_DIR"
fi

cd "$FVDB_DIR"
git fetch origin pull/1808/head:feature/fvdb
git checkout feature/fvdb

# Use XCube's custom setup.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cp "$SCRIPT_DIR/assets/setup.py" fvdb/setup.py

cd fvdb
pip install .
```

**Create**: `scripts/install_nksr.sh` (new helper script)
```bash
#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../ext/nksr-cuda"
pip install -e .
```

---

## Step 4: Create unified install script

**Create**: `scripts/install.sh` (new)

Documents the full pip-based install flow, replacing conda:
```bash
#!/bin/bash
set -e

# System pre-requisites (documented, not automated):
# - Python 3.12
# - CUDA 12.6 toolkit (nvcc, libraries)
# - GCC 11+ (for CUDA extension compilation)
# - cmake, ninja-build

# 1. Install PyTorch with CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2. Install torch-scatter (requires matching PyTorch/CUDA)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

# 3. Install XCube and all pip dependencies
pip install -e ".[dev]"

# 4. Install fvdb from source
bash scripts/install_fvdb.sh

# 5. Install NKSR CUDA extension
bash scripts/install_nksr.sh
```

---

## Step 5: Update `ext/nksr-cuda/setup.py` and `ext/nksr-cuda/pyproject.toml`

Minimal changes — update Python requirement:
- `python_requires='>=3.12'` (was `>=3.7`)

---

## Step 6: Remove conda files

**Delete**:
- `environment.yml`
- `environment_py311.yml`

---

## Step 7: Update `datagen/frame-geo/pyproject.toml`

Update Python requirement to match:
```toml
requires-python = ">=3.12"
```

---

## Step 8: Update documentation

### `README.md` — Replace conda setup instructions with:

**System pre-requisites section:**
- Python 3.12+
- CUDA 12.6 toolkit with nvcc
- GCC 11+ / G++ 11+
- cmake, ninja-build
- NVIDIA GPU (Ampere or newer)

**Installation section:**
```bash
# Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Run the install script
bash scripts/install.sh
```

### `AGENTS.md` — Update the "Environment Setup" section to reflect pip-based workflow.

### `CLAUDE.md` — No changes needed.

---

## Files to modify (summary)

| File | Action |
|------|--------|
| `pyproject.toml` | **CREATE** — root package definition |
| `train.py` | **EDIT** — Lightning 2.x migration (~50 lines changed) |
| `test.py` | **EDIT** — Lightning 2.x migration (~10 lines changed) |
| `xcube/models/base_model.py` | **EDIT** — Lightning 2.x API changes (~15 lines changed) |
| `scripts/install.sh` | **CREATE** — unified install script |
| `scripts/install_fvdb.sh` | **CREATE** — fvdb build helper |
| `scripts/install_nksr.sh` | **CREATE** — nksr build helper |
| `ext/nksr-cuda/setup.py` | **EDIT** — update python_requires |
| `datagen/frame-geo/pyproject.toml` | **EDIT** — update python_requires |
| `README.md` | **EDIT** — replace conda instructions with pip |
| `AGENTS.md` | **EDIT** — update environment setup section |
| `environment.yml` | **DELETE** |
| `environment_py311.yml` | **DELETE** |

---

## Verification

1. **Clean install test**: Create fresh Python 3.12 venv, run `scripts/install.sh`, verify all imports work
2. **fvdb build**: Verify `import fvdb` works after running `scripts/install_fvdb.sh`
3. **NKSR build**: Verify `import nksr` works after running `scripts/install_nksr.sh`
4. **Lightning migration**: Run `python -c "import train"` and check for import errors
5. **Inference smoke test**: Run a minimal inference script to verify the full stack works
6. **frame-geo tests**: Run `cd datagen/frame-geo && pytest` to verify that sub-package still works

---

## Risks and mitigations

- **numpy<2.0.0 constraint**: fvdb and other native extensions may break with numpy 2.x. Keep the pin.
- **torch-scatter wheel availability**: PyG hosts wheels at data.pyg.org. If the exact PyTorch+CUDA combo isn't available, torch-scatter must be built from source (requires GCC). The install script uses `-f` find-links to handle this.
- **point_cloud_utils version**: Relaxed pin from `==0.29.5` to `>=0.29.5`. If latest version causes issues, re-pin to `==0.29.5`.
- **fvdb C++20 requirement**: fvdb's setup.py auto-detects CUDA version and uses C++20 for CUDA 12+. This should work with system CUDA 12.6 toolkit.
- **Lightning 2.x Trainer API**: The biggest code change risk. `Trainer.from_argparse_args()` and `Trainer.add_argparse_args()` are removed — must be replaced with explicit Trainer construction.
