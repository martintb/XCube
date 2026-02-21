#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

command -v uv &>/dev/null || { echo "Error: uv not installed. See https://docs.astral.sh/uv/"; exit 1; }

# 1. Create venv
uv venv "$REPO_ROOT/.venv" --python 3.12
source "$REPO_ROOT/.venv/bin/activate"

uv pip install pip setuptools wheel

# 2. Install PyTorch 2.8.0 + CUDA 12.6
uv pip install torch==2.8.0 torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu126

# 3. Install torch-scatter (or replace with native ops if wheels unavailable)
uv pip install torch-scatter \
    --find-links "https://data.pyg.org/whl/torch-2.8.0+cu126.html" || \
    uv pip install torch-scatter

# 4. Build fvdb-core from source
uv pip install scikit-build-core cmake ninja
FVDB_DIR="${FVDB_DIR:-$REPO_ROOT/.build/fvdb-core}"
[ -d "$FVDB_DIR" ] || git clone https://github.com/openvdb/fvdb-core.git "$FVDB_DIR"
pushd "$FVDB_DIR"
export CPM_SOURCE_CACHE="$HOME/.cache/CPM"
# CMake's FindCUDA needs CUDA_TOOLKIT_ROOT_DIR; CUDA_HOME is set by the cuda module
export CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME:-${CUDA_DIR:-/usr/local/cuda}}"
echo "Using CUDA_TOOLKIT_ROOT_DIR=$CUDA_TOOLKIT_ROOT_DIR"
# Wipe any stale CMakeCache that may have CUDA_CUDART_LIBRARY set to -NOTFOUND
# from a previous run with the wrong toolkit root.
find build -name "CMakeCache.txt" -delete 2>/dev/null || true
# fvdb-core's get_torch.cmake overrides CUDA_TOOLKIT_ROOT_DIR with
# $CONDA_PREFIX/targets/x86_64-linux when CONDA_PREFIX is set, which breaks
# the build on machines with a system miniforge that has no CUDA libraries.
unset CONDA_PREFIX
# Patch: fvdb-core gsplat files use ::cuda::std::exp / ::cuda::std::fabs etc.,
# but cuda/std/cmath cannot be included alongside PyTorch's
# -D__CUDA_NO_HALF_CONVERSIONS__ flag (triggers fp16 conversion errors in
# cuda/std/__cuda/cmath_nvfp16.h). Replace the namespace-qualified calls with
# plain global-scope intrinsics available in all CUDA device code.
# GaussianProjectionUT.cu ships with an explicit #include <cuda/std/cmath>
# that must also be removed for the same reason.
GSPLAT_DIR="src/fvdb/detail/ops/gsplat"
sed -i '/#include <cuda\/std\/cmath>/d' "$GSPLAT_DIR/GaussianProjectionUT.cu"
for f in GaussianMCMCAddNoise.cu GaussianMCMCRelocation.cu \
          GaussianProjectionBackward.cu GaussianProjectionForward.cu \
          GaussianProjectionUT.cu; do
    sed -i 's/::cuda::std::/\:\:/g' "$GSPLAT_DIR/$f"
done
uv pip install --no-build-isolation .
popd

# 5. Install XCube (editable)
uv pip install -e "$REPO_ROOT"

# 6. Build nksr CUDA extensions
pushd "$REPO_ROOT/ext/nksr-cuda"
uv pip install --no-build-isolation -e .
popd

# 7. Verify
python -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
import fvdb; print('fvdb OK')
import pytorch_lightning as pl; print(f'Lightning {pl.__version__}')
import nksr; print(f'nksr {nksr.__version__}')
print('All imports successful!')
"
