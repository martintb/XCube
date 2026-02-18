#!/bin/bash
set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate xcube

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python \
    "$SCRIPT_DIR/voxel10_to_xcube.py" \
    --source frame \
    --frame_library_path /wrk/tbm/frame_data/libraries/lib_17adb8df5a19/voxels.zarr/voxels.zarr/voxel_data.zarr \
    --structures_parameters_path /wrk/tbm/frame_data/libraries/lib_17adb8df5a19/structures.zarr/parameters \
    --output_root /wrk/tbm/frame_data/library_xcube/lib_17adb8df5a19 \
    --category lnp \
    --occupancy_channel -1 \
    --occupancy_threshold 0.01