import argparse
import csv
import importlib
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import fvdb
import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert (10, 128, 128, 128) voxel grids to XCube ShapeNet-style dataset format."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="frame",
        choices=["frame", "custom"],
        help="Data source mode. 'frame' reads FRAME VoxelLibrary directly; 'custom' uses --iterator.",
    )
    parser.add_argument(
        "--iterator",
        type=str,
        default=None,
        help=(
            "Iterator factory in the format 'module.submodule:function'. "
            "Required when --source custom. The function should return an iterator over samples."
        ),
    )
    parser.add_argument(
        "--frame_library_path",
        type=str,
        default=None,
        help="Path to FRAME voxel library directory containing manifest.json and voxel_data.zarr.",
    )
    parser.add_argument(
        "--frame_src_root",
        type=str,
        default="/home/tbm/FRAME/packages/frame-core/src",
        help="FRAME source root to append to sys.path when importing frame.VoxelLibrary.",
    )
    parser.add_argument(
        "--frame_query",
        type=str,
        default=None,
        help="Optional pandas query to filter FRAME structures, e.g. 'diameter_nm > 100'.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional max number of samples to convert.",
    )
    parser.add_argument(
        "--structures_parameters_path",
        type=str,
        default=None,
        help="Optional path to parallel FRAME zarr group containing parameter arrays.",
    )
    parser.add_argument(
        "--parameters_filename",
        type=str,
        default="parameters.csv",
        help="Top-level output parameter table filename.",
    )
    parser.add_argument(
        "--parameters_format",
        type=str,
        default="csv",
        choices=["csv", "parquet"],
        help="Format for exported parameter table.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Dataset root. Output is written to <output_root>/128/<category>/.",
    )
    parser.add_argument("--category", type=str, default="custom")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--expected_channels", type=int, default=10)
    parser.add_argument(
        "--occupancy_channel",
        type=int,
        default=-1,
        help="Occupancy channel index. Use -1 to derive occupancy from sum over all channels.",
    )
    parser.add_argument("--occupancy_threshold", type=float, default=0.5)
    parser.add_argument(
        "--normal_channels",
        type=str,
        default="1,2,3",
        help="Channel indices for xyz normal components.",
    )
    parser.add_argument(
        "--normal_mode",
        type=str,
        default="density_gradient",
        choices=["density_gradient", "channels"],
        help="How to compute normals: from field gradient or from explicit normal channels.",
    )
    parser.add_argument("--split_ratios", type=str, default="0.9,0.05,0.05")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save_extra_channels",
        action="store_true",
        help="Also save per-voxel extra channels as 'features' in the .pkl.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_iterator_factory(spec: str):
    if ":" not in spec:
        raise ValueError("--iterator must be in format 'module:function'.")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, func_name):
        raise ValueError(f"Function '{func_name}' not found in module '{module_name}'.")
    return getattr(module, func_name)


def resolve_iterator(args) -> Iterable:
    if args.source == "custom":
        if args.iterator is None:
            raise ValueError("--iterator is required when --source custom.")
        iterator_factory = load_iterator_factory(args.iterator)
        return iterator_factory()

    if args.frame_library_path is None:
        raise ValueError("--frame_library_path is required when --source frame.")

    frame_src_root = Path(args.frame_src_root)
    if str(frame_src_root) not in sys.path:
        sys.path.append(str(frame_src_root))

    try:
        from frame.storage import VoxelLibrary
    except ImportError as exc:
        raise ImportError(
            "Failed to import FRAME VoxelLibrary. Set --frame_src_root to FRAME core src path "
            "or install FRAME in the active environment."
        ) from exc

    library = VoxelLibrary(args.frame_library_path, mode="r")
    if args.frame_query:
        library_view = library.filter(args.frame_query)
    else:
        library_view = library

    parameters_reader = None
    if args.structures_parameters_path is not None:
        parameters_reader = build_structures_param_reader(args.structures_parameters_path)

    def frame_iterator():
        total = len(library_view)
        limit = total if args.max_samples is None else min(total, args.max_samples)
        if hasattr(library_view, "indices"):
            parent_indices = library_view.indices
        else:
            parent_indices = list(range(total))

        for idx in range(limit):
            vg = library_view[idx]
            parent_idx = int(parent_indices[idx])
            metadata = vg.metadata if vg.metadata is not None else {}
            sample_id = str(metadata.get("structure_id", args.start_index + idx))

            sample = {
                "id": sample_id,
                "grid": vg.data,
                "source_index": parent_idx,
            }

            if parameters_reader is not None:
                params = parameters_reader(parent_idx)
                params["source_index"] = parent_idx
                sample["parameters"] = params
            elif len(metadata) > 0:
                params = dict(metadata)
                params["source_index"] = parent_idx
                sample["parameters"] = params

            yield sample

    return frame_iterator()


def build_structures_param_reader(structures_parameters_path: str) -> Callable[[int], Dict[str, object]]:
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "zarr is required to read --structures_parameters_path. Install it in your environment."
        ) from exc

    param_group = zarr.open(str(structures_parameters_path), mode="r")
    keys = sorted(list(param_group.keys()))
    if len(keys) == 0:
        raise ValueError(f"No parameter arrays found at: {structures_parameters_path}")

    def to_python_scalar(value):
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def reader(index: int) -> Dict[str, object]:
        row = {}
        for key in keys:
            row[key] = to_python_scalar(param_group[key][index])
        return row

    return reader


def parse_sample(
    sample: Union[np.ndarray, torch.Tensor, Tuple[str, Union[np.ndarray, torch.Tensor]], dict],
    fallback_id: int,
) -> Tuple[str, np.ndarray]:
    if isinstance(sample, dict):
        if "grid" not in sample:
            raise ValueError("Dict sample must contain key 'grid'.")
        sample_id = str(sample.get("id", fallback_id))
        grid = sample["grid"]
    elif isinstance(sample, tuple) and len(sample) == 2:
        sample_id, grid = sample
        sample_id = str(sample_id)
    else:
        sample_id = str(fallback_id)
        grid = sample

    if isinstance(grid, torch.Tensor):
        grid = grid.detach().cpu().numpy()
    grid = np.asarray(grid)
    return sample_id, grid


def normalize_vectors(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / (torch.linalg.norm(v, dim=1, keepdim=True) + eps)


def build_xcube_sample(
    grid_10ch: np.ndarray,
    resolution: int,
    expected_channels: int,
    occupancy_channel: int,
    occupancy_threshold: float,
    normal_channels: Tuple[int, int, int],
    normal_mode: str,
    device: torch.device,
    save_extra_channels: bool,
):
    expected_shape = (expected_channels, resolution, resolution, resolution)
    if grid_10ch.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {grid_10ch.shape}.")

    if occupancy_channel == -1:
        occ = np.sum(grid_10ch, axis=0) > occupancy_threshold
    else:
        occ = grid_10ch[occupancy_channel] > occupancy_threshold
    if not np.any(occ):
        raise ValueError("No occupied voxels found after occupancy thresholding.")

    ijk_np = np.argwhere(occ).astype(np.int64)
    if normal_mode == "channels":
        normals_np = np.stack(
            [
                grid_10ch[normal_channels[0]][occ],
                grid_10ch[normal_channels[1]][occ],
                grid_10ch[normal_channels[2]][occ],
            ],
            axis=-1,
        )
    else:
        density = np.sum(grid_10ch, axis=0).astype(np.float32)
        grad_i, grad_j, grad_k = np.gradient(density)
        normals_np = np.stack([grad_i[occ], grad_j[occ], grad_k[occ]], axis=-1)

    voxel_size = 1.0 / 100.0
    origin = [voxel_size / 2.0] * 3

    ijk = torch.from_numpy(ijk_np).to(device=device, dtype=torch.long)
    ref_normal = torch.from_numpy(normals_np).to(device=device, dtype=torch.float32)

    ref_xyz = (ijk.to(torch.float32) + 0.5) * voxel_size

    sparse_grid = fvdb.sparse_grid_from_ijk(
        fvdb.JaggedTensor([ijk]), voxel_sizes=voxel_size, origins=origin
    )

    target_normal = sparse_grid.splat_trilinear(
        fvdb.JaggedTensor(ref_xyz), fvdb.JaggedTensor(ref_normal)
    )
    target_normal.jdata = normalize_vectors(target_normal.jdata)

    save_dict = {
        "points": sparse_grid.to("cpu"),
        "normals": target_normal.cpu(),
        "ref_xyz": ref_xyz.cpu(),
        "ref_normal": ref_normal.cpu(),
    }

    if save_extra_channels:
        features_np = grid_10ch[:, occ].transpose(1, 0)
        save_dict["features"] = torch.from_numpy(features_np).to(torch.float32)

    return save_dict


def write_splits(sample_ids, category_dir: Path, split_ratios: Tuple[float, float, float], seed: int):
    train_r, val_r, test_r = split_ratios
    ratio_sum = train_r + val_r + test_r
    if ratio_sum <= 0:
        raise ValueError("Split ratio sum must be > 0.")

    train_r, val_r, test_r = train_r / ratio_sum, val_r / ratio_sum, test_r / ratio_sum

    rng = np.random.default_rng(seed)
    shuffled = list(sample_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    n_test = n - n_train - n_val

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val : n_train + n_val + n_test]

    (category_dir / "train.lst").write_text("\n".join(train_ids) + ("\n" if train_ids else ""))
    (category_dir / "val.lst").write_text("\n".join(val_ids) + ("\n" if val_ids else ""))
    (category_dir / "test.lst").write_text("\n".join(test_ids) + ("\n" if test_ids else ""))


def write_parameters_table(
    rows: List[Dict[str, object]],
    output_root: Path,
    filename: str,
    file_format: str,
):
    if len(rows) == 0:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    table_path = output_root / filename

    if file_format == "csv":
        all_keys = sorted({k for row in rows for k in row.keys()})
        with table_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)
        return

    if file_format == "parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for --parameters_format parquet") from exc
        pd.DataFrame(rows).to_parquet(table_path, index=False)
        return

    raise ValueError(f"Unsupported parameters format: {file_format}")


def main():
    args = parse_args()

    if args.resolution != 128:
        raise ValueError("This converter currently supports only --resolution 128.")

    normal_channels = tuple(int(x) for x in args.normal_channels.split(","))
    if len(normal_channels) != 3:
        raise ValueError("--normal_channels must provide exactly 3 indices, e.g. '1,2,3'.")

    split_ratios = tuple(float(x) for x in args.split_ratios.split(","))
    if len(split_ratios) != 3:
        raise ValueError("--split_ratios must contain 3 values, e.g. '0.9,0.05,0.05'.")

    iterator: Iterable = resolve_iterator(args)

    output_category_dir = Path(args.output_root) / str(args.resolution) / args.category
    output_category_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    sample_ids = []
    parameter_rows = []

    for i, sample in enumerate(tqdm(iterator, desc="Converting")):
        sample_params = None
        if isinstance(sample, dict):
            sample_params = sample.get("parameters")

        sample_id, grid_10ch = parse_sample(sample, args.start_index + i)
        target_path = output_category_dir / f"{sample_id}.pkl"

        if target_path.exists() and not args.overwrite:
            sample_ids.append(sample_id)
            if sample_params is not None:
                parameter_rows.append({"sample_id": sample_id, **sample_params})
            continue

        converted = build_xcube_sample(
            grid_10ch=grid_10ch,
            resolution=args.resolution,
            expected_channels=args.expected_channels,
            occupancy_channel=args.occupancy_channel,
            occupancy_threshold=args.occupancy_threshold,
            normal_channels=normal_channels,
            normal_mode=args.normal_mode,
            device=device,
            save_extra_channels=args.save_extra_channels,
        )
        torch.save(converted, target_path)
        sample_ids.append(sample_id)
        if sample_params is not None:
            parameter_rows.append({"sample_id": sample_id, **sample_params})

    if len(sample_ids) == 0:
        raise RuntimeError("No samples were converted.")

    write_splits(sample_ids, output_category_dir, split_ratios, args.seed)
    write_parameters_table(
        rows=parameter_rows,
        output_root=Path(args.output_root),
        filename=args.parameters_filename,
        file_format=args.parameters_format,
    )
    print(f"Wrote {len(sample_ids)} samples to {output_category_dir}")


if __name__ == "__main__":
    main()