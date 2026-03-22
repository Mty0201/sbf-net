import argparse
import shutil
from pathlib import Path

import numpy as np

from _bootstrap import ensure_bf_edge_v3_root_on_path

ensure_bf_edge_v3_root_on_path()

from core.pointwise_core import (
    build_pointwise_edge_supervision,
    export_edge_supervision_xyz,
    load_supports,
)
from utils.stage_io import load_scene


SPLITS = ("training", "validation")
BASE_FILES = ("coord.npy", "color.npy", "normal.npy", "segment.npy")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset-level edge dataset construction."""
    parser = argparse.ArgumentParser(description="BF Edge v3: build compact edge dataset")
    parser.add_argument("--input", type=str, required=True, help="Source dataset root containing supports.npz")
    parser.add_argument("--output", type=str, required=True, help="Target edge dataset root")
    parser.add_argument(
        "--support-radius",
        "--max-edge-dist",
        dest="support_radius",
        type=float,
        default=0.08,
        help="Support supervision radius for boundary snapping (legacy alias: --max-edge-dist)",
    )
    parser.add_argument("--ignore-index", type=int, default=-1, help="Semantic ignore label")
    return parser.parse_args()


def collect_scene_dirs(dataset_root: Path) -> list[Path]:
    """Collect sample directories under training/validation."""
    scene_dirs: list[Path] = []
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if (scene_dir / "coord.npy").exists() and (scene_dir / "segment.npy").exists() and (scene_dir / "supports.npz").exists():
                scene_dirs.append(scene_dir)
    return scene_dirs


def build_edge_array(payload: dict) -> np.ndarray:
    """Build compact training-friendly edge.npy payload."""
    edge = np.zeros((payload["edge_vec"].shape[0], 5), dtype=np.float32)
    edge[:, 0:3] = payload["edge_vec"].astype(np.float32)
    edge[:, 3] = payload["edge_support"].astype(np.float32)
    edge[:, 4] = payload["edge_valid"].astype(np.float32)
    return edge


def copy_base_files(src_scene_dir: Path, dst_scene_dir: Path) -> None:
    """Copy original base arrays into the edge dataset."""
    dst_scene_dir.mkdir(parents=True, exist_ok=True)
    for name in BASE_FILES:
        src_path = src_scene_dir / name
        if src_path.exists():
            shutil.copy2(src_path, dst_scene_dir / name)


def run_scene(src_scene_dir: Path, dst_scene_dir: Path, args: argparse.Namespace) -> None:
    """Build compact edge dataset outputs for one scene."""
    scene = load_scene(src_scene_dir)
    supports = load_supports(src_scene_dir)
    payload, meta = build_pointwise_edge_supervision(
        scene=scene,
        supports=supports,
        support_radius=float(args.support_radius),
        ignore_index=int(args.ignore_index),
    )

    copy_base_files(src_scene_dir=src_scene_dir, dst_scene_dir=dst_scene_dir)
    np.save(dst_scene_dir / "edge.npy", build_edge_array(payload))
    export_edge_supervision_xyz(scene=scene, payload=payload, output_dir=dst_scene_dir)

    print("=" * 70)
    print("BF Edge v3 - build_edge_dataset_v3")
    print(f"  input: {src_scene_dir}")
    print(f"  output: {dst_scene_dir}")
    print(f"  points: {meta['num_points']}")
    print(f"  valid_points: {meta['num_valid_points']}")
    print(f"  support_radius: {meta['support_radius']:.6f}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    scene_dirs = collect_scene_dirs(input_root)
    if not scene_dirs:
        print("No valid training/validation scene containing coord.npy, segment.npy and supports.npz was found.")
        return

    for src_scene_dir in scene_dirs:
        rel_path = src_scene_dir.relative_to(input_root)
        run_scene(src_scene_dir=src_scene_dir, dst_scene_dir=output_root / rel_path, args=args)


if __name__ == "__main__":
    main()
