"""Add edge_support_id.npy to an existing compact edge dataset.

This is a supplementary script for Route A experiments.  The compact edge
dataset built by build_edge_dataset_v3.py does not include edge_support_id.npy.
This script re-runs the nearest-support lookup for each scene using the
supports.npz from the SOURCE dataset and writes edge_support_id.npy into the
TARGET (compact edge) dataset.

Usage:
    python add_support_id_to_edge_dataset.py \
        --source /path/to/source_dataset \
        --target /path/to/edge_dataset \
        [--support-radius 0.08] \
        [--ignore-index -1]

Prerequisites:
    - source dataset must contain supports.npz for every scene.
    - target dataset must contain edge.npy (6-column) for every scene.
    - source and target must share the same training/validation split structure.

Output:
    Writes edge_support_id.npy (int32, shape (N,), -1 for invalid) alongside
    existing edge.npy in each target scene directory.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Inline path setup (tools/ is one level below scripts/)
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from core.pointwise_core import build_pointwise_edge_supervision, load_supports
from utils.stage_io import load_scene


SPLITS = ("training", "validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add edge_support_id.npy to an existing compact edge dataset"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source dataset root that contains supports.npz per scene",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target compact edge dataset root that already has edge.npy",
    )
    parser.add_argument(
        "--support-radius",
        dest="support_radius",
        type=float,
        default=0.08,
        help="Must match the radius used when the edge dataset was built (default: 0.08)",
    )
    parser.add_argument(
        "--ignore-index",
        dest="ignore_index",
        type=int,
        default=-1,
        help="Semantic ignore label (default: -1)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing edge_support_id.npy files (default: skip)",
    )
    return parser.parse_args()


def collect_paired_scene_dirs(
    source_root: Path, target_root: Path
) -> list[tuple[Path, Path]]:
    """Return (source_scene_dir, target_scene_dir) pairs for all valid scenes."""
    pairs: list[tuple[Path, Path]] = []
    for split in SPLITS:
        src_split = source_root / split
        tgt_split = target_root / split
        if not src_split.is_dir() or not tgt_split.is_dir():
            continue
        for src_scene in sorted(p for p in src_split.iterdir() if p.is_dir()):
            tgt_scene = tgt_split / src_scene.name
            if (
                (src_scene / "coord.npy").exists()
                and (src_scene / "segment.npy").exists()
                and (src_scene / "supports.npz").exists()
                and (tgt_scene / "edge.npy").exists()
            ):
                pairs.append((src_scene, tgt_scene))
    return pairs


def run_scene(
    src_scene_dir: Path,
    tgt_scene_dir: Path,
    args: argparse.Namespace,
) -> None:
    out_path = tgt_scene_dir / "edge_support_id.npy"
    if out_path.exists() and not args.overwrite:
        print(f"  skip (already exists): {out_path}")
        return

    scene = load_scene(src_scene_dir)
    supports = load_supports(src_scene_dir)
    payload, meta = build_pointwise_edge_supervision(
        scene=scene,
        supports=supports,
        support_radius=float(args.support_radius),
        ignore_index=int(args.ignore_index),
    )

    np.save(out_path, payload["edge_support_id"])
    print(
        f"  wrote: {out_path} "
        f"(valid={meta['num_valid_points']}/{meta['num_points']})"
    )


def main() -> None:
    args = parse_args()
    source_root = Path(args.source)
    target_root = Path(args.target)

    pairs = collect_paired_scene_dirs(source_root, target_root)
    if not pairs:
        print(
            "No matching scene pairs found.  Check that source has supports.npz "
            "and target has edge.npy under matching training/validation splits."
        )
        return

    print(f"Found {len(pairs)} scene pair(s) to process.")
    for src_scene, tgt_scene in pairs:
        print(f"Processing: {src_scene.name}")
        run_scene(src_scene_dir=src_scene, tgt_scene_dir=tgt_scene, args=args)

    print("Done.")


if __name__ == "__main__":
    main()
