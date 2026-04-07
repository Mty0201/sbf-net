"""Generate support_geometry.xyz for every validation scene.

Usage:
    python data_pre/bf_edge_v3/scripts/gen_val_support_geometry.py \
        --input /mnt/e/WSL/data/BF_edge_chunk_npy

Only touches the validation/ split.  Produces support_geometry.xyz in each
scene directory; does NOT write supports.npz or remove any existing files.
"""

import argparse
from pathlib import Path

from _bootstrap import ensure_bf_edge_v3_root_on_path

ensure_bf_edge_v3_root_on_path()

from core.boundary_centers_core import build_boundary_centers
from core.config import Stage1Config, Stage2Config, Stage3Config
from core.local_clusters_core import cluster_boundary_centers
from core.supports_core import build_supports_payload
from core.supports_export import export_support_geometry_xyz
from utils.stage_io import load_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate support_geometry.xyz for validation split")
    parser.add_argument("--input", type=str, required=True, help="Dataset root containing validation/")
    return parser.parse_args()


def run_scene(scene_dir: Path, s1: Stage1Config, s2: Stage2Config, params: dict) -> None:
    scene = load_scene(scene_dir)
    _, bc, _ = build_boundary_centers(
        scene=scene,
        k=s1.k,
        min_cross_ratio=s1.min_cross_ratio,
        min_side_points=s1.min_side_points,
        ignore_index=s1.ignore_index,
    )
    lc, _ = cluster_boundary_centers(boundary_centers=bc, config=s2)
    sp, meta, _ = build_supports_payload(
        boundary_centers=bc, local_clusters=lc, params=params,
    )
    export_support_geometry_xyz(sp, output_dir=scene_dir)
    print(f"  {scene_dir.name}: {meta['num_supports']} supports "
          f"(line={meta['support_type_hist']['line']}, "
          f"polyline={meta['support_type_hist']['polyline']})")


def main() -> None:
    args = parse_args()
    val_dir = Path(args.input) / "validation"
    scenes = sorted(
        p for p in val_dir.iterdir()
        if p.is_dir() and (p / "coord.npy").exists() and (p / "segment.npy").exists()
    )
    print(f"Found {len(scenes)} validation scenes")

    s1 = Stage1Config()
    s2 = Stage2Config()
    params = Stage3Config().to_runtime_dict()

    todo = [s for s in scenes if not (s / "support_geometry.xyz").exists()]
    print(f"  {len(scenes) - len(todo)} already done, {len(todo)} remaining")

    for i, scene_dir in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}]", end="")
        run_scene(scene_dir, s1, s2, params)


if __name__ == "__main__":
    main()
