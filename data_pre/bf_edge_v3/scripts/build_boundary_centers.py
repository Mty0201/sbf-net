import argparse
from pathlib import Path

from _bootstrap import ensure_bf_edge_v3_root_on_path

ensure_bf_edge_v3_root_on_path()

from core.boundary_centers_core import (
    build_boundary_centers,
    export_boundary_centers_npz,
    export_candidate_xyz,
    export_centers_xyz,
)
from utils.stage_io import load_scene


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for boundary center construction."""
    parser = argparse.ArgumentParser(description="BF Edge v3: build boundary centers")
    parser.add_argument("--scene", type=str, required=True, help="Scene directory containing coord.npy and segment.npy")
    parser.add_argument("--output", type=str, default=None, help="Output directory. Default: <scene>/bf_edge_v3")
    parser.add_argument("--k", type=int, default=32, help="kNN size without self")
    parser.add_argument("--min-cross-ratio", type=float, default=0.15, help="Minimum fraction of cross-semantic neighbors")
    parser.add_argument("--min-side-points", type=int, default=4, help="Minimum number of points on each side of a semantic pair")
    parser.add_argument("--ignore-index", type=int, default=-1, help="Semantic ignore label")
    return parser.parse_args()


def run_scene(scene_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    """Run one scene and export boundary center artifacts."""
    scene = load_scene(scene_dir)
    candidates, centers_payload, meta = build_boundary_centers(
        scene=scene,
        k=int(args.k),
        min_cross_ratio=float(args.min_cross_ratio),
        min_side_points=int(args.min_side_points),
        ignore_index=int(args.ignore_index),
    )

    export_boundary_centers_npz(output_dir / "boundary_centers.npz", centers_payload)
    export_centers_xyz(centers_payload=centers_payload, output_dir=output_dir)
    export_candidate_xyz(scene=scene, candidates=candidates, output_dir=output_dir)

    print("=" * 70)
    print("BF Edge v3 - build_boundary_centers")
    print(f"  scene: {scene_dir}")
    print(f"  output: {output_dir}")
    print(f"  points: {meta['num_points']}")
    print(f"  candidates: {meta['num_candidates']}")
    print(f"  centers: {meta['num_centers']}")
    print(f"  semantic_pairs: {meta['num_semantic_pairs']}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene)
    output_dir = Path(args.output) if args.output is not None else scene_dir / "bf_edge_v3"
    run_scene(scene_dir=scene_dir, output_dir=output_dir, args=args)


if __name__ == "__main__":
    main()
