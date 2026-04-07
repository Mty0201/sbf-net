import argparse
from pathlib import Path

from _bootstrap import ensure_bf_edge_v3_root_on_path

ensure_bf_edge_v3_root_on_path()

from core.config import Stage2Config
from core.validation import validate_local_clusters, validate_cluster_contract
from core.local_clusters_core import (
    cluster_boundary_centers,
    export_clustered_boundary_centers_xyz,
    export_npz,
)
from utils.stage_io import load_boundary_centers


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for local clustering."""
    parser = argparse.ArgumentParser(description="BF Edge v3: build local clusters (bottom-up merge)")
    parser.add_argument("--input", type=str, required=True, help="Directory containing boundary_centers.npz")
    parser.add_argument("--output", type=str, default=None, help="Output directory. Default: same as input")
    parser.add_argument("--micro-eps-scale", type=float, default=3.5, help="Micro-cluster eps = scale * global_median_spacing")
    parser.add_argument("--merge-radius-scale", type=float, default=8.0, help="Merge radius = scale * global_median_spacing")
    parser.add_argument("--rescue-radius-scale", type=float, default=10.0, help="Rescue radius = scale * global_median_spacing")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Stage2Config:
    """Build Stage2Config from CLI arguments."""
    return Stage2Config(
        micro_eps_scale=float(args.micro_eps_scale),
        merge_radius_scale=float(args.merge_radius_scale),
        rescue_radius_scale=float(args.rescue_radius_scale),
    )


def run_scene(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    """Run one stage directory and export local clusters."""
    cfg = build_config(args)
    boundary_centers = load_boundary_centers(input_dir)
    local_clusters, meta = cluster_boundary_centers(
        boundary_centers=boundary_centers,
        config=cfg,
    )

    validate_local_clusters(local_clusters, num_boundary_centers=boundary_centers["center_coord"].shape[0])
    validate_cluster_contract(
        boundary_centers=boundary_centers,
        local_clusters=local_clusters,
        direction_cos_th=cfg.merge_direction_cos_th,
    )
    export_npz(output_dir / "local_clusters.npz", local_clusters)
    export_clustered_boundary_centers_xyz(
        boundary_centers=boundary_centers,
        local_clusters=local_clusters,
        output_dir=output_dir,
    )

    print("=" * 70)
    print("BF Edge v3 - build_local_clusters (bottom-up merge)")
    print(f"  input: {input_dir}")
    print(f"  output: {output_dir}")
    print(f"  centers: {meta['num_boundary_centers']}")
    print(f"  clusters: {meta['num_clusters']}")
    print(f"  assigned: {meta['num_assigned']}")
    print(f"  rescued: {meta['num_rescued']}")
    print(f"  noise: {meta['num_noise']}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output is not None else input_dir
    run_scene(input_dir=input_dir, output_dir=output_dir, args=args)


if __name__ == "__main__":
    main()
