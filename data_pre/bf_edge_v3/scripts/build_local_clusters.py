import argparse
from pathlib import Path

from _bootstrap import ensure_bf_edge_v3_root_on_path

ensure_bf_edge_v3_root_on_path()

from core.config import Stage2Config
from core.local_clusters_core import (
    cluster_boundary_centers,
    export_clustered_boundary_centers_xyz,
    export_npz,
)
from utils.stage_io import load_boundary_centers


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for coarse local clustering."""
    parser = argparse.ArgumentParser(description="BF Edge v3: build coarse local clusters")
    parser.add_argument("--input", type=str, required=True, help="Directory containing boundary_centers.npz")
    parser.add_argument("--output", type=str, default=None, help="Output directory. Default: same as input")
    parser.add_argument("--eps", type=float, default=0.08, help="DBSCAN eps in scene units")
    parser.add_argument("--min-samples", type=int, default=8, help="DBSCAN min_samples")
    parser.add_argument("--denoise-knn", type=int, default=8, help="kNN size used by light cluster-internal denoise")
    parser.add_argument("--sparse-distance-ratio", type=float, default=1.75, help="Remove points whose local spacing exceeds median * ratio")
    parser.add_argument("--sparse-mad-scale", type=float, default=3.0, help="Also require spacing to exceed median + scale * MAD")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Stage2Config:
    """Build Stage2Config from CLI arguments."""
    return Stage2Config(
        eps=float(args.eps),
        min_samples=int(args.min_samples),
        denoise_knn=int(args.denoise_knn),
        sparse_distance_ratio=float(args.sparse_distance_ratio),
        sparse_mad_scale=float(args.sparse_mad_scale),
    )


def run_scene(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    """Run one stage directory and export local clusters."""
    cfg = build_config(args)
    boundary_centers = load_boundary_centers(input_dir)
    local_clusters, meta = cluster_boundary_centers(
        boundary_centers=boundary_centers,
        eps=cfg.eps,
        min_samples=cfg.min_samples,
        denoise_knn=cfg.denoise_knn,
        sparse_distance_ratio=cfg.sparse_distance_ratio,
        sparse_mad_scale=cfg.sparse_mad_scale,
    )

    export_npz(output_dir / "local_clusters.npz", local_clusters)
    export_clustered_boundary_centers_xyz(
        boundary_centers=boundary_centers,
        local_clusters=local_clusters,
        output_dir=output_dir,
    )

    print("=" * 70)
    print("BF Edge v3 - build_local_clusters")
    print(f"  input: {input_dir}")
    print(f"  output: {output_dir}")
    print(f"  centers: {meta['num_boundary_centers']}")
    print(f"  clusters: {meta['num_clusters']}")
    print(f"  assigned: {meta['num_assigned']}")
    print(f"  trigger_clusters: {meta['num_trigger_clusters']}")
    print(f"  removed_by_denoise: {meta['num_removed_by_denoise']}")
    print(f"  noise: {meta['num_noise']}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output is not None else input_dir
    run_scene(input_dir=input_dir, output_dir=output_dir, args=args)


if __name__ == "__main__":
    main()
