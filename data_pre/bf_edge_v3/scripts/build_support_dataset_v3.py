import argparse
from pathlib import Path

from _bootstrap import ensure_bf_edge_v3_root_on_path

ensure_bf_edge_v3_root_on_path()

from core.boundary_centers_core import build_boundary_centers
from core.config import Stage1Config, Stage2Config, Stage3Config
from core.local_clusters_core import cluster_boundary_centers
from core.validation import validate_boundary_centers, validate_local_clusters, validate_supports
from core.supports_core import (
    build_supports_payload,
)
from core.supports_export import (
    export_npz,
    export_support_geometry_xyz,
)
from utils.stage_io import load_scene


SPLITS = ("training", "validation")
INTERMEDIATE_FILES_TO_REMOVE = (
    "boundary_centers.npz",
    "boundary_centers.xyz",
    "boundary_candidates.xyz",
    "local_clusters.npz",
    "clustered_boundary_centers.xyz",
    "trigger_group_classes.xyz",
    "edge_dist.npy",
    "edge_dir.npy",
    "edge_valid.npy",
    "edge_mask.npy",
    "edge_support_id.npy",
    "edge_vec.npy",
    "edge_support.npy",
    "edge_strength.npy",
    "edge_supervision.xyz",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset-level support generation."""
    parser = argparse.ArgumentParser(description="BF Edge v3: build dataset-level supports")
    parser.add_argument("--input", type=str, required=True, help="Dataset root containing training/validation")
    parser.add_argument("--k", type=int, default=32, help="kNN size without self for boundary center stage")
    parser.add_argument("--min-cross-ratio", type=float, default=0.15, help="Minimum fraction of cross-semantic neighbors")
    parser.add_argument("--min-side-points", type=int, default=4, help="Minimum number of points on each side of a semantic pair")
    parser.add_argument("--ignore-index", type=int, default=-1, help="Semantic ignore label")
    parser.add_argument("--eps", type=float, default=0.08, help="DBSCAN eps in scene units")
    parser.add_argument("--min-samples", type=int, default=8, help="DBSCAN min_samples")
    parser.add_argument("--denoise-knn", type=int, default=8, help="kNN size used by light cluster-internal denoise")
    parser.add_argument("--sparse-distance-ratio", type=float, default=1.75, help="Remove points whose local spacing exceeds median * ratio")
    parser.add_argument("--sparse-mad-scale", type=float, default=3.0, help="Also require spacing to exceed median + scale * MAD")
    parser.add_argument("--line-residual-th", type=float, default=0.01, help="Line mean residual threshold before falling back to polyline")
    parser.add_argument("--min-cluster-size", type=int, default=8, help="Minimum cluster size for support fitting")
    parser.add_argument("--max-polyline-vertices", type=int, default=32, help="Maximum polyline vertex count for non-trigger fallback")
    return parser.parse_args()


def collect_scene_dirs(dataset_root: Path) -> list[Path]:
    """Collect sample directories under training/validation."""
    scene_dirs: list[Path] = []
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if (scene_dir / "coord.npy").exists() and (scene_dir / "segment.npy").exists():
                scene_dirs.append(scene_dir)
    return scene_dirs


def build_stage1_config(args: argparse.Namespace) -> Stage1Config:
    """Build Stage1Config from CLI arguments."""
    return Stage1Config(
        k=int(args.k),
        min_cross_ratio=float(args.min_cross_ratio),
        min_side_points=int(args.min_side_points),
        ignore_index=int(args.ignore_index),
    )


def build_stage2_config(args: argparse.Namespace) -> Stage2Config:
    """Build Stage2Config from CLI arguments."""
    return Stage2Config(
        eps=float(args.eps),
        min_samples=int(args.min_samples),
        denoise_knn=int(args.denoise_knn),
        sparse_distance_ratio=float(args.sparse_distance_ratio),
        sparse_mad_scale=float(args.sparse_mad_scale),
    )


def build_stage3_config(args: argparse.Namespace) -> Stage3Config:
    """Build Stage3Config from CLI arguments."""
    return Stage3Config(
        line_residual_th=float(args.line_residual_th),
        min_cluster_size=int(args.min_cluster_size),
        max_polyline_vertices=int(args.max_polyline_vertices),
    )


def cleanup_scene_dir(scene_dir: Path) -> None:
    """Remove intermediate or stale outputs not kept after support build."""
    for name in INTERMEDIATE_FILES_TO_REMOVE:
        path = scene_dir / name
        if path.exists():
            path.unlink()


def run_scene(
    scene_dir: Path,
    s1_cfg: Stage1Config,
    s2_cfg: Stage2Config,
    support_params: dict,
) -> None:
    """Run the first three stages in-memory and keep only support outputs."""
    scene = load_scene(scene_dir)
    _, boundary_centers, _ = build_boundary_centers(
        scene=scene,
        k=s1_cfg.k,
        min_cross_ratio=s1_cfg.min_cross_ratio,
        min_side_points=s1_cfg.min_side_points,
        ignore_index=s1_cfg.ignore_index,
    )
    validate_boundary_centers(boundary_centers)
    local_clusters, _ = cluster_boundary_centers(
        boundary_centers=boundary_centers,
        eps=s2_cfg.eps,
        min_samples=s2_cfg.min_samples,
        denoise_knn=s2_cfg.denoise_knn,
        sparse_distance_ratio=s2_cfg.sparse_distance_ratio,
        sparse_mad_scale=s2_cfg.sparse_mad_scale,
    )
    validate_local_clusters(local_clusters, num_boundary_centers=boundary_centers["center_coord"].shape[0])
    supports_payload, meta_payload, _ = build_supports_payload(
        boundary_centers=boundary_centers,
        local_clusters=local_clusters,
        params=support_params,
    )
    validate_supports(supports_payload)

    cleanup_scene_dir(scene_dir)
    export_npz(scene_dir / "supports.npz", supports_payload)
    export_support_geometry_xyz(supports_payload, output_dir=scene_dir)

    print("=" * 70)
    print("BF Edge v3 - build_support_dataset_v3")
    print(f"  scene: {scene_dir}")
    print(f"  supports: {meta_payload['num_supports']}")
    print(f"  line: {meta_payload['support_type_hist']['line']}")
    print(f"  polyline: {meta_payload['support_type_hist']['polyline']}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.input)
    scene_dirs = collect_scene_dirs(dataset_root)
    if not scene_dirs:
        print("No valid training/validation scene containing coord.npy and segment.npy was found.")
        return

    s1_cfg = build_stage1_config(args)
    s2_cfg = build_stage2_config(args)
    s3_cfg = build_stage3_config(args)
    support_params = s3_cfg.to_runtime_dict()
    for scene_dir in scene_dirs:
        run_scene(
            scene_dir=scene_dir,
            s1_cfg=s1_cfg,
            s2_cfg=s2_cfg,
            support_params=support_params,
        )


if __name__ == "__main__":
    main()
