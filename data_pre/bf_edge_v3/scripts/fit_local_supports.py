import argparse
from pathlib import Path

import numpy as np

from _bootstrap import ensure_bf_edge_v3_root_on_path

ensure_bf_edge_v3_root_on_path()

from core.supports_core import (
    DEFAULT_FIT_PARAMS,
    build_supports_payload,
)
from core.supports_export import (
    export_npz,
    export_support_geometry_xyz,
    export_trigger_group_classes_xyz,
)
from utils.stage_io import collect_stage_tasks, load_boundary_centers, load_local_clusters


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for support fitting."""
    parser = argparse.ArgumentParser(description="BF Edge v3: fit local supports")
    parser.add_argument("--input", type=str, required=True, help="Scene dir or root dir")
    parser.add_argument("--output", type=str, default=None, help="Output dir. Default: same as input")
    parser.add_argument("--line-residual-th", type=float, default=0.01, help="Line mean residual threshold before falling back to polyline")
    parser.add_argument("--min-cluster-size", type=int, default=8, help="Minimum cluster size for support fitting")
    parser.add_argument("--max-polyline-vertices", type=int, default=32, help="Maximum polyline vertex count for non-trigger fallback")
    return parser.parse_args()


def build_runtime_params(args: argparse.Namespace) -> dict:
    """Build internal support fitting parameters from stable CLI."""
    internal_params = dict(DEFAULT_FIT_PARAMS)
    params = {
        "line_residual_th": float(args.line_residual_th),
        "min_cluster_size": int(args.min_cluster_size),
        "max_polyline_vertices": int(args.max_polyline_vertices),
    }
    params.update(
        {
            "segment_direction_cos_th": float(np.cos(np.deg2rad(float(internal_params["segment_direction_angle_deg"])))),
            "segment_run_gap_scale": float(internal_params["segment_run_gap_scale"]),
            "segment_run_lateral_gap_scale": float(internal_params["segment_run_lateral_gap_scale"]),
            "segment_run_lateral_band_scale": float(internal_params["segment_run_lateral_band_scale"]),
            "segment_min_points": int(internal_params["segment_min_points"]),
            "trigger_main_min_points": int(internal_params["trigger_main_min_points"]),
            "trigger_main_linearity_th": float(internal_params["trigger_main_linearity_th"]),
            "trigger_main_tangent_cos_th": float(np.cos(np.deg2rad(float(internal_params["trigger_main_tangent_angle_deg"])))),
            "trigger_main_length_scale": float(internal_params["trigger_main_length_scale"]),
            "trigger_main_lateral_scale": float(internal_params["trigger_main_lateral_scale"]),
            "trigger_fragment_min_points": int(internal_params["trigger_fragment_min_points"]),
            "trigger_fragment_linearity_th": float(internal_params["trigger_fragment_linearity_th"]),
            "trigger_fragment_tangent_cos_th": float(np.cos(np.deg2rad(float(internal_params["trigger_fragment_tangent_angle_deg"])))),
            "trigger_fragment_lateral_scale": float(internal_params["trigger_fragment_lateral_scale"]),
            "trigger_fragment_attach_dist_scale": float(internal_params["trigger_fragment_attach_dist_scale"]),
            "trigger_fragment_attach_gap_scale": float(internal_params["trigger_fragment_attach_gap_scale"]),
            "trigger_fragment_attach_cos_th": float(np.cos(np.deg2rad(float(internal_params["trigger_fragment_attach_angle_deg"])))),
            "trigger_main_merge_cos_th": float(np.cos(np.deg2rad(float(internal_params["trigger_main_merge_angle_deg"])))),
            "trigger_main_merge_dist_scale": float(internal_params["trigger_main_merge_dist_scale"]),
            "trigger_main_merge_gap_scale": float(internal_params["trigger_main_merge_gap_scale"]),
            "trigger_main_merge_lateral_scale": float(internal_params["trigger_main_merge_lateral_scale"]),
            "trigger_endpoint_absorb_dist_scale": float(internal_params["trigger_endpoint_absorb_dist_scale"]),
            "trigger_endpoint_absorb_line_dist_scale": float(internal_params["trigger_endpoint_absorb_line_dist_scale"]),
            "trigger_endpoint_absorb_proj_scale": float(internal_params["trigger_endpoint_absorb_proj_scale"]),
            "trigger_endpoint_absorb_max_points_per_end": int(internal_params["trigger_endpoint_absorb_max_points_per_end"]),
        }
    )
    return params


def run_scene(input_dir: Path, output_dir: Path, params: dict) -> None:
    """Run support fitting for one stage directory."""
    boundary_centers = load_boundary_centers(input_dir)
    local_clusters = load_local_clusters(input_dir)
    supports_payload, meta_payload, debug_payload = build_supports_payload(
        boundary_centers=boundary_centers,
        local_clusters=local_clusters,
        params=params,
    )

    export_npz(output_dir / "supports.npz", supports_payload)
    export_support_geometry_xyz(supports_payload, output_dir=output_dir)
    export_trigger_group_classes_xyz(debug_payload, output_dir=output_dir)

    print("=" * 70)
    print("BF Edge v3 - fit_local_supports")
    print(f"  input: {input_dir}")
    print(f"  output: {output_dir}")
    print(f"  clusters: {meta_payload['num_clusters']}")
    print(f"  supports: {meta_payload['num_supports']}")
    print(f"  line: {meta_payload['support_type_hist']['line']}")
    print(f"  polyline: {meta_payload['support_type_hist']['polyline']}")
    print(f"  total_segments: {supports_payload['segment_start'].shape[0]}")
    print("=" * 70)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output is not None else None
    params = build_runtime_params(args)
    tasks = collect_stage_tasks(
        input_path=input_path,
        output_path=output_path,
        stage_file="local_clusters.npz",
        required_files=("boundary_centers.npz", "local_clusters.npz"),
    )
    if not tasks:
        print("No valid directory containing boundary_centers.npz and local_clusters.npz was found.")
        return
    for input_dir, output_dir in tasks:
        run_scene(input_dir=input_dir, output_dir=output_dir, params=params)


if __name__ == "__main__":
    main()
