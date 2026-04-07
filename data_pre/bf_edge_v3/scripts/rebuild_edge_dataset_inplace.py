"""Rebuild edge.npy + edge_supervision.xyz + support_geometry.xyz in-place.

Runs the full 4-stage pipeline in memory per scene, outputs only the three
target files. Does NOT write intermediate artifacts (supports.npz,
boundary_centers.npz, local_clusters.npz, etc.).

Usage:
    python data_pre/bf_edge_v3/scripts/rebuild_edge_dataset_inplace.py \
        --input /mnt/e/WSL/data/BF_edge_chunk_npy

    # 4 parallel workers:
    python data_pre/bf_edge_v3/scripts/rebuild_edge_dataset_inplace.py \
        --input /mnt/e/WSL/data/BF_edge_chunk_npy --workers 4
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from _bootstrap import ensure_bf_edge_v3_root_on_path

ensure_bf_edge_v3_root_on_path()


SPLITS = ("training", "validation")
TARGET_FILES = ("edge.npy", "edge_supervision.xyz", "support_geometry.xyz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild edge.npy (5-col) + edge_supervision.xyz + support_geometry.xyz in-place"
    )
    parser.add_argument("--input", type=str, required=True, help="Dataset root")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--force", action="store_true", help="Overwrite even if all targets exist")
    return parser.parse_args()


def collect_scene_dirs(dataset_root: Path, force: bool) -> list[Path]:
    scene_dirs: list[Path] = []
    for split in SPLITS:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if not ((scene_dir / "coord.npy").exists() and (scene_dir / "segment.npy").exists()):
                continue
            if not force and all((scene_dir / f).exists() for f in TARGET_FILES):
                edge = np.load(scene_dir / "edge.npy")
                if edge.ndim == 2 and edge.shape[1] == 5:
                    continue
            scene_dirs.append(scene_dir)
    return scene_dirs


def build_edge_array(payload: dict) -> np.ndarray:
    """Build compact 5-col edge.npy: [vec_x, vec_y, vec_z, support, valid]."""
    edge = np.zeros((payload["edge_vec"].shape[0], 5), dtype=np.float32)
    edge[:, 0:3] = payload["edge_vec"].astype(np.float32)
    edge[:, 3] = payload["edge_support"].astype(np.float32)
    edge[:, 4] = payload["edge_valid"].astype(np.float32)
    return edge


try:
    import bf_edge_cpp
    _USE_CPP = True
except ImportError:
    _USE_CPP = False


def _run_scene_cpp(scene_dir: Path) -> dict:
    """Process one scene using C++ accelerated pipeline."""
    from core.config import Stage1Config, Stage2Config, Stage3Config, Stage4Config
    from core.pointwise_core import export_edge_supervision_xyz
    from core.supports_export import export_support_geometry_xyz
    from utils.stage_io import load_scene

    s1 = Stage1Config()
    s2 = Stage2Config()
    s3 = Stage3Config()
    s4 = Stage4Config()

    scene = load_scene(scene_dir)
    coord = scene["coord"].astype(np.float32)
    segment = scene["segment"].astype(np.int32)
    normal = scene.get("normal")
    if normal is not None:
        normal = normal.astype(np.float32)

    # Stage 1: boundary centers
    bc = bf_edge_cpp.build_boundary_centers(
        coord, segment, normal,
        s1.k, s1.min_cross_ratio, s1.min_side_points, s1.ignore_index,
    )

    # Stage 2: clustering
    lc = bf_edge_cpp.cluster_boundary_centers(
        bc["center_coord"], bc["center_tangent"], bc["semantic_pair"],
        micro_eps_scale=s2.micro_eps_scale,
        micro_min_samples=s2.micro_min_samples,
        split_lateral_threshold_scale=s2.split_lateral_threshold_scale,
        merge_radius_scale=s2.merge_radius_scale,
        merge_direction_cos_th=s2.merge_direction_cos_th,
        merge_lateral_scale=s2.merge_lateral_scale,
        rescue_radius_scale=s2.rescue_radius_scale,
        min_cluster_points=s2.min_cluster_points,
    )

    n_clusters = lc["cluster_size"].shape[0]

    # Stage 3: support fitting
    sup = bf_edge_cpp.build_supports(
        bc["center_coord"], bc["confidence"],
        lc["center_index"], lc["cluster_id"],
        lc["semantic_pair"], lc["cluster_size"], lc["cluster_centroid"],
        line_residual_th=s3.line_residual_th,
        min_cluster_size=s3.min_cluster_size,
        max_polyline_vertices=s3.max_polyline_vertices,
        polyline_residual_th=s3.polyline_residual_th,
        min_cluster_density=s3.min_cluster_density,
    )

    n_supports = sup["support_id"].shape[0]

    # Stage 4: find bad supports
    hollow_ids = bf_edge_cpp.find_bad_supports(
        sup["support_id"], sup["semantic_pair"],
        sup["segment_offset"], sup["segment_length"],
        sup["segment_start"], sup["segment_end"],
        sup["line_start"], sup["line_end"],
        sup["cluster_id"], sup["support_type"],
        bc["center_coord"], bc["center_tangent"],
        lc["cluster_id"], lc["center_index"],
    )

    # Stage 4: pointwise supervision
    edge = bf_edge_cpp.build_pointwise_edge_supervision(
        coord, segment,
        sup["support_id"], sup["semantic_pair"],
        sup["segment_offset"], sup["segment_length"],
        sup["segment_start"], sup["segment_end"],
        sup["line_start"], sup["line_end"],
        sup["cluster_id"], sup["support_type"],
        s4.support_radius, s4.ignore_index,
        skip_supports=hollow_ids,
    )

    # Build full supports_payload dict for export functions
    supports_payload = dict(sup)
    # Add fields expected by export but with defaults
    supports_payload.setdefault("center", np.zeros((n_supports, 3), dtype=np.float32))
    supports_payload.setdefault("radius", np.zeros(n_supports, dtype=np.float32))
    supports_payload.setdefault("normal", np.zeros((n_supports, 3), dtype=np.float32))
    supports_payload.setdefault("angle_min", np.zeros(n_supports, dtype=np.float32))
    supports_payload.setdefault("angle_max", np.zeros(n_supports, dtype=np.float32))
    supports_payload.setdefault("polyline_offset", np.zeros(n_supports, dtype=np.int32))
    supports_payload.setdefault("polyline_length", np.zeros(n_supports, dtype=np.int32))
    supports_payload.setdefault("polyline_vertices", np.empty((0, 3), dtype=np.float32))
    supports_payload.setdefault("segment_origin", np.zeros_like(sup["segment_start"]))
    supports_payload.setdefault("segment_direction", np.zeros_like(sup["segment_start"]))
    supports_payload.setdefault("segment_point_count", np.ones(sup["segment_start"].shape[0], dtype=np.int32))

    # Gaussian support weight
    sigma = max(s4.support_radius / 2.0, 1e-8)
    edge_support = np.zeros(coord.shape[0], dtype=np.float32)
    valid = edge["edge_valid"] == 1
    if np.any(valid):
        d = edge["edge_dist"][valid]
        edge_support[valid] = np.exp(-(d * d) / (2.0 * sigma * sigma)).astype(np.float32)

    edge_payload = {
        "edge_dist": edge["edge_dist"],
        "edge_dir": edge["edge_dir"],
        "edge_valid": edge["edge_valid"],
        "edge_support_id": edge["edge_support_id"],
        "edge_vec": edge["edge_vec"],
        "edge_support": edge.get("edge_support", edge_support),
    }

    n_line = int(np.sum(sup["support_type"] == 0))
    n_poly = int(np.sum(sup["support_type"] == 2))

    np.save(scene_dir / "edge.npy", build_edge_array(edge_payload))
    export_edge_supervision_xyz(scene=scene, payload=edge_payload, output_dir=scene_dir)
    export_support_geometry_xyz(supports_payload, output_dir=scene_dir)

    return {
        "scene": f"{scene_dir.parent.name}/{scene_dir.name}",
        "num_points": int(coord.shape[0]),
        "num_valid": int(np.count_nonzero(edge["edge_valid"])),
        "num_supports": int(n_supports),
        "num_hollow": len(hollow_ids),
        "line": n_line,
        "polyline": n_poly,
    }


def _run_scene_python(scene_dir: Path) -> dict:
    """Process one scene using Python fallback pipeline."""
    from core.boundary_centers_core import build_boundary_centers
    from core.config import Stage1Config, Stage2Config, Stage3Config, Stage4Config
    from core.local_clusters_core import cluster_boundary_centers
    from core.pointwise_core import (
        build_pointwise_edge_supervision,
        export_edge_supervision_xyz,
        find_bad_supports,
    )
    from core.supports_core import build_supports_payload
    from core.supports_export import export_support_geometry_xyz
    from utils.stage_io import load_scene

    s1 = Stage1Config()
    s2 = Stage2Config()
    s3_params = Stage3Config().to_runtime_dict()
    s4 = Stage4Config()

    scene = load_scene(scene_dir)

    _, boundary_centers, _ = build_boundary_centers(
        scene=scene, k=s1.k, min_cross_ratio=s1.min_cross_ratio,
        min_side_points=s1.min_side_points, ignore_index=s1.ignore_index,
    )

    local_clusters, _ = cluster_boundary_centers(
        boundary_centers=boundary_centers, config=s2,
    )

    supports_payload, support_meta, _ = build_supports_payload(
        boundary_centers=boundary_centers, local_clusters=local_clusters,
        params=s3_params,
    )

    hollow_ids = find_bad_supports(supports_payload, boundary_centers, local_clusters)

    edge_payload, edge_meta = build_pointwise_edge_supervision(
        scene=scene, supports=supports_payload,
        support_radius=s4.support_radius, ignore_index=s4.ignore_index,
        skip_supports=hollow_ids,
    )

    np.save(scene_dir / "edge.npy", build_edge_array(edge_payload))
    export_edge_supervision_xyz(scene=scene, payload=edge_payload, output_dir=scene_dir)
    export_support_geometry_xyz(supports_payload, output_dir=scene_dir)

    return {
        "scene": f"{scene_dir.parent.name}/{scene_dir.name}",
        "num_points": edge_meta["num_points"],
        "num_valid": edge_meta["num_valid_points"],
        "num_supports": support_meta["num_supports"],
        "num_hollow": len(hollow_ids),
        "line": support_meta["support_type_hist"]["line"],
        "polyline": support_meta["support_type_hist"]["polyline"],
    }


def run_scene(scene_dir: Path) -> dict:
    """Process one scene end-to-end. Uses C++ when available, Python fallback otherwise."""
    if _USE_CPP:
        return _run_scene_cpp(scene_dir)
    return _run_scene_python(scene_dir)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.input)
    scenes = collect_scene_dirs(dataset_root, force=args.force)
    workers = max(1, args.workers)

    total = sum(
        1 for split in SPLITS
        for p in (dataset_root / split).iterdir()
        if p.is_dir() and (p / "coord.npy").exists()
    ) if any((dataset_root / s).is_dir() for s in SPLITS) else 0

    backend = "C++" if _USE_CPP else "Python"
    print(f"Backend: {backend} | Total scenes: {total}, to process: {len(scenes)}, skipped: {total - len(scenes)}, workers: {workers}", flush=True)
    if not scenes:
        print("Nothing to do.")
        return

    t0 = time.time()
    done = 0

    if workers == 1:
        for scene_dir in scenes:
            ts = time.time()
            meta = run_scene(scene_dir)
            done += 1
            elapsed = time.time() - ts
            print(
                f"[{done}/{len(scenes)}] {meta['scene']}  "
                f"{meta['num_points']}pts  {meta['num_valid']}valid  "
                f"{meta['num_supports']}sup(L{meta['line']}+P{meta['polyline']})  "
                f"hollow={meta['num_hollow']}  {elapsed:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_scene, sd): sd for sd in scenes}
            for future in as_completed(futures):
                done += 1
                ts_wall = time.time() - t0
                try:
                    meta = future.result()
                    print(
                        f"[{done}/{len(scenes)}] {meta['scene']}  "
                        f"{meta['num_points']}pts  {meta['num_valid']}valid  "
                        f"{meta['num_supports']}sup(L{meta['line']}+P{meta['polyline']})  "
                        f"hollow={meta['num_hollow']}  wall={ts_wall:.0f}s",
                        flush=True,
                    )
                except Exception as exc:
                    scene_dir = futures[future]
                    print(f"[{done}/{len(scenes)}] {scene_dir.parent.name}/{scene_dir.name}  FAILED: {exc}", flush=True)

    total_time = time.time() - t0
    print(f"\nDone. {done} scenes in {total_time:.0f}s ({total_time/done:.1f}s/scene)")


if __name__ == "__main__":
    main()
