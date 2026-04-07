"""
Build fine-grained local clusters for BF Edge v3.

Each output cluster is a (semantic_pair, direction_class, spatial_run) triple
that directly satisfies Stage 3's fitter assumptions by construction.

Pipeline per semantic_pair:
1. Spatial DBSCAN clustering
2. Density-adaptive noise rescue
3. Per-cluster lightweight denoise
4. Per-cluster direction grouping + spatial run splitting
5. Each run becomes one output cluster

Phase 4 redesign: trigger judgment/classification/merging eliminated.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from sklearn.cluster import DBSCAN
except ImportError as error:
    raise ImportError("Requires scikit-learn. Install via: pip install scikit-learn") from error

from utils.common import EPS, normalize_rows, normalize_vector, save_npz, save_xyz
from utils.stage_io import load_boundary_centers

from core.config import Stage2Config
from core.fitting import estimate_local_spacing


def spatial_dbscan(coords: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Run spatial DBSCAN on 3D center coordinates."""
    if coords.shape[0] == 0:
        return np.empty((0,), dtype=np.int32)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(coords).labels_
    return labels.astype(np.int32)


def build_cluster_record(
    cluster_id: int,
    semantic_pair: np.ndarray,
    center_coord: np.ndarray,
) -> dict:
    """Build minimal cluster-level statistics for later refinement stages."""
    return {
        "cluster_id": int(cluster_id),
        "semantic_pair": semantic_pair.astype(np.int32),
        "cluster_size": int(center_coord.shape[0]),
        "cluster_centroid": center_coord.mean(axis=0).astype(np.float32),
    }


def build_knn(coords: np.ndarray, k: int) -> np.ndarray:
    """Build local kNN distances without self."""
    n_points = coords.shape[0]
    if n_points == 0:
        return np.empty((0, 0), dtype=np.float32)

    query_k = min(max(k + 1, 1), n_points)
    tree = cKDTree(coords)
    dist, _ = tree.query(coords, k=query_k, workers=-1)
    if query_k == 1:
        dist = dist.reshape(-1, 1)
    if query_k <= 1:
        return np.empty((n_points, 0), dtype=np.float32)
    return dist[:, 1:].astype(np.float32)


def lightweight_denoise_cluster(
    coords: np.ndarray,
    density_knn: int,
    sparse_distance_ratio: float,
    sparse_mad_scale: float,
    max_remove_ratio: float,
    min_keep_points: int,
) -> tuple[np.ndarray, dict]:
    """
    Remove only clearly sparse points inside one coarse cluster.

    This stays intentionally simple:
    - use cluster-internal mean kNN distance
    - mark points whose local spacing is far above the cluster median
    - cancel denoise if removal would be too aggressive
    """
    n_points = coords.shape[0]
    keep_mask = np.ones((n_points,), dtype=bool)
    if n_points == 0:
        return keep_mask, {"denoise_applied": False, "num_removed": 0}

    knn_dist = build_knn(coords, k=density_knn)
    if knn_dist.shape[1] == 0:
        return keep_mask, {"denoise_applied": False, "num_removed": 0}

    local_spacing = knn_dist.mean(axis=1).astype(np.float32)
    finite_spacing = local_spacing[np.isfinite(local_spacing)]
    if finite_spacing.size == 0:
        return keep_mask, {"denoise_applied": False, "num_removed": 0}

    median_spacing = float(np.median(finite_spacing))
    mad_spacing = float(np.median(np.abs(finite_spacing - median_spacing)))
    threshold = max(
        median_spacing * float(sparse_distance_ratio),
        median_spacing + float(sparse_mad_scale) * mad_spacing,
    )
    sparse_mask = local_spacing > threshold
    num_removed = int(np.count_nonzero(sparse_mask))
    if num_removed == 0:
        return keep_mask, {"denoise_applied": False, "num_removed": 0}

    if num_removed > int(max_remove_ratio * n_points):
        return keep_mask, {"denoise_applied": False, "num_removed": 0}
    if int(np.count_nonzero(~sparse_mask)) < int(min_keep_points):
        return keep_mask, {"denoise_applied": False, "num_removed": 0}

    keep_mask = ~sparse_mask
    return keep_mask, {"denoise_applied": True, "num_removed": num_removed}


# -------------------------------------------------------------------------
# Functions moved from trigger_regroup.py (verbatim copy)
# -------------------------------------------------------------------------


def group_tangents(
    tangents: np.ndarray,
    direction_cos_th: float,
) -> np.ndarray:
    """
    Simple sign-invariant tangent grouping.

    Groups tangent vectors by direction similarity (sign-invariant cosine).
    Each group contains tangents whose pairwise |cos(angle)| >= direction_cos_th.
    """
    tangents = normalize_rows(tangents)
    n_points = tangents.shape[0]
    labels = np.full((n_points,), -1, dtype=np.int32)
    if n_points == 0:
        return labels

    representatives: list[np.ndarray] = []
    accumulator: list[np.ndarray] = []

    for idx in range(n_points):
        tangent = tangents[idx]
        best_group = -1
        best_score = -1.0
        best_sign = 1.0
        for group_id, rep in enumerate(representatives):
            score_signed = float(np.dot(tangent, rep))
            score = abs(score_signed)
            if score > best_score:
                best_score = score
                best_group = group_id
                best_sign = 1.0 if score_signed >= 0.0 else -1.0

        if best_group >= 0 and best_score >= float(direction_cos_th):
            labels[idx] = best_group
            accumulator[best_group] = accumulator[best_group] + best_sign * tangent
            representatives[best_group] = normalize_vector(accumulator[best_group])
            continue

        labels[idx] = len(representatives)
        accumulator.append(tangent.copy())
        representatives.append(tangent.copy())

    return labels.astype(np.int32)


def estimate_direction_group_axis(
    coords: np.ndarray,
    tangents: np.ndarray,
) -> np.ndarray:
    """
    Estimate the dominant run axis inside one direction group.

    Prefer the mean tangent because the group is already direction-consistent.
    Fall back to PCA when the mean tangent becomes unstable.
    """
    axis = normalize_vector(np.mean(normalize_rows(tangents), axis=0))
    if np.linalg.norm(axis) >= EPS:
        return axis

    centered = coords - coords.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return normalize_vector(vh[0])


def split_sorted_indices_by_gap(values: np.ndarray, gap_th: float) -> list[np.ndarray]:
    """Split sorted 1D values into consecutive runs by adaptive gap threshold."""
    if values.shape[0] == 0:
        return []

    split_points = [0]
    for idx in range(1, values.shape[0]):
        if float(values[idx] - values[idx - 1]) > float(gap_th):
            split_points.append(idx)
    split_points.append(values.shape[0])

    groups = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        groups.append(np.arange(start, end, dtype=np.int32))
    return groups


def split_direction_group_into_runs(
    coords: np.ndarray,
    tangents: np.ndarray,
    params: dict,
) -> list[np.ndarray]:
    """
    Split one tangent direction group into edge-runs.

    Minimal rule set:
    1. Estimate one dominant axis for the direction group
    2. Separate parallel edges by lateral offset gaps
    3. Split each lateral band by along-axis projection gaps
    """
    n_points = coords.shape[0]
    if n_points == 0:
        return []
    if n_points < int(params["segment_min_points"]):
        return [np.arange(n_points, dtype=np.int32)]

    axis = estimate_direction_group_axis(coords=coords, tangents=tangents)
    if np.linalg.norm(axis) < EPS:
        return [np.arange(n_points, dtype=np.int32)]

    centroid = coords.mean(axis=0, keepdims=True)
    centered = coords - centroid
    along = centered @ axis
    perp_vec = centered - along[:, None] * axis[None, :]

    spacing = max(estimate_local_spacing(coords), 1e-6)
    along_gap_th = max(float(params["segment_run_gap_scale"]) * spacing, 1e-4)
    lateral_gap_th = max(float(params["segment_run_lateral_gap_scale"]) * spacing, 1e-4)
    lateral_band_th = max(float(params["segment_run_lateral_band_scale"]) * spacing, 1e-4)

    if n_points >= 3 and np.max(np.linalg.norm(perp_vec, axis=1)) > 1e-6:
        _, singular_values, vh = np.linalg.svd(perp_vec, full_matrices=False)
        if singular_values.shape[0] > 0 and float(singular_values[0]) > 1e-6:
            lateral_axis = normalize_vector(vh[0])
        else:
            lateral_axis = np.zeros(3, dtype=np.float32)
    else:
        lateral_axis = np.zeros(3, dtype=np.float32)

    if np.linalg.norm(lateral_axis) >= EPS:
        lateral_coord = perp_vec @ lateral_axis
    else:
        lateral_coord = np.zeros((n_points,), dtype=np.float32)

    lateral_order = np.argsort(lateral_coord, kind="mergesort")
    lateral_sorted = lateral_coord[lateral_order]
    lateral_groups = split_sorted_indices_by_gap(lateral_sorted, lateral_gap_th)

    runs: list[np.ndarray] = []
    for group in lateral_groups:
        band_indices = lateral_order[group]
        if band_indices.size == 0:
            continue

        band_lateral = lateral_coord[band_indices]
        if float(np.max(band_lateral) - np.min(band_lateral)) > float(lateral_band_th):
            sub_order = np.argsort(band_lateral, kind="mergesort")
            sub_sorted = band_lateral[sub_order]
            sub_groups = split_sorted_indices_by_gap(sub_sorted, lateral_gap_th)
            candidate_bands = [band_indices[sub_order[sub_group]] for sub_group in sub_groups]
        else:
            candidate_bands = [band_indices]

        for candidate_indices in candidate_bands:
            if candidate_indices.size == 0:
                continue
            candidate_along = along[candidate_indices]
            along_order = np.argsort(candidate_along, kind="mergesort")
            along_sorted = candidate_along[along_order]
            along_groups = split_sorted_indices_by_gap(along_sorted, along_gap_th)
            for along_group in along_groups:
                run_indices = candidate_indices[along_order[along_group]]
                if run_indices.size > 0:
                    runs.append(np.sort(run_indices.astype(np.int32)))

    if not runs:
        return [np.arange(n_points, dtype=np.int32)]

    return runs


# -------------------------------------------------------------------------
# New Stage 2 functions (Phase 4)
# -------------------------------------------------------------------------


def rescue_noise_centers(
    coords: np.ndarray,
    labels: np.ndarray,
    k: int = 8,
    rescue_distance_scale: float = 2.0,
) -> np.ndarray:
    """Assign noise-labeled points to nearest cluster if within density-adaptive threshold.

    For each noise point (label == -1), find its nearest cluster point via cKDTree.
    Compute per-cluster median kNN spacing. Assign the noise point to the nearest
    cluster if distance <= rescue_distance_scale * cluster_median_spacing.
    """
    noise_mask = labels == -1
    if not np.any(noise_mask) or not np.any(~noise_mask):
        return labels.copy()

    cluster_coords = coords[~noise_mask]
    cluster_labels = labels[~noise_mask]
    noise_coords = coords[noise_mask]

    # Build kNN on cluster points
    tree = cKDTree(cluster_coords)

    # For each noise point, find nearest cluster point
    dist, idx = tree.query(noise_coords, k=1)
    nearest_cluster = cluster_labels[idx]

    # Compute per-cluster median spacing for adaptive threshold
    cluster_spacing: dict[int, float] = {}
    for cid in np.unique(cluster_labels):
        cmask = cluster_labels == cid
        cc = cluster_coords[cmask]
        if cc.shape[0] >= k + 1:
            ctree = cKDTree(cc)
            cdist, _ = ctree.query(cc, k=min(k + 1, cc.shape[0]))
            cluster_spacing[int(cid)] = float(np.median(cdist[:, 1:]))
        else:
            # Small cluster: use a fallback based on global median
            cluster_spacing[int(cid)] = float(np.median(dist))

    # Rescue if distance < scale * cluster_spacing
    result = labels.copy()
    noise_indices = np.flatnonzero(noise_mask)
    for i, ni in enumerate(noise_indices):
        cid = int(nearest_cluster[i])
        threshold = rescue_distance_scale * cluster_spacing.get(cid, 999.0)
        if float(dist[i]) <= threshold:
            result[ni] = cid

    return result


def refine_cluster_into_runs(
    coords: np.ndarray,
    tangents: np.ndarray,
    config: Stage2Config,
) -> list[np.ndarray]:
    """Split one DBSCAN cluster into direction-consistent, spatially-continuous runs.

    Returns list of index arrays (indices into the input coords/tangents).
    Each returned array is one output cluster.
    """
    direction_labels = group_tangents(tangents, config.segment_direction_cos_th)
    valid_groups = np.unique(direction_labels[direction_labels >= 0])

    run_params = {
        "segment_min_points": int(config.segment_min_points),
        "segment_run_gap_scale": float(config.segment_run_gap_scale),
        "segment_run_lateral_gap_scale": float(config.segment_run_lateral_gap_scale),
        "segment_run_lateral_band_scale": float(config.segment_run_lateral_band_scale),
    }

    all_runs: list[np.ndarray] = []
    for dg in valid_groups:
        dg_mask = direction_labels == int(dg)
        dg_coords = coords[dg_mask]
        dg_tangents = tangents[dg_mask]
        dg_indices = np.flatnonzero(dg_mask).astype(np.int32)

        if dg_coords.shape[0] < config.segment_min_points:
            continue

        runs = split_direction_group_into_runs(dg_coords, dg_tangents, run_params)
        for run in runs:
            if run.shape[0] >= config.segment_min_points:
                all_runs.append(dg_indices[run])

    if not all_runs:
        # Fallback: return entire cluster as single run
        return [np.arange(coords.shape[0], dtype=np.int32)]

    return all_runs


# -------------------------------------------------------------------------
# Main clustering function
# -------------------------------------------------------------------------


def cluster_boundary_centers(
    boundary_centers: dict,
    config: Stage2Config,
) -> tuple[dict, dict]:
    """Group centers by semantic_pair, cluster, rescue noise, refine into runs.

    Each output cluster is a (semantic_pair, direction_class, spatial_run) triple
    that directly satisfies Stage 3's fitter assumptions by construction.
    """
    center_coord = boundary_centers["center_coord"]
    center_tangent = boundary_centers["center_tangent"]
    semantic_pair = boundary_centers["semantic_pair"]

    eps = config.eps
    min_samples = config.min_samples

    # Compute global median boundary center spacing for density-conditional denoise
    global_median_spacing = float(
        estimate_local_spacing(center_coord, k=min(8, center_coord.shape[0] - 1))
    )

    num_centers = center_coord.shape[0]
    num_removed_by_denoise = 0
    num_rescued = 0
    num_runs = 0
    num_denoise_skipped = 0

    # Each element: (semantic_pair, center_indices_array)
    run_records: list[tuple[np.ndarray, np.ndarray]] = []

    unique_pairs = np.unique(semantic_pair, axis=0)
    for pair in unique_pairs:
        pair_mask = np.all(semantic_pair == pair[None, :], axis=1)
        pair_indices = np.where(pair_mask)[0].astype(np.int32)
        pair_coords = center_coord[pair_indices]
        if pair_indices.size == 0:
            continue

        labels = spatial_dbscan(pair_coords, eps=eps, min_samples=min_samples)

        # Rescue noise points
        noise_before = int(np.count_nonzero(labels == -1))
        labels = rescue_noise_centers(
            pair_coords, labels,
            k=config.rescue_knn,
            rescue_distance_scale=config.rescue_distance_scale,
        )
        noise_after = int(np.count_nonzero(labels == -1))
        num_rescued += noise_before - noise_after

        valid_local_labels = np.unique(labels[labels >= 0]).astype(np.int32)

        for local_label in valid_local_labels:
            local_mask = labels == int(local_label)
            global_indices = pair_indices[local_mask]

            cluster_coords = center_coord[global_indices]
            cluster_spacing = float(
                estimate_local_spacing(cluster_coords, k=min(config.denoise_knn, cluster_coords.shape[0] - 1))
            )

            if cluster_spacing <= config.denoise_density_threshold * global_median_spacing:
                # Dense cluster: apply standard denoise
                keep_mask, denoise_stats = lightweight_denoise_cluster(
                    coords=cluster_coords,
                    density_knn=config.denoise_knn,
                    sparse_distance_ratio=config.sparse_distance_ratio,
                    sparse_mad_scale=config.sparse_mad_scale,
                    max_remove_ratio=float(config.max_remove_ratio),
                    min_keep_points=config.min_keep_points,
                )
            else:
                # Sparse cluster: skip denoise to preserve coverage
                keep_mask = np.ones(cluster_coords.shape[0], dtype=bool)
                denoise_stats = {"denoise_applied": False, "num_removed": 0, "density_skip": True}
                num_denoise_skipped += 1

            kept_indices = global_indices[keep_mask]
            num_removed_by_denoise += int(np.count_nonzero(~keep_mask))

            if kept_indices.size == 0:
                continue

            # Refine cluster into direction-consistent spatial runs
            runs = refine_cluster_into_runs(
                coords=center_coord[kept_indices],
                tangents=center_tangent[kept_indices],
                config=config,
            )

            for run_local_indices in runs:
                run_global_indices = kept_indices[run_local_indices]
                run_records.append((pair.astype(np.int32), run_global_indices.astype(np.int32)))
                num_runs += 1

    # Build output payload
    cluster_records: list[dict] = []
    assigned_center_indices: list[np.ndarray] = []

    for cluster_id, (pair, indices) in enumerate(run_records):
        cluster_records.append(
            build_cluster_record(
                cluster_id=int(cluster_id),
                semantic_pair=pair,
                center_coord=center_coord[indices],
            )
        )
        assigned_center_indices.append(indices)

    if assigned_center_indices:
        keep_center_index = np.concatenate(assigned_center_indices).astype(np.int32)
        cluster_id_arr = np.concatenate(
            [
                np.full((indices.shape[0],), cluster_idx, dtype=np.int32)
                for cluster_idx, indices in enumerate(assigned_center_indices)
            ]
        ).astype(np.int32)
    else:
        keep_center_index = np.empty((0,), dtype=np.int32)
        cluster_id_arr = np.empty((0,), dtype=np.int32)

    payload = {
        "center_index": keep_center_index.astype(np.int32),
        "cluster_id": cluster_id_arr.astype(np.int32),
        "semantic_pair": np.stack([item["semantic_pair"] for item in cluster_records]).astype(
            np.int32
        )
        if cluster_records
        else np.empty((0, 2), dtype=np.int32),
        "cluster_size": np.asarray(
            [item["cluster_size"] for item in cluster_records], dtype=np.int32
        )
        if cluster_records
        else np.empty((0,), dtype=np.int32),
        "cluster_centroid": np.stack([item["cluster_centroid"] for item in cluster_records]).astype(
            np.float32
        )
        if cluster_records
        else np.empty((0, 3), dtype=np.float32),
    }
    meta = {
        "version": "bf_edge_v3_build_local_clusters",
        "num_boundary_centers": int(num_centers),
        "num_clusters": int(len(cluster_records)),
        "num_noise": int(num_centers - keep_center_index.shape[0]),
        "num_assigned": int(keep_center_index.shape[0]),
        "num_rescued": int(num_rescued),
        "num_runs": int(num_runs),
        "num_removed_by_denoise": int(num_removed_by_denoise),
        "num_denoise_skipped": int(num_denoise_skipped),
        "params": {
            "eps": float(config.eps),
            "min_samples": int(config.min_samples),
            "denoise_knn": int(config.denoise_knn),
            "sparse_distance_ratio": float(config.sparse_distance_ratio),
            "sparse_mad_scale": float(config.sparse_mad_scale),
            "max_remove_ratio": float(config.max_remove_ratio),
            "min_keep_points": config.min_keep_points,
            "rescue_knn": int(config.rescue_knn),
            "rescue_distance_scale": float(config.rescue_distance_scale),
            "segment_direction_angle_deg": float(config.segment_direction_angle_deg),
            "segment_min_points": int(config.segment_min_points),
            "denoise_density_threshold": float(config.denoise_density_threshold),
        },
    }
    return payload, meta


def cluster_to_color(cluster_id: int) -> np.ndarray:
    """Stable pseudo color for one cluster. Noise stays gray."""
    if int(cluster_id) < 0:
        return np.array([140, 140, 140], dtype=np.uint8)
    seed = (int(cluster_id) * 1103515245 + 12345) & 0x7FFFFFFF
    return np.array(
        [
            64 + (seed & 127),
            64 + ((seed >> 8) & 127),
            64 + ((seed >> 16) & 127),
        ],
        dtype=np.uint8,
    )


def build_cluster_colors(cluster_ids: np.ndarray) -> np.ndarray:
    """Batch stable pseudo colors for cluster ids."""
    if cluster_ids.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    return np.stack([cluster_to_color(int(cluster_id)) for cluster_id in cluster_ids]).astype(
        np.uint8
    )


def export_npz(path: Path, payload: dict) -> None:
    """Export cluster payload."""
    save_npz(path, payload)


def export_clustered_boundary_centers_xyz(
    boundary_centers: dict,
    local_clusters: dict,
    output_dir: Path,
) -> None:
    """Export only surviving centers for CloudCompare style inspection.

    Columns: x,y,z, r,g,b, cluster_id, pair_a, pair_b, cluster_size, confidence (11 total).
    """
    keep_center_index = local_clusters["center_index"].astype(np.int32)
    center_coord = boundary_centers["center_coord"][keep_center_index]
    cluster_id = local_clusters["cluster_id"].astype(np.int32)

    if center_coord.shape[0] == 0:
        save_xyz(
            output_dir / "clustered_boundary_centers.xyz",
            np.empty((0, 11), dtype=np.float32),
            ["%.6f"] * 11,
        )
        return

    colors = build_cluster_colors(cluster_id).astype(np.float32)
    cluster_pair = local_clusters["semantic_pair"][cluster_id].astype(np.float32)
    cluster_size = local_clusters["cluster_size"][cluster_id].reshape(-1, 1).astype(np.float32)
    confidence = boundary_centers["confidence"][keep_center_index].reshape(-1, 1).astype(np.float32)

    data = np.concatenate(
        [
            center_coord.astype(np.float32),
            colors,
            cluster_id.reshape(-1, 1).astype(np.float32),
            cluster_pair,
            cluster_size,
            confidence,
        ],
        axis=1,
    )
    save_xyz(
        output_dir / "clustered_boundary_centers.xyz",
        data,
        [
            "%.6f", "%.6f", "%.6f",
            "%.0f", "%.0f", "%.0f",
            "%.0f",
            "%.0f", "%.0f",
            "%.0f",
            "%.6f",
        ],
    )
