"""
Build coarse local clusters for BF Edge v3.

输入目录包含:
- boundary_centers.npz

默认输出:
- local_clusters.npz
- clustered_boundary_centers.xyz

这一版只负责:
1. 读取 boundary centers
2. 按 semantic_pair 分组
3. 对每组 center_coord 做空间 DBSCAN
4. 对 coarse cluster 做轻量去噪
5. 对 coarse cluster 做 trigger 判定
6. 导出正式聚类结果和轻量可视化

不处理 split / refinement / merge / RANSAC / supports / vector field.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from sklearn.cluster import DBSCAN
except ImportError as error:
    raise ImportError("Requires scikit-learn. Install via: pip install scikit-learn") from error

from utils.common import normalize_rows, save_npz, save_xyz
from utils.stage_io import load_boundary_centers

from core.params import DEFAULT_TRIGGER_PARAMS, DEFAULT_DENOISE_PARAMS


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


def compute_cluster_trigger_metrics(
    coords: np.ndarray,
    tangents: np.ndarray,
) -> dict:
    """Compute a small set of cluster-level metrics for split triggering."""
    cluster_size = int(coords.shape[0])
    if cluster_size == 0:
        return {
            "cluster_size": 0,
            "linearity": 0.0,
            "tangent_coherence": 0.0,
            "bbox_anisotropy": 0.0,
        }

    centered = coords - coords.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0].astype(np.float32)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-8)
    linearity = float(singular_values[0] / (float(np.sum(singular_values)) + 1e-8))

    tangents = normalize_rows(tangents)
    tangent_coherence = float(np.mean(np.abs(tangents @ direction))) if tangents.shape[0] > 0 else 0.0

    bbox_extent = coords.max(axis=0) - coords.min(axis=0)
    bbox_sorted = np.sort(bbox_extent.astype(np.float32))[::-1]
    bbox_anisotropy = float(
        bbox_sorted[0] / max(float(bbox_sorted[1]), 1e-8)
    ) if bbox_sorted.shape[0] >= 2 else 0.0

    return {
        "cluster_size": cluster_size,
        "linearity": linearity,
        "tangent_coherence": tangent_coherence,
        "bbox_anisotropy": bbox_anisotropy,
    }


def should_trigger_split(metrics: dict, params: dict) -> bool:
    """Conservative trigger for coarse clusters that are likely direction-mixed."""
    if int(metrics["cluster_size"]) < int(params["trigger_min_cluster_size"]):
        return False

    low_linearity = float(metrics["linearity"]) < float(params["trigger_linearity_th"])
    low_tangent_coherence = float(metrics["tangent_coherence"]) < float(
        params["trigger_tangent_coherence_th"]
    )
    not_long_strip = float(metrics["bbox_anisotropy"]) < float(params["trigger_bbox_anisotropy_th"])
    return bool(low_linearity and low_tangent_coherence and not_long_strip)


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


def cluster_boundary_centers(
    boundary_centers: dict,
    eps: float,
    min_samples: int,
    denoise_knn: int,
    sparse_distance_ratio: float,
    sparse_mad_scale: float,
) -> tuple[dict, dict]:
    """Group centers by semantic_pair and apply coarse spatial DBSCAN."""
    center_coord = boundary_centers["center_coord"]
    center_tangent = boundary_centers["center_tangent"]
    semantic_pair = boundary_centers["semantic_pair"]

    num_centers = center_coord.shape[0]
    coarse_cluster_records: list[dict] = []
    num_removed_by_denoise = 0
    trigger_params = {
        "trigger_min_cluster_size": max(
            int(min_samples * DEFAULT_TRIGGER_PARAMS["trigger_min_cluster_size_factor"]),
            int(DEFAULT_TRIGGER_PARAMS["trigger_min_cluster_size_floor"]),
        ),
        "trigger_linearity_th": float(DEFAULT_TRIGGER_PARAMS["linearity_th"]),
        "trigger_tangent_coherence_th": float(DEFAULT_TRIGGER_PARAMS["tangent_coherence_th"]),
        "trigger_bbox_anisotropy_th": float(DEFAULT_TRIGGER_PARAMS["bbox_anisotropy_th"]),
    }
    unique_pairs = np.unique(semantic_pair, axis=0)
    for pair in unique_pairs:
        pair_mask = np.all(semantic_pair == pair[None, :], axis=1)
        pair_indices = np.where(pair_mask)[0].astype(np.int32)
        pair_coords = center_coord[pair_indices]
        if pair_indices.size == 0:
            continue

        labels = spatial_dbscan(pair_coords, eps=eps, min_samples=min_samples)
        valid_local_labels = np.unique(labels[labels >= 0]).astype(np.int32)

        for local_label in valid_local_labels:
            local_mask = labels == int(local_label)
            global_indices = pair_indices[local_mask]

            keep_mask, denoise_stats = lightweight_denoise_cluster(
                coords=center_coord[global_indices],
                density_knn=denoise_knn,
                sparse_distance_ratio=sparse_distance_ratio,
                sparse_mad_scale=sparse_mad_scale,
                max_remove_ratio=float(DEFAULT_DENOISE_PARAMS["max_remove_ratio"]),
                min_keep_points=max(
                    int(min_samples * DEFAULT_DENOISE_PARAMS["min_keep_points_factor"]),
                    int(DEFAULT_DENOISE_PARAMS["min_keep_points_floor"]),
                ),
            )
            kept_indices = global_indices[keep_mask]
            removed_indices = global_indices[~keep_mask]

            num_removed_by_denoise += int(removed_indices.size)

            metrics = compute_cluster_trigger_metrics(
                coords=center_coord[kept_indices],
                tangents=center_tangent[kept_indices],
            )
            coarse_cluster_records.append(
                {
                    "semantic_pair": pair.astype(np.int32),
                    "center_indices": kept_indices.astype(np.int32),
                    "trigger_flag": 1 if should_trigger_split(metrics, trigger_params) else 0,
                }
            )

    cluster_records: list[dict] = []
    trigger_flags: list[int] = []
    assigned_center_indices: list[np.ndarray] = []

    for cluster_id, coarse_record in enumerate(coarse_cluster_records):
        center_indices = coarse_record["center_indices"]
        pair = coarse_record["semantic_pair"]
        trigger_flag = int(coarse_record["trigger_flag"])
        cluster_records.append(
            build_cluster_record(
                cluster_id=int(cluster_id),
                semantic_pair=pair,
                center_coord=center_coord[center_indices],
            )
        )
        trigger_flags.append(trigger_flag)
        assigned_center_indices.append(center_indices.astype(np.int32))

    if assigned_center_indices:
        keep_center_index = np.concatenate(assigned_center_indices).astype(np.int32)
        cluster_id = np.concatenate(
            [
                np.full((indices.shape[0],), cluster_idx, dtype=np.int32)
                for cluster_idx, indices in enumerate(assigned_center_indices)
            ]
        ).astype(np.int32)
    else:
        keep_center_index = np.empty((0,), dtype=np.int32)
        cluster_id = np.empty((0,), dtype=np.int32)
    payload = {
        "center_index": keep_center_index.astype(np.int32),
        "cluster_id": cluster_id.astype(np.int32),
        "semantic_pair": np.stack([item["semantic_pair"] for item in cluster_records]).astype(
            np.int32
        )
        if cluster_records
        else np.empty((0, 2), dtype=np.int32),
        "cluster_trigger_flag": np.asarray(trigger_flags, dtype=np.uint8)
        if cluster_records
        else np.empty((0,), dtype=np.uint8),
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
        "num_trigger_clusters": int(sum(trigger_flags)),
        "num_trigger_centers": int(
            np.sum(payload["cluster_size"][payload["cluster_trigger_flag"] > 0])
        ) if cluster_records else 0,
        "num_removed_by_denoise": int(num_removed_by_denoise),
        "params": {
            "eps": float(eps),
            "min_samples": int(min_samples),
            "denoise_knn": int(denoise_knn),
            "sparse_distance_ratio": float(sparse_distance_ratio),
            "sparse_mad_scale": float(sparse_mad_scale),
            "max_remove_ratio": float(DEFAULT_DENOISE_PARAMS["max_remove_ratio"]),
            "min_keep_points": max(
                int(min_samples * DEFAULT_DENOISE_PARAMS["min_keep_points_factor"]),
                int(DEFAULT_DENOISE_PARAMS["min_keep_points_floor"]),
            ),
            **trigger_params,
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
    """Export only surviving centers for CloudCompare style inspection."""
    keep_center_index = local_clusters["center_index"].astype(np.int32)
    center_coord = boundary_centers["center_coord"][keep_center_index]
    cluster_id = local_clusters["cluster_id"].astype(np.int32)

    if center_coord.shape[0] == 0:
        save_xyz(
            output_dir / "clustered_boundary_centers.xyz",
            np.empty((0, 12), dtype=np.float32),
            ["%.6f"] * 12,
        )
        return

    colors = build_cluster_colors(cluster_id).astype(np.float32)
    trigger_flag = local_clusters["cluster_trigger_flag"][cluster_id].reshape(-1, 1).astype(np.float32)
    trigger_mask = trigger_flag[:, 0] > 0
    if np.any(trigger_mask):
        colors[trigger_mask] = np.array([255, 90, 90], dtype=np.float32)
    cluster_pair = local_clusters["semantic_pair"][cluster_id].astype(np.float32)
    cluster_size = local_clusters["cluster_size"][cluster_id].reshape(-1, 1).astype(np.float32)
    confidence = boundary_centers["confidence"][keep_center_index].reshape(-1, 1).astype(np.float32)

    data = np.concatenate(
        [
            center_coord.astype(np.float32),
            colors,
            cluster_id.reshape(-1, 1).astype(np.float32),
            trigger_flag,
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
            "%.0f",
            "%.0f", "%.0f",
            "%.0f",
            "%.6f",
        ],
    )
