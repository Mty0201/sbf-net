"""
Build boundary centers for BF Edge v3.

输入场景目录至少包含:
- coord.npy: (N, 3)
- segment.npy: (N,)
- normal.npy: optional, (N, 3)
- color.npy: optional, only used for visualization export

默认输出:
- boundary_centers.npz
- boundary_centers.xyz
- boundary_candidates.xyz

这一版只负责:
1. 读取场景
2. 检测语义边界候选点
3. 为候选点构造 boundary centers
4. 顺手导出轻量可视化文件

不处理 clustering / supports / vector field.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from utils.common import (
    EPS,
    normalize_rows,
    normalize_vector,
    save_npz,
    save_xyz,
)
from utils.stage_io import load_scene


def build_knn_graph(coord: np.ndarray, k: int) -> np.ndarray:
    """Build a kNN graph without self indices."""
    n_points = coord.shape[0]
    if n_points == 0:
        return np.empty((0, 0), dtype=np.int32)

    query_k = min(max(k + 1, 1), n_points)
    tree = cKDTree(coord)
    _, index = tree.query(coord, k=query_k, workers=-1)

    if query_k == 1:
        index = index.reshape(-1, 1)

    if query_k <= 1:
        return np.empty((n_points, 0), dtype=np.int32)

    return index[:, 1:].astype(np.int32)


def detect_boundary_candidates(
    segment: np.ndarray,
    knn_index: np.ndarray,
    ignore_index: int,
    min_cross_ratio: float,
) -> dict:
    """Detect point-wise semantic boundary candidates from local label changes."""
    point_index: list[int] = []
    semantic_pair: list[np.ndarray] = []
    cross_ratio: list[float] = []

    for point_idx, neighbors in enumerate(knn_index):
        label_self = int(segment[point_idx])
        if label_self == ignore_index or neighbors.size == 0:
            continue

        neighbor_labels = segment[neighbors]
        valid_labels = neighbor_labels[neighbor_labels != ignore_index]
        if valid_labels.size == 0:
            continue

        diff_labels = valid_labels[valid_labels != label_self]
        if diff_labels.size == 0:
            continue

        ratio = float(diff_labels.size / max(valid_labels.size, 1))
        if ratio < min_cross_ratio:
            continue

        unique_labels, counts = np.unique(diff_labels, return_counts=True)
        paired_label = int(unique_labels[np.argmax(counts)])
        point_index.append(int(point_idx))
        semantic_pair.append(np.array(sorted((label_self, paired_label)), dtype=np.int32))
        cross_ratio.append(ratio)

    return {
        "point_index": np.asarray(point_index, dtype=np.int32),
        "semantic_pair": np.asarray(semantic_pair, dtype=np.int32).reshape(-1, 2),
        "cross_ratio": np.asarray(cross_ratio, dtype=np.float32),
    }


def estimate_pca_tangent(
    points: np.ndarray,
    center_coord: np.ndarray,
    center_normal: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Estimate tangent in the plane orthogonal to the boundary normal."""
    if points.shape[0] < 2:
        return np.zeros(3, dtype=np.float32), 0.0

    centered = points - center_coord[None, :]
    planar = centered - np.outer(centered @ center_normal, center_normal)
    planar = planar[np.linalg.norm(planar, axis=1) > EPS]
    if planar.shape[0] < 2:
        return np.zeros(3, dtype=np.float32), 0.0

    _, singular_values, vh = np.linalg.svd(planar, full_matrices=False)
    tangent = normalize_vector(vh[0])
    if np.linalg.norm(tangent) < EPS:
        return np.zeros(3, dtype=np.float32), 0.0

    denom = float(np.sum(singular_values)) + EPS
    score = float(singular_values[0] / denom)
    return tangent, float(np.clip(score, 0.0, 1.0))


def estimate_fallback_tangent(
    normal: np.ndarray | None,
    local_index: np.ndarray,
    center_normal: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fallback tangent from mean surface normal if PCA is unstable."""
    if normal is None or local_index.size == 0:
        return np.zeros(3, dtype=np.float32), 0.0

    local_normals = normal[local_index]
    local_normals = local_normals[np.linalg.norm(local_normals, axis=1) > EPS]
    if local_normals.shape[0] == 0:
        return np.zeros(3, dtype=np.float32), 0.0

    surface_normal = normalize_vector(local_normals.mean(axis=0))
    tangent = normalize_vector(np.cross(surface_normal, center_normal))
    if np.linalg.norm(tangent) < EPS:
        return np.zeros(3, dtype=np.float32), 0.0
    return tangent, 0.3


def estimate_boundary_center(
    coord: np.ndarray,
    segment: np.ndarray,
    normal: np.ndarray | None,
    source_point_index: int,
    neighbor_index: np.ndarray,
    semantic_pair: np.ndarray,
    candidate_cross_ratio: float,
    min_side_points: int,
) -> dict | None:
    """Estimate one boundary center record from a candidate point."""
    label_a = int(semantic_pair[0])
    label_b = int(semantic_pair[1])

    local_index = np.concatenate(
        [
            np.asarray([source_point_index], dtype=np.int32),
            neighbor_index.astype(np.int32),
        ]
    )
    local_index = np.unique(local_index)
    local_labels = segment[local_index]

    index_a = local_index[local_labels == label_a]
    index_b = local_index[local_labels == label_b]
    if index_a.size < min_side_points or index_b.size < min_side_points:
        return None

    centroid_a = coord[index_a].mean(axis=0).astype(np.float32)
    centroid_b = coord[index_b].mean(axis=0).astype(np.float32)
    separation = centroid_b - centroid_a
    separation_norm = float(np.linalg.norm(separation))
    if separation_norm < EPS:
        return None

    center_coord = (0.5 * (centroid_a + centroid_b)).astype(np.float32)
    center_normal = (separation / separation_norm).astype(np.float32)

    pair_index = np.concatenate([index_a, index_b]).astype(np.int32)
    tangent, tangent_score = estimate_pca_tangent(
        points=coord[pair_index],
        center_coord=center_coord,
        center_normal=center_normal,
    )
    if np.linalg.norm(tangent) < EPS:
        tangent, tangent_score = estimate_fallback_tangent(
            normal=normal,
            local_index=pair_index,
            center_normal=center_normal,
        )
    if np.linalg.norm(tangent) < EPS:
        return None

    side_balance = 1.0 - abs(index_a.size - index_b.size) / max(index_a.size + index_b.size, 1)
    separation_score = float(separation_norm / (separation_norm + 1.0))
    confidence = (
        0.45 * float(candidate_cross_ratio)
        + 0.30 * float(side_balance)
        + 0.15 * float(separation_score)
        + 0.10 * float(tangent_score)
    )
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return {
        "center_coord": center_coord,
        # Kept for later grouping/refinement where side-separation direction may help.
        "center_normal": center_normal,
        # Kept because downstream clustering already depends on local tangent consistency.
        "center_tangent": tangent.astype(np.float32),
        "semantic_pair": semantic_pair.astype(np.int32),
        "source_point_index": int(source_point_index),
        "confidence": np.float32(confidence),
    }


def build_boundary_centers(
    scene: dict,
    k: int,
    min_cross_ratio: float,
    min_side_points: int,
    ignore_index: int,
) -> tuple[dict, dict, dict]:
    """Run the full center-building stage for one scene."""
    coord = scene["coord"]
    segment = scene["segment"]
    normal = scene["normal"]

    knn_index = build_knn_graph(coord, k=k)
    candidates = detect_boundary_candidates(
        segment=segment,
        knn_index=knn_index,
        ignore_index=ignore_index,
        min_cross_ratio=min_cross_ratio,
    )

    center_records: list[dict] = []
    for idx, source_point_index in enumerate(candidates["point_index"]):
        record = estimate_boundary_center(
            coord=coord,
            segment=segment,
            normal=normal,
            source_point_index=int(source_point_index),
            neighbor_index=knn_index[int(source_point_index)],
            semantic_pair=candidates["semantic_pair"][idx],
            candidate_cross_ratio=float(candidates["cross_ratio"][idx]),
            min_side_points=min_side_points,
        )
        if record is not None:
            center_records.append(record)

    if center_records:
        centers_payload = {
            "center_coord": np.stack([item["center_coord"] for item in center_records]).astype(
                np.float32
            ),
            "center_normal": normalize_rows(
                np.stack([item["center_normal"] for item in center_records]).astype(np.float32)
            ),
            "center_tangent": normalize_rows(
                np.stack([item["center_tangent"] for item in center_records]).astype(np.float32)
            ),
            "semantic_pair": np.stack([item["semantic_pair"] for item in center_records]).astype(
                np.int32
            ),
            "source_point_index": np.asarray(
                [item["source_point_index"] for item in center_records], dtype=np.int32
            ),
            "confidence": np.asarray(
                [item["confidence"] for item in center_records], dtype=np.float32
            ),
        }
    else:
        centers_payload = {
            "center_coord": np.empty((0, 3), dtype=np.float32),
            "center_normal": np.empty((0, 3), dtype=np.float32),
            "center_tangent": np.empty((0, 3), dtype=np.float32),
            "semantic_pair": np.empty((0, 2), dtype=np.int32),
            "source_point_index": np.empty((0,), dtype=np.int32),
            "confidence": np.empty((0,), dtype=np.float32),
        }

    meta = {
        "version": "bf_edge_v3_build_boundary_centers",
        "num_points": int(coord.shape[0]),
        "num_candidates": int(candidates["point_index"].shape[0]),
        "num_centers": int(centers_payload["center_coord"].shape[0]),
        "num_semantic_pairs": int(
            np.unique(centers_payload["semantic_pair"], axis=0).shape[0]
            if centers_payload["semantic_pair"].size > 0
            else 0
        ),
        "has_normal": bool(normal is not None),
        "params": {
            "k": int(k),
            "min_cross_ratio": float(min_cross_ratio),
            "min_side_points": int(min_side_points),
            "ignore_index": int(ignore_index),
        },
    }
    return candidates, centers_payload, meta


def pair_to_color(pair: np.ndarray) -> np.ndarray:
    """Stable pseudo color for one semantic pair."""
    label_a = int(pair[0])
    label_b = int(pair[1])
    seed = (label_a * 73856093) ^ (label_b * 19349663) ^ 0x9E3779B1
    return np.array(
        [
            64 + (seed & 127),
            64 + ((seed >> 8) & 127),
            64 + ((seed >> 16) & 127),
        ],
        dtype=np.uint8,
    )


def build_pair_colors(pairs: np.ndarray) -> np.ndarray:
    """Batch stable pseudo colors for semantic pairs."""
    if pairs.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    return np.stack([pair_to_color(pair) for pair in pairs]).astype(np.uint8)


def export_boundary_centers_npz(path: Path, payload: dict) -> None:
    """Export the center payload used by later pipeline stages."""
    save_npz(path, payload)


def export_candidate_xyz(
    scene: dict,
    candidates: dict,
    output_dir: Path,
) -> None:
    """Export lightweight candidate visualization for quick inspection."""
    point_index = candidates["point_index"]
    if point_index.shape[0] == 0:
        save_xyz(output_dir / "boundary_candidates.xyz", np.empty((0, 10)), ["%.6f"] * 10)
        return

    coord = scene["coord"][point_index]
    pair = candidates["semantic_pair"]
    color = build_pair_colors(pair).astype(np.float32)
    cross_ratio = candidates["cross_ratio"].reshape(-1, 1).astype(np.float32)
    source_index = point_index.reshape(-1, 1).astype(np.float32)

    data = np.concatenate(
        [
            coord.astype(np.float32),
            color,
            pair.astype(np.float32),
            cross_ratio,
            source_index,
        ],
        axis=1,
    )
    save_xyz(
        output_dir / "boundary_candidates.xyz",
        data,
        [
            "%.6f", "%.6f", "%.6f",
            "%.0f", "%.0f", "%.0f",
            "%.0f", "%.0f",
            "%.6f",
            "%.0f",
        ],
    )


def export_centers_xyz(
    centers_payload: dict,
    output_dir: Path,
) -> None:
    """Export center visualization with fields needed for quick sanity checks."""
    center_coord = centers_payload["center_coord"]
    if center_coord.shape[0] == 0:
        save_xyz(output_dir / "boundary_centers.xyz", np.empty((0, 16)), ["%.6f"] * 16)
        return

    pair = centers_payload["semantic_pair"]
    color = build_pair_colors(pair).astype(np.float32)
    confidence = centers_payload["confidence"].reshape(-1, 1).astype(np.float32)
    source_index = centers_payload["source_point_index"].reshape(-1, 1).astype(np.float32)
    data = np.concatenate(
        [
            center_coord.astype(np.float32),
            color,
            centers_payload["center_normal"].astype(np.float32),
            centers_payload["center_tangent"].astype(np.float32),
            pair.astype(np.float32),
            confidence,
            source_index,
        ],
        axis=1,
    )
    save_xyz(
        output_dir / "boundary_centers.xyz",
        data,
        [
            "%.6f", "%.6f", "%.6f",
            "%.0f", "%.0f", "%.0f",
            "%.6f", "%.6f", "%.6f",
            "%.6f", "%.6f", "%.6f",
            "%.0f", "%.0f",
            "%.6f",
            "%.0f",
        ],
    )
