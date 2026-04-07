"""
Build pointwise edge supervision from scene points and fitted supports.

输入目录至少包含:
- coord.npy
- segment.npy
- supports.npz

默认输出:
- edge_dist.npy
- edge_dir.npy
- edge_valid.npy
- edge_support_id.npy
- edge_vec.npy
- edge_support.npy
- edge_supervision.xyz

这一版只负责:
1. 读取场景点和已有 supports
2. 按 point segment -> support semantic_pair 做候选查询
3. 计算点到最近有限 support 几何的投影位移和距离
4. 导出逐点监督数组与轻量可视化

不处理 clustering / support 重构 / smoothing / 全局优化.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from utils.common import EPS, normalize_rows, save_xyz
from utils.stage_io import load_scene


FILL_VALUE = np.inf


def load_supports(input_dir: Path) -> dict:
    """Load minimal fields required for pointwise support query."""
    payload = np.load(input_dir / "supports.npz")

    supports = {
        "support_id": payload["support_id"].astype(np.int32),
        "semantic_pair": payload["semantic_pair"].astype(np.int32),
        "segment_offset": payload["segment_offset"].astype(np.int32),
        "segment_length": payload["segment_length"].astype(np.int32),
        "segment_start": payload["segment_start"].astype(np.float32),
        "segment_end": payload["segment_end"].astype(np.float32),
        "line_start": payload["line_start"].astype(np.float32),
        "line_end": payload["line_end"].astype(np.float32),
    }

    support_count = supports["support_id"].shape[0]
    if supports["semantic_pair"].shape != (support_count, 2):
        raise ValueError("semantic_pair shape/count mismatch")
    if supports["segment_offset"].shape[0] != support_count:
        raise ValueError("segment_offset count mismatch")
    if supports["segment_length"].shape[0] != support_count:
        raise ValueError("segment_length count mismatch")
    if supports["line_start"].shape != (support_count, 3):
        raise ValueError("line_start shape/count mismatch")
    if supports["line_end"].shape != (support_count, 3):
        raise ValueError("line_end shape/count mismatch")

    num_segments = supports["segment_start"].shape[0]
    if supports["segment_end"].shape != (num_segments, 3):
        raise ValueError("segment_end shape/count mismatch")

    return supports


def closest_point_on_segment(
    points: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project points onto one finite segment."""
    seg_vec = (seg_end - seg_start).astype(np.float32)
    seg_len2 = float(np.dot(seg_vec, seg_vec))
    if seg_len2 < EPS:
        closest = np.tile(seg_start[None, :], (points.shape[0], 1)).astype(np.float32)
        dist = np.linalg.norm(points - closest, axis=1).astype(np.float32)
        return closest, dist

    t = ((points - seg_start[None, :]) @ seg_vec) / seg_len2
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    closest = seg_start[None, :] + t[:, None] * seg_vec[None, :]
    dist = np.linalg.norm(points - closest, axis=1).astype(np.float32)
    return closest.astype(np.float32), dist


def closest_points_to_support(
    points: np.ndarray,
    support_id: int,
    supports: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return closest points on one support and corresponding distances."""
    offset = int(supports["segment_offset"][support_id])
    length = int(supports["segment_length"][support_id])

    if length <= 0:
        return closest_point_on_segment(
            points=points,
            seg_start=supports["line_start"][support_id],
            seg_end=supports["line_end"][support_id],
        )

    best_dist = np.full((points.shape[0],), np.inf, dtype=np.float32)
    best_point = np.zeros((points.shape[0], 3), dtype=np.float32)
    for seg_idx in range(offset, offset + length):
        closest, dist = closest_point_on_segment(
            points=points,
            seg_start=supports["segment_start"][seg_idx],
            seg_end=supports["segment_end"][seg_idx],
        )
        better = dist < best_dist
        if np.any(better):
            best_dist[better] = dist[better]
            best_point[better] = closest[better]
    return best_point.astype(np.float32), best_dist.astype(np.float32)


def build_label_to_supports(semantic_pair: np.ndarray) -> dict[int, np.ndarray]:
    """Build reverse index from segment label to candidate support ids."""
    label_to_supports: dict[int, list[int]] = {}
    for support_id, pair in enumerate(semantic_pair.astype(np.int32)):
        for label in np.unique(pair):
            label_to_supports.setdefault(int(label), []).append(int(support_id))
    return {
        label: np.asarray(support_ids, dtype=np.int32)
        for label, support_ids in label_to_supports.items()
    }


def build_edge_support(
    edge_dist: np.ndarray,
    edge_valid: np.ndarray,
    support_radius: float,
) -> tuple[np.ndarray, float]:
    """Build truncated Gaussian boundary support weights inside the supervision radius."""
    sigma = max(float(support_radius) / 2.0, EPS)
    edge_support = np.zeros(edge_dist.shape, dtype=np.float32)
    valid_mask = edge_valid == 1
    if np.any(valid_mask):
        valid_dist = edge_dist[valid_mask]
        edge_support[valid_mask] = np.exp(
            -(valid_dist * valid_dist) / (2.0 * sigma * sigma)
        ).astype(np.float32)
    return edge_support, float(sigma)


def build_pointwise_edge_supervision(
    scene: dict,
    supports: dict,
    support_radius: float,
    ignore_index: int,
) -> tuple[dict, dict]:
    """Build nearest-support boundary snapping supervision."""
    coord = scene["coord"]
    segment = scene["segment"]
    num_points = coord.shape[0]

    edge_dist = np.full((num_points,), FILL_VALUE, dtype=np.float32)
    edge_dir = np.zeros((num_points, 3), dtype=np.float32)
    edge_valid = np.zeros((num_points,), dtype=np.uint8)
    edge_support_id = np.full((num_points,), -1, dtype=np.int32)
    edge_vec = np.zeros((num_points, 3), dtype=np.float32)

    label_to_supports = build_label_to_supports(supports["semantic_pair"])

    unique_labels = np.unique(segment).astype(np.int32)
    for label in unique_labels:
        if int(label) == int(ignore_index):
            continue

        point_index = np.where(segment == label)[0].astype(np.int32)
        candidate_supports = label_to_supports.get(int(label))
        if point_index.size == 0 or candidate_supports is None or candidate_supports.size == 0:
            continue

        points = coord[point_index]
        best_dist = np.full((point_index.shape[0],), np.inf, dtype=np.float32)
        best_q = np.zeros((point_index.shape[0], 3), dtype=np.float32)
        best_support = np.full((point_index.shape[0],), -1, dtype=np.int32)

        for support_id in candidate_supports:
            closest, dist = closest_points_to_support(points=points, support_id=int(support_id), supports=supports)
            better = dist < best_dist
            if np.any(better):
                best_dist[better] = dist[better]
                best_q[better] = closest[better]
                best_support[better] = int(support_id)

        valid = np.isfinite(best_dist) & (best_support >= 0) & (best_dist <= float(support_radius))
        if not np.any(valid):
            continue

        query_index = point_index[valid]
        query_vector = best_q[valid] - points[valid]

        edge_dist[query_index] = best_dist[valid].astype(np.float32)
        edge_vec[query_index] = query_vector.astype(np.float32)
        edge_valid[query_index] = 1
        edge_support_id[query_index] = best_support[valid].astype(np.int32)

    valid_mask = edge_valid == 1
    if np.any(valid_mask):
        edge_dir[valid_mask] = normalize_rows(edge_vec[valid_mask])

    edge_support, sigma = build_edge_support(
        edge_dist=edge_dist,
        edge_valid=edge_valid,
        support_radius=float(support_radius),
    )

    payload = {
        "edge_dist": edge_dist,
        "edge_dir": edge_dir,
        "edge_valid": edge_valid,
        "edge_support_id": edge_support_id,
        "edge_vec": edge_vec.astype(np.float32),
        "edge_support": edge_support.astype(np.float32),
    }
    meta = {
        "version": "bf_edge_v3_build_boundary_snapping_supervision",
        "num_points": int(num_points),
        "num_supports": int(supports["support_id"].shape[0]),
        "num_valid_points": int(np.count_nonzero(edge_valid)),
        "num_invalid_points": int(num_points - np.count_nonzero(edge_valid)),
        "support_radius": float(support_radius),
        "ignore_index": int(ignore_index),
        "support_sigma": float(sigma),
        "support_function": "truncated_gaussian",
        "max_edge_dist": float(support_radius),
        "strength_sigma": float(sigma),
        "edge_dist_fill_value": "inf",
    }
    return payload, meta


def export_edge_arrays(output_dir: Path, payload: dict) -> None:
    """Export final supervision arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "edge_dist.npy", payload["edge_dist"])
    np.save(output_dir / "edge_dir.npy", payload["edge_dir"])
    np.save(output_dir / "edge_valid.npy", payload["edge_valid"])
    np.save(output_dir / "edge_support_id.npy", payload["edge_support_id"])
    np.save(output_dir / "edge_vec.npy", payload["edge_vec"])
    np.save(output_dir / "edge_support.npy", payload["edge_support"])


def export_edge_supervision_xyz(
    scene: dict,
    payload: dict,
    output_dir: Path,
) -> None:
    """Export only valid supervision points for lightweight visualization."""
    valid = payload["edge_valid"] == 1
    if not np.any(valid):
        save_xyz(
            output_dir / "edge_supervision.xyz",
            np.empty((0, 13), dtype=np.float32),
            ["%.6f"] * 13,
        )
        return

    data = np.concatenate(
        [
            scene["coord"][valid].astype(np.float32),
            payload["edge_dist"][valid].reshape(-1, 1).astype(np.float32),
            payload["edge_support"][valid].reshape(-1, 1).astype(np.float32),
            payload["edge_support_id"][valid].reshape(-1, 1).astype(np.float32),
            payload["edge_dir"][valid].astype(np.float32),
            payload["edge_vec"][valid].astype(np.float32),
            scene["segment"][valid].reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    )
    save_xyz(
        output_dir / "edge_supervision.xyz",
        data,
        [
            "%.6f", "%.6f", "%.6f",
            "%.6f",
            "%.6f",
            "%.0f",
            "%.6f", "%.6f", "%.6f",
            "%.6f", "%.6f", "%.6f",
            "%.0f",
        ],
    )
