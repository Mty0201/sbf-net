"""
Core fitting algorithms for support element generation.

Extracted from supports_core.py during Phase 2 refactor.
All functions preserve original behavior exactly.
"""

from __future__ import annotations
import math

import numpy as np

from utils.common import (
    EPS,
    normalize_rows,
    normalize_vector,
)


def point_to_line_distance(points: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Point-to-line distance in 3D."""
    direction = normalize_vector(direction)
    if np.linalg.norm(direction) < EPS:
        return np.full((points.shape[0],), np.inf, dtype=np.float32)
    diff = points - origin[None, :]
    proj = np.outer(diff @ direction, direction)
    perp = diff - proj
    return np.linalg.norm(perp, axis=1).astype(np.float32)


def point_to_segment_distance(points: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> np.ndarray:
    """Point-to-segment distance in 3D."""
    seg_vec = seg_end - seg_start
    seg_len2 = float(np.dot(seg_vec, seg_vec))
    if seg_len2 < EPS:
        return np.linalg.norm(points - seg_start[None, :], axis=1).astype(np.float32)
    t = ((points - seg_start[None, :]) @ seg_vec) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    proj = seg_start[None, :] + t[:, None] * seg_vec[None, :]
    return np.linalg.norm(points - proj, axis=1).astype(np.float32)


def point_to_polyline_distance(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Point-to-polyline minimum distance."""
    if vertices.shape[0] == 0:
        return np.full((points.shape[0],), np.inf, dtype=np.float32)
    if vertices.shape[0] == 1:
        return np.linalg.norm(points - vertices[0][None, :], axis=1).astype(np.float32)

    dist = np.full((points.shape[0],), np.inf, dtype=np.float32)
    for idx in range(vertices.shape[0] - 1):
        seg_dist = point_to_segment_distance(points, vertices[idx], vertices[idx + 1])
        dist = np.minimum(dist, seg_dist)
    return dist


def line_to_endpoints(points: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a PCA line into finite endpoints."""
    direction = normalize_vector(direction)
    t = (points - origin[None, :]) @ direction
    p_start = origin + float(np.min(t)) * direction
    p_end = origin + float(np.max(t)) * direction
    return p_start.astype(np.float32), p_end.astype(np.float32)


def fit_line_support(points: np.ndarray) -> dict | None:
    """Fit a line support by PCA."""
    if points.shape[0] < 2:
        return None

    centroid = points.mean(axis=0).astype(np.float32)
    centered = points - centroid[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = normalize_vector(vh[0])
    if np.linalg.norm(direction) < EPS:
        return None

    start, end = line_to_endpoints(points, centroid, direction)
    midpoint = (0.5 * (start + end)).astype(np.float32)
    direction = normalize_vector(end - start)
    if np.linalg.norm(direction) < EPS:
        return None

    residual = float(point_to_line_distance(points, midpoint, direction).mean())
    coverage_radius = float(np.max(np.linalg.norm(points - midpoint[None, :], axis=1)))
    length = float(np.linalg.norm(end - start))
    return {
        "origin": midpoint,
        "direction": direction,
        "fit_residual": residual,
        "coverage_radius": coverage_radius,
        "length": length,
        "line_start": start,
        "line_end": end,
    }


def build_polyline_vertices(points: np.ndarray, max_vertices: int = 32) -> np.ndarray:
    """Build polyline vertices by binning along the primary axis.

    The number of bins adapts to the cluster's geometry: one vertex per
    ``vertex_spacing_scale * local_spacing`` of along-axis extent, clamped
    to [2, max_vertices].  Each vertex is the centroid of all points in its
    bin, suppressing lateral noise that would otherwise produce snake/zigzag
    artefacts.
    """
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    if points.shape[0] == 1:
        return points.astype(np.float32)

    centroid = points.mean(axis=0)
    centered = points - centroid[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = normalize_vector(vh[0])
    t = centered @ direction  # along-axis projection

    t_min, t_max = float(t.min()), float(t.max())
    extent = t_max - t_min
    if extent < 1e-12:
        return points.mean(axis=0, keepdims=True).astype(np.float32)

    # Adaptive vertex count: one vertex per 5× local spacing
    spacing = max(estimate_local_spacing(points), 1e-6)
    n_bins = int(np.clip(round(extent / (5.0 * spacing)), 2, max_vertices))
    n_bins = min(n_bins, points.shape[0])

    if n_bins == points.shape[0]:
        return points[np.argsort(t)].astype(np.float32)

    bin_edges = np.linspace(t_min, t_max, n_bins + 1)
    bin_idx = np.digitize(t, bin_edges[1:-1])  # 0 .. n_bins-1

    vertices = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            vertices.append(points[mask].mean(axis=0))
    if not vertices:
        return points.mean(axis=0, keepdims=True).astype(np.float32)

    return np.stack(vertices).astype(np.float32)


def fit_polyline_support(points: np.ndarray, max_vertices: int = 32) -> dict:
    """Fit a polyline support."""
    vertices = build_polyline_vertices(points, max_vertices=max_vertices)
    residual = float(point_to_polyline_distance(points, vertices).mean())
    centroid = points.mean(axis=0).astype(np.float32)
    coverage_radius = float(np.max(np.linalg.norm(points - centroid[None, :], axis=1)))
    direction = (
        normalize_vector(vertices[-1] - vertices[0])
        if vertices.shape[0] >= 2
        else np.zeros(3, dtype=np.float32)
    )
    return {
        "vertices": vertices.astype(np.float32),
        "origin": centroid,
        "direction": direction.astype(np.float32),
        "fit_residual": residual,
        "coverage_radius": coverage_radius,
    }


def regularize_support_orientation(direction: np.ndarray) -> tuple[np.ndarray, float]:
    """Light orientation prior toward major axes."""
    direction = normalize_vector(direction)
    if np.linalg.norm(direction) < EPS:
        return direction, 0.0

    axes = np.eye(3, dtype=np.float32)
    dots = axes @ direction
    best_axis_id = int(np.argmax(np.abs(dots)))
    best_dot = float(dots[best_axis_id])
    best_score = float(abs(best_dot))

    snap_threshold = math.cos(math.radians(15.0))
    if best_score < snap_threshold:
        return direction, best_score

    target_axis = axes[best_axis_id] * (1.0 if best_dot >= 0.0 else -1.0)
    alpha = min(max((best_score - snap_threshold) / (1.0 - snap_threshold + EPS), 0.0), 1.0)
    alpha = 0.3 * alpha
    regularized = normalize_vector((1.0 - alpha) * direction + alpha * target_axis)
    return regularized, best_score


def segment_record_from_endpoints(start: np.ndarray, end: np.ndarray, num_points: int) -> dict:
    """Build one segment record from endpoints."""
    start = start.astype(np.float32)
    end = end.astype(np.float32)
    origin = (0.5 * (start + end)).astype(np.float32)
    direction = normalize_vector(end - start)
    length = float(np.linalg.norm(end - start))
    return {
        "segment_start": start,
        "segment_end": end,
        "segment_origin": origin,
        "segment_direction": direction.astype(np.float32),
        "segment_length_geom": length,
        "segment_point_count": int(num_points),
    }


def estimate_local_spacing(coords: np.ndarray, k: int = 6) -> float:
    """Estimate point spacing from local kNN statistics. O(n log n) via cKDTree."""
    if coords.shape[0] < 2:
        return 0.01
    from scipy.spatial import cKDTree
    query_k = min(k + 1, coords.shape[0])
    tree = cKDTree(coords)
    dist, _ = tree.query(coords, k=query_k)
    if query_k <= 1:
        return 0.01
    knn_dist = dist[:, 1:query_k] if dist.ndim == 2 else dist[1:query_k].reshape(1, -1)
    if knn_dist.size == 0:
        return 0.01
    return float(np.median(knn_dist))
