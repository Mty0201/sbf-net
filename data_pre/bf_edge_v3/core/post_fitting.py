"""
Post-fitting point rescue for bf_edge_v3 pipeline.

Retained from the former trigger_regroup.py -- provides endpoint absorption
for sparse points near fitted support endpoints.
"""

from __future__ import annotations

import numpy as np

from utils.common import EPS, normalize_vector
from core.fitting import estimate_local_spacing, fit_line_support


def absorb_sparse_endpoint_points(
    bundle: dict,
    points: np.ndarray,
    bad_point_indices: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Attach sparse unused bad points only near one bundle's two endpoints."""
    member_indices = bundle["member_indices"].astype(np.int32)
    if bad_point_indices.size == 0:
        return member_indices, bad_point_indices, []

    base_points = points[member_indices]
    line_support = fit_line_support(base_points)
    if line_support is None:
        return member_indices, bad_point_indices, []

    start = line_support["line_start"]
    end = line_support["line_end"]
    direction = normalize_vector(end - start)
    if np.linalg.norm(direction) < EPS:
        return member_indices, bad_point_indices, []

    bad_points = points[bad_point_indices]
    seg_len = float(np.linalg.norm(end - start))
    if seg_len < EPS:
        return member_indices, bad_point_indices, []

    spacing = max(estimate_local_spacing(base_points), 1e-6)
    endpoint_dist_th = float(params["trigger_endpoint_absorb_dist_scale"]) * spacing
    line_dist_th = float(params["trigger_endpoint_absorb_line_dist_scale"]) * spacing
    endpoint_proj_th = float(params["trigger_endpoint_absorb_proj_scale"]) * spacing
    endpoint_limit = int(params["trigger_endpoint_absorb_max_points_per_end"])

    seg_vec = end - start
    seg_len2 = float(np.dot(seg_vec, seg_vec))
    t = ((bad_points - start[None, :]) @ seg_vec) / max(seg_len2, EPS)
    proj = start[None, :] + np.clip(t, 0.0, 1.0)[:, None] * seg_vec[None, :]
    line_dist = np.linalg.norm(bad_points - proj, axis=1)
    start_dist = np.linalg.norm(bad_points - start[None, :], axis=1)
    end_dist = np.linalg.norm(bad_points - end[None, :], axis=1)
    along = (bad_points - start[None, :]) @ direction

    absorbed_rows: list[np.ndarray] = []
    absorbed_mask = np.zeros((bad_point_indices.shape[0],), dtype=bool)

    for endpoint_id, endpoint_dist, endpoint_proj_ref in (
        (0, start_dist, 0.0),
        (1, end_dist, seg_len),
    ):
        endpoint_proj_dist = np.abs(along - endpoint_proj_ref)
        candidate_mask = (
            (endpoint_dist <= endpoint_dist_th)
            & (line_dist <= line_dist_th)
            & (endpoint_proj_dist <= endpoint_proj_th)
            & (~absorbed_mask)
        )
        candidate_indices = np.flatnonzero(candidate_mask).astype(np.int32)
        if candidate_indices.size == 0:
            continue
        order = np.argsort(endpoint_dist[candidate_indices], kind="mergesort")
        selected = candidate_indices[order[:endpoint_limit]]
        absorbed_mask[selected] = True

        absorbed_points = bad_points[selected]
        cluster_col = np.full((absorbed_points.shape[0], 1), float(bundle["cluster_id"]), dtype=np.float32)
        seed_col = np.full((absorbed_points.shape[0], 1), float(bundle["seed_subgroup_id"]), dtype=np.float32)
        endpoint_col = np.full((absorbed_points.shape[0], 1), float(endpoint_id), dtype=np.float32)
        rows = np.concatenate(
            [
                absorbed_points.astype(np.float32),
                np.tile(np.asarray([[70.0, 160.0, 245.0]], dtype=np.float32), (absorbed_points.shape[0], 1)),
                cluster_col,
                seed_col,
                endpoint_col,
            ],
            axis=1,
        )
        absorbed_rows.append(rows)

    if not np.any(absorbed_mask):
        return member_indices, bad_point_indices, absorbed_rows

    absorbed_point_indices = bad_point_indices[absorbed_mask].astype(np.int32)
    remaining_bad = bad_point_indices[~absorbed_mask].astype(np.int32)
    updated_member_indices = np.unique(
        np.concatenate([member_indices, absorbed_point_indices]).astype(np.int32)
    )
    return updated_member_indices, remaining_bad, absorbed_rows
