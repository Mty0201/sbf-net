"""
Fit local supports for BF Edge v3.

输入目录包含:
- boundary_centers.npz
- local_clusters.npz

默认输出:
- supports.npz
- support_geometry.xyz
- trigger_group_classes.xyz

这一版的 support fitting 采用双路径:
1. 非 trigger cluster:
   - 优先拟合 line support
   - 失败后回退 polyline support
2. trigger cluster:
   - 先做方向 + 空间再分组
   - 候选子组分成主边组 / 碎片组 / 坏组
   - 碎片优先并入主边组
   - 重组后的子簇直接复用普通 line/polyline support 拟合

不处理 vector field / support graph / branch polyline.
"""

from __future__ import annotations

import numpy as np

from utils.common import (
    normalize_rows,
)

from core.fitting import (
    estimate_local_spacing,
    fit_line_support,
    fit_polyline_support,
    regularize_support_orientation,
)
from core.trigger_regroup import (
    absorb_sparse_endpoint_points,
    regroup_trigger_cluster,
)
from core.supports_export import (
    export_npz,
    export_support_geometry_xyz,
    export_trigger_group_classes_xyz,
)


SUPPORT_TYPE_LINE = 0
SUPPORT_TYPE_POLYLINE = 2

DEFAULT_FIT_PARAMS = {
    "segment_direction_angle_deg": 20.0,
    "segment_run_gap_scale": 3.0,
    "segment_run_lateral_gap_scale": 2.5,
    "segment_run_lateral_band_scale": 3.0,
    "segment_min_points": 6,
    "trigger_main_min_points": 12,
    "trigger_main_linearity_th": 0.88,
    "trigger_main_tangent_angle_deg": 20.0,
    "trigger_main_length_scale": 6.0,
    "trigger_main_lateral_scale": 2.5,
    "trigger_fragment_min_points": 6,
    "trigger_fragment_linearity_th": 0.78,
    "trigger_fragment_tangent_angle_deg": 28.0,
    "trigger_fragment_lateral_scale": 3.5,
    "trigger_fragment_attach_dist_scale": 2.5,
    "trigger_fragment_attach_gap_scale": 4.0,
    "trigger_fragment_attach_angle_deg": 20.0,
    "trigger_main_merge_angle_deg": 10.0,
    "trigger_main_merge_dist_scale": 1.5,
    "trigger_main_merge_gap_scale": 3.0,
    "trigger_main_merge_lateral_scale": 1.4,
    "trigger_endpoint_absorb_dist_scale": 2.2,
    "trigger_endpoint_absorb_line_dist_scale": 1.6,
    "trigger_endpoint_absorb_proj_scale": 2.6,
    "trigger_endpoint_absorb_max_points_per_end": 12,
}


def rebuild_cluster_records(
    boundary_centers: dict,
    local_clusters: dict,
) -> list[dict]:
    """Rebuild cluster records from v3 compact cluster payload."""
    center_coord = boundary_centers["center_coord"]
    center_confidence = boundary_centers["confidence"]
    center_index = local_clusters["center_index"]
    cluster_id = local_clusters["cluster_id"]
    semantic_pair = local_clusters["semantic_pair"]
    cluster_size = local_clusters["cluster_size"]
    cluster_centroid = local_clusters["cluster_centroid"]
    cluster_trigger_flag = local_clusters["cluster_trigger_flag"]

    if center_index.shape[0] != cluster_id.shape[0]:
        raise ValueError("center_index and cluster_id count mismatch")
    if np.any(center_index < 0) or np.any(center_index >= center_coord.shape[0]):
        raise ValueError("center_index out of range")

    records: list[dict] = []
    unique_cluster_ids = np.unique(cluster_id).astype(np.int32)
    for cid in unique_cluster_ids:
        if cid < 0 or cid >= semantic_pair.shape[0]:
            raise ValueError(f"cluster_id out of range: {cid}")
        member_mask = cluster_id == cid
        member_center_index = center_index[member_mask].astype(np.int32)
        observed_size = int(member_center_index.shape[0])
        declared_size = int(cluster_size[cid])
        if observed_size != declared_size:
            raise ValueError(
                f"cluster_size mismatch: cid={cid}, observed={observed_size}, declared={declared_size}"
            )

        records.append(
            {
                "cluster_id": int(cid),
                "semantic_pair": semantic_pair[cid].astype(np.int32),
                "cluster_size": declared_size,
                "cluster_centroid": cluster_centroid[cid].astype(np.float32),
                "cluster_trigger_flag": int(cluster_trigger_flag[cid]),
                "center_indices": member_center_index,
                "cluster_confidence": float(np.mean(center_confidence[member_center_index])),
            }
        )
    return records


def build_standard_support_record(
    cluster_record: dict,
    points: np.ndarray,
    params: dict,
) -> dict | None:
    """Build one ordinary line/polyline support from one line-like cluster."""
    line_support = fit_line_support(points)
    if line_support is not None and line_support["fit_residual"] <= float(params["line_residual_th"]):
        direction, orientation_prior_score = regularize_support_orientation(line_support["direction"])
        return {
            "support_type": SUPPORT_TYPE_LINE,
            "cluster_id": int(cluster_record["cluster_id"]),
            "semantic_pair": cluster_record["semantic_pair"].astype(np.int32),
            "confidence": float(cluster_record["cluster_confidence"]),
            "fit_residual": float(line_support["fit_residual"]),
            "coverage_radius": float(line_support["coverage_radius"]),
            "origin": line_support["origin"].astype(np.float32),
            "direction": direction.astype(np.float32),
            "center": np.zeros(3, dtype=np.float32),
            "radius": 0.0,
            "normal": np.zeros(3, dtype=np.float32),
            "angle_min": 0.0,
            "angle_max": 0.0,
            "polyline_vertices": np.empty((0, 3), dtype=np.float32),
            "line_start": line_support["line_start"].astype(np.float32),
            "line_end": line_support["line_end"].astype(np.float32),
            "orientation_prior_score": float(orientation_prior_score),
            "segments": {
                "segment_start": line_support["line_start"][None, :].astype(np.float32),
                "segment_end": line_support["line_end"][None, :].astype(np.float32),
                "segment_origin": line_support["origin"][None, :].astype(np.float32),
                "segment_direction": direction[None, :].astype(np.float32),
                "segment_point_count": np.asarray([points.shape[0]], dtype=np.int32),
            },
        }

    polyline_support = fit_polyline_support(points, max_vertices=int(params["max_polyline_vertices"]))
    direction, orientation_prior_score = regularize_support_orientation(polyline_support["direction"])
    vertices = polyline_support["vertices"].astype(np.float32)
    if vertices.shape[0] >= 2:
        segment_start = vertices[:-1].astype(np.float32)
        segment_end = vertices[1:].astype(np.float32)
        segment_origin = (0.5 * (segment_start + segment_end)).astype(np.float32)
        segment_direction = normalize_rows(segment_end - segment_start)
        point_count = max(points.shape[0] // max(vertices.shape[0] - 1, 1), 1)
        segment_point_count = np.full((segment_start.shape[0],), point_count, dtype=np.int32)
    else:
        segment_start = np.empty((0, 3), dtype=np.float32)
        segment_end = np.empty((0, 3), dtype=np.float32)
        segment_origin = np.empty((0, 3), dtype=np.float32)
        segment_direction = np.empty((0, 3), dtype=np.float32)
        segment_point_count = np.empty((0,), dtype=np.int32)

    return {
        "support_type": SUPPORT_TYPE_POLYLINE,
        "cluster_id": int(cluster_record["cluster_id"]),
        "semantic_pair": cluster_record["semantic_pair"].astype(np.int32),
        "confidence": float(cluster_record["cluster_confidence"]),
        "fit_residual": float(polyline_support["fit_residual"]),
        "coverage_radius": float(polyline_support["coverage_radius"]),
        "origin": polyline_support["origin"].astype(np.float32),
        "direction": direction.astype(np.float32),
        "center": np.zeros(3, dtype=np.float32),
        "radius": 0.0,
        "normal": np.zeros(3, dtype=np.float32),
        "angle_min": 0.0,
        "angle_max": 0.0,
        "polyline_vertices": vertices,
        "line_start": np.zeros(3, dtype=np.float32),
        "line_end": np.zeros(3, dtype=np.float32),
        "orientation_prior_score": float(orientation_prior_score),
        "segments": {
            "segment_start": segment_start,
            "segment_end": segment_end,
            "segment_origin": segment_origin,
            "segment_direction": segment_direction,
            "segment_point_count": segment_point_count,
        },
    }


def build_trigger_support_records(
    cluster_record: dict,
    points: np.ndarray,
    tangents: np.ndarray,
    params: dict,
) -> tuple[list[dict], list[np.ndarray]]:
    """
    Trigger path: regroup first, then reuse ordinary support fitting.

    The regrouping step only decides which subgroups are worth fitting.
    Final supports are ordinary line/polyline supports, not a special trigger-only fitter.
    """
    final_bundles, visualization_rows, unused_bad_indices = regroup_trigger_cluster(
        cluster_record=cluster_record,
        points=points,
        tangents=tangents,
        params=params,
    )

    support_records: list[dict] = []
    remaining_bad = unused_bad_indices.astype(np.int32)

    for bundle in final_bundles:
        bundle_record = {
            "cluster_id": int(cluster_record["cluster_id"]),
            "seed_subgroup_id": int(bundle["seed_subgroup_id"]),
            "member_indices": bundle["member_indices"].astype(np.int32),
        }
        updated_indices, remaining_bad, absorbed = absorb_sparse_endpoint_points(
            bundle=bundle_record,
            points=points,
            bad_point_indices=remaining_bad,
            params=params,
        )

        subgroup_points = points[updated_indices]
        if subgroup_points.shape[0] < int(params["segment_min_points"]):
            continue
        subgroup_record = {
            "cluster_id": int(cluster_record["cluster_id"]),
            "semantic_pair": cluster_record["semantic_pair"].astype(np.int32),
            "cluster_confidence": float(cluster_record["cluster_confidence"]),
        }
        support = build_standard_support_record(
            cluster_record=subgroup_record,
            points=subgroup_points,
            params=params,
        )
        if support is not None:
            support_records.append(support)

    return support_records, visualization_rows


def build_support_record(
    cluster_record: dict,
    points: np.ndarray,
    tangents: np.ndarray,
    params: dict,
) -> tuple[list[dict], list[np.ndarray]]:
    """Build support record(s) from one cluster."""
    if int(cluster_record["cluster_trigger_flag"]) > 0:
        return build_trigger_support_records(
            cluster_record=cluster_record,
            points=points,
            tangents=tangents,
            params=params,
        )

    support = build_standard_support_record(
        cluster_record=cluster_record,
        points=points,
        params=params,
    )
    return ([support] if support is not None else []), []


def build_supports_payload(
    boundary_centers: dict,
    local_clusters: dict,
    params: dict,
) -> tuple[dict, dict]:
    """Build supports payload and metadata from boundary_centers + local_clusters."""
    center_coord = boundary_centers["center_coord"]
    center_tangent = boundary_centers["center_tangent"]
    cluster_records = rebuild_cluster_records(boundary_centers, local_clusters)

    support_records = []
    trigger_group_visualization = []
    for record in cluster_records:
        if int(record["cluster_size"]) < int(params["min_cluster_size"]):
            continue
        points = center_coord[record["center_indices"]]
        tangents = center_tangent[record["center_indices"]]
        supports, visualization_rows = build_support_record(
            cluster_record=record,
            points=points,
            tangents=tangents,
            params=params,
        )
        support_records.extend(supports)
        trigger_group_visualization.extend(visualization_rows)

    polyline_vertices = []
    polyline_offset = []
    polyline_length = []
    segment_start = []
    segment_end = []
    segment_origin = []
    segment_direction = []
    segment_point_count = []
    segment_offset = []
    segment_length = []

    for support in support_records:
        vertices = support["polyline_vertices"]
        polyline_offset.append(len(polyline_vertices))
        polyline_length.append(vertices.shape[0])
        if vertices.shape[0] > 0:
            polyline_vertices.extend(vertices.tolist())

        segments = support["segments"]
        seg_count = int(segments["segment_start"].shape[0])
        segment_offset.append(len(segment_start))
        segment_length.append(seg_count)
        if seg_count > 0:
            segment_start.extend(segments["segment_start"].tolist())
            segment_end.extend(segments["segment_end"].tolist())
            segment_origin.extend(segments["segment_origin"].tolist())
            segment_direction.extend(segments["segment_direction"].tolist())
            segment_point_count.extend(segments["segment_point_count"].tolist())

    if support_records:
        supports_payload = {
            "support_id": np.arange(len(support_records), dtype=np.int32),
            "support_type": np.asarray([record["support_type"] for record in support_records], dtype=np.int32),
            "semantic_pair": np.stack([record["semantic_pair"] for record in support_records]).astype(np.int32),
            "confidence": np.asarray([record["confidence"] for record in support_records], dtype=np.float32),
            "fit_residual": np.asarray([record["fit_residual"] for record in support_records], dtype=np.float32),
            "coverage_radius": np.asarray([record["coverage_radius"] for record in support_records], dtype=np.float32),
            "cluster_id": np.asarray([record["cluster_id"] for record in support_records], dtype=np.int32),
            "origin": np.stack([record["origin"] for record in support_records]).astype(np.float32),
            "direction": np.stack([record["direction"] for record in support_records]).astype(np.float32),
            "center": np.stack([record["center"] for record in support_records]).astype(np.float32),
            "radius": np.asarray([record["radius"] for record in support_records], dtype=np.float32),
            "normal": np.stack([record["normal"] for record in support_records]).astype(np.float32),
            "angle_min": np.asarray([record["angle_min"] for record in support_records], dtype=np.float32),
            "angle_max": np.asarray([record["angle_max"] for record in support_records], dtype=np.float32),
            "polyline_offset": np.asarray(polyline_offset, dtype=np.int32),
            "polyline_length": np.asarray(polyline_length, dtype=np.int32),
            "polyline_vertices": np.asarray(polyline_vertices, dtype=np.float32).reshape(-1, 3),
            "line_start": np.stack([record["line_start"] for record in support_records]).astype(np.float32),
            "line_end": np.stack([record["line_end"] for record in support_records]).astype(np.float32),
            "orientation_prior_score": np.asarray(
                [record["orientation_prior_score"] for record in support_records], dtype=np.float32
            ),
            "segment_offset": np.asarray(segment_offset, dtype=np.int32),
            "segment_length": np.asarray(segment_length, dtype=np.int32),
            "segment_start": np.asarray(segment_start, dtype=np.float32).reshape(-1, 3),
            "segment_end": np.asarray(segment_end, dtype=np.float32).reshape(-1, 3),
            "segment_origin": np.asarray(segment_origin, dtype=np.float32).reshape(-1, 3),
            "segment_direction": np.asarray(segment_direction, dtype=np.float32).reshape(-1, 3),
            "segment_point_count": np.asarray(segment_point_count, dtype=np.int32),
        }
    else:
        supports_payload = {
            "support_id": np.empty((0,), dtype=np.int32),
            "support_type": np.empty((0,), dtype=np.int32),
            "semantic_pair": np.empty((0, 2), dtype=np.int32),
            "confidence": np.empty((0,), dtype=np.float32),
            "fit_residual": np.empty((0,), dtype=np.float32),
            "coverage_radius": np.empty((0,), dtype=np.float32),
            "cluster_id": np.empty((0,), dtype=np.int32),
            "origin": np.empty((0, 3), dtype=np.float32),
            "direction": np.empty((0, 3), dtype=np.float32),
            "center": np.empty((0, 3), dtype=np.float32),
            "radius": np.empty((0,), dtype=np.float32),
            "normal": np.empty((0, 3), dtype=np.float32),
            "angle_min": np.empty((0,), dtype=np.float32),
            "angle_max": np.empty((0,), dtype=np.float32),
            "polyline_offset": np.empty((0,), dtype=np.int32),
            "polyline_length": np.empty((0,), dtype=np.int32),
            "polyline_vertices": np.empty((0, 3), dtype=np.float32),
            "line_start": np.empty((0, 3), dtype=np.float32),
            "line_end": np.empty((0, 3), dtype=np.float32),
            "orientation_prior_score": np.empty((0,), dtype=np.float32),
            "segment_offset": np.empty((0,), dtype=np.int32),
            "segment_length": np.empty((0,), dtype=np.int32),
            "segment_start": np.empty((0, 3), dtype=np.float32),
            "segment_end": np.empty((0, 3), dtype=np.float32),
            "segment_origin": np.empty((0, 3), dtype=np.float32),
            "segment_direction": np.empty((0, 3), dtype=np.float32),
            "segment_point_count": np.empty((0,), dtype=np.int32),
        }

    support_type_hist = {
        "line": int(np.sum(supports_payload["support_type"] == SUPPORT_TYPE_LINE)),
        "polyline": int(np.sum(supports_payload["support_type"] == SUPPORT_TYPE_POLYLINE)),
    }
    meta_payload = {
        "version": "bf_edge_v3_fit_local_supports",
        "num_boundary_centers": int(center_coord.shape[0]),
        "num_clusters": int(len(cluster_records)),
        "num_supports": int(supports_payload["support_id"].shape[0]),
        "support_type_hist": support_type_hist,
        "params": params,
    }
    debug_payload = {
        "trigger_group_visualization": (
            np.concatenate(trigger_group_visualization, axis=0).astype(np.float32)
            if trigger_group_visualization
            else np.empty((0, 10), dtype=np.float32)
        )
    }
    return supports_payload, meta_payload, debug_payload


