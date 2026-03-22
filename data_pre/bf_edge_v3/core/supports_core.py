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
import math
from pathlib import Path

import numpy as np

from utils.common import (
    EPS,
    normalize_rows,
    normalize_vector,
    save_npz,
    save_xyz,
)
from utils.stage_io import load_boundary_centers, load_local_clusters


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
    """Sort points along the primary direction and downsample to polyline vertices."""
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    if points.shape[0] == 1:
        return points.astype(np.float32)

    centroid = points.mean(axis=0)
    centered = points - centroid[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = normalize_vector(vh[0])
    t = centered @ direction
    order = np.argsort(t)
    ordered = points[order]

    keep = min(max_vertices, ordered.shape[0])
    if keep == ordered.shape[0]:
        return ordered.astype(np.float32)

    sample_index = np.linspace(0, ordered.shape[0] - 1, num=keep)
    sample_index = np.unique(np.round(sample_index).astype(np.int32))
    return ordered[sample_index].astype(np.float32)


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
    """Estimate point spacing from local kNN statistics."""
    if coords.shape[0] < 2:
        return 0.01
    query_k = min(max(k + 1, 1), coords.shape[0])
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    sorted_dist = np.sort(dist, axis=1)
    knn_dist = sorted_dist[:, 1:query_k]
    if knn_dist.size == 0:
        return 0.01
    return float(np.median(knn_dist))


def group_tangents(
    tangents: np.ndarray,
    direction_cos_th: float,
) -> np.ndarray:
    """
    Simple sign-invariant tangent grouping.

    This is the trigger support baseline:
    - grouping uses only tangent consistency
    - no extra spatial gate
    - no later fake-edge suppression chain is coupled here
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


def compute_subgroup_metrics(
    points: np.ndarray,
    tangents: np.ndarray,
) -> dict | None:
    """Compute a small set of quality metrics for one trigger subgroup."""
    if points.shape[0] < 2:
        return None

    line_support = fit_line_support(points)
    if line_support is None:
        return None

    centered = points - points.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values.shape[0] == 0:
        return None

    principal = normalize_vector(vh[0])
    secondary = float(singular_values[1]) if singular_values.shape[0] > 1 else 0.0
    primary = float(singular_values[0])
    linearity = primary / max(primary + secondary, EPS)

    t_aligned = normalize_rows(tangents)
    sign = np.sign(t_aligned @ principal)
    sign[sign == 0.0] = 1.0
    t_aligned = t_aligned * sign[:, None]
    tangent_coherence = float(np.mean(np.clip(t_aligned @ principal, -1.0, 1.0)))

    proj = centered @ principal
    length = float(np.max(proj) - np.min(proj))
    lateral = centered - proj[:, None] * principal[None, :]
    lateral_spread = float(np.percentile(np.linalg.norm(lateral, axis=1), 90))

    return {
        "line_support": line_support,
        "direction": principal.astype(np.float32),
        "linearity": float(linearity),
        "tangent_coherence": float(tangent_coherence),
        "length": float(length),
        "lateral_spread": float(lateral_spread),
        "point_count": int(points.shape[0]),
        "centroid": points.mean(axis=0).astype(np.float32),
    }


def classify_trigger_subgroup(
    metrics: dict,
    spacing: float,
    params: dict,
) -> str:
    """Classify one trigger subgroup as main / fragment / bad."""
    point_count = int(metrics["point_count"])
    linearity = float(metrics["linearity"])
    length = float(metrics["length"])
    lateral_spread = float(metrics["lateral_spread"])
    tangent_coherence = float(metrics["tangent_coherence"])

    if (
        point_count >= int(params["trigger_main_min_points"])
        and linearity >= float(params["trigger_main_linearity_th"])
        and tangent_coherence >= float(params["trigger_main_tangent_cos_th"])
        and length >= float(params["trigger_main_length_scale"]) * spacing
        and lateral_spread <= float(params["trigger_main_lateral_scale"]) * spacing
    ):
        return "main"

    if (
        point_count >= int(params["trigger_fragment_min_points"])
        and linearity >= float(params["trigger_fragment_linearity_th"])
        and tangent_coherence >= float(params["trigger_fragment_tangent_cos_th"])
        and lateral_spread <= float(params["trigger_fragment_lateral_scale"]) * spacing
    ):
        return "fragment"

    return "bad"


def merge_subgroup_points(groups: list[dict]) -> np.ndarray:
    """Merge member indices from multiple subgroup records."""
    if not groups:
        return np.empty((0,), dtype=np.int32)
    merged = np.concatenate([group["member_indices"] for group in groups]).astype(np.int32)
    return np.unique(merged).astype(np.int32)


def classify_color(label: str) -> np.ndarray:
    """Fixed color for trigger group class visualization."""
    if label == "main":
        return np.asarray([80.0, 200.0, 90.0], dtype=np.float32)
    if label == "fragment":
        return np.asarray([240.0, 190.0, 70.0], dtype=np.float32)
    return np.asarray([210.0, 80.0, 80.0], dtype=np.float32)


def try_attach_group_to_main_bundle(
    group: dict,
    main_bundles: list[dict],
    points: np.ndarray,
    attach_cos_th: float,
    attach_dist_th: float,
    attach_along_gap_th: float,
) -> dict | None:
    """Try to attach one fragment-like subgroup to the best matching main bundle."""
    group_dir = normalize_rows(group["tangents"])
    group_centroid = group["metrics"]["centroid"]
    best_bundle = None
    best_cost = np.inf
    for bundle in main_bundles:
        main_dir = normalize_vector(bundle["line_support"]["line_end"] - bundle["line_support"]["line_start"])
        direction_support = float(np.mean(np.abs(group_dir @ main_dir)))
        if direction_support < attach_cos_th:
            continue

        dist_to_line = float(
            point_to_line_distance(
                group_centroid[None, :],
                bundle["line_support"]["origin"],
                main_dir,
            )[0]
        )
        if dist_to_line > attach_dist_th:
            continue

        main_member_indices = merge_subgroup_points(bundle["groups"])
        main_centered = points[main_member_indices] - bundle["line_support"]["origin"][None, :]
        main_proj = main_centered @ main_dir
        group_proj = (points[group["member_indices"]] - bundle["line_support"]["origin"][None, :]) @ main_dir
        main_min, main_max = float(np.min(main_proj)), float(np.max(main_proj))
        group_min, group_max = float(np.min(group_proj)), float(np.max(group_proj))
        gap = max(main_min - group_max, group_min - main_max, 0.0)
        if gap > attach_along_gap_th:
            continue

        cost = dist_to_line + gap
        if cost < best_cost:
            best_cost = cost
            best_bundle = bundle

    return best_bundle


def refresh_main_bundle_geometry(bundle: dict, points: np.ndarray) -> None:
    """Recompute one main bundle geometry after fragment/main merging."""
    member_indices = merge_subgroup_points(bundle["groups"])
    bundle["member_indices"] = member_indices
    line_support = fit_line_support(points[member_indices])
    if line_support is not None:
        bundle["line_support"] = line_support


def should_merge_main_bundles(
    bundle_a: dict,
    bundle_b: dict,
    points: np.ndarray,
    direction_cos_th: float,
    line_dist_th: float,
    along_gap_th: float,
    lateral_offset_th: float,
) -> bool:
    """Check whether two main bundles are close enough to be treated as one real edge."""
    line_a = bundle_a.get("line_support")
    line_b = bundle_b.get("line_support")
    if line_a is None or line_b is None:
        return False

    dir_a = normalize_vector(line_a["line_end"] - line_a["line_start"])
    dir_b = normalize_vector(line_b["line_end"] - line_b["line_start"])
    if min(np.linalg.norm(dir_a), np.linalg.norm(dir_b)) < EPS:
        return False

    if float(abs(np.dot(dir_a, dir_b))) < direction_cos_th:
        return False

    avg_dir = normalize_vector(dir_a + np.sign(np.dot(dir_a, dir_b)) * dir_b)
    if np.linalg.norm(avg_dir) < EPS:
        avg_dir = dir_a

    centroid_a = points[bundle_a["member_indices"]].mean(axis=0)
    centroid_b = points[bundle_b["member_indices"]].mean(axis=0)
    lateral_offset = float(
        np.linalg.norm((centroid_b - centroid_a) - np.dot(centroid_b - centroid_a, avg_dir) * avg_dir)
    )
    if lateral_offset > lateral_offset_th:
        return False

    dist_ab = float(point_to_line_distance(centroid_a[None, :], line_b["origin"], dir_b)[0])
    dist_ba = float(point_to_line_distance(centroid_b[None, :], line_a["origin"], dir_a)[0])
    if max(dist_ab, dist_ba) > line_dist_th:
        return False

    endpoints_a = np.stack([line_a["line_start"], line_a["line_end"]]).astype(np.float32)
    endpoints_b = np.stack([line_b["line_start"], line_b["line_end"]]).astype(np.float32)
    proj_a = endpoints_a @ avg_dir
    proj_b = endpoints_b @ avg_dir
    a_min, a_max = float(np.min(proj_a)), float(np.max(proj_a))
    b_min, b_max = float(np.min(proj_b)), float(np.max(proj_b))
    gap = max(a_min - b_max, b_min - a_max, 0.0)
    if gap > along_gap_th:
        return False

    return True


def regroup_trigger_cluster(
    cluster_record: dict,
    points: np.ndarray,
    tangents: np.ndarray,
    params: dict,
) -> tuple[list[dict], list[np.ndarray], np.ndarray]:
    """
    Regroup one trigger cluster into a small set of line-like subclusters.

    Returns:
    - final_bundles: regrouped main bundles, ready for ordinary fitting
    - visualization_rows: rows describing candidate subgroup class / merge target
    - unused_bad_indices: bad-point pool for later endpoint absorption
    """
    direction_labels = group_tangents(
        tangents=tangents,
        direction_cos_th=float(params["segment_direction_cos_th"]),
    )
    cluster_spacing = max(estimate_local_spacing(points), 1e-6)
    candidate_groups: list[dict] = []

    valid_direction_groups = np.unique(direction_labels[direction_labels >= 0]).astype(np.int32)
    subgroup_counter = 0
    for direction_group in valid_direction_groups:
        member_mask = direction_labels == direction_group
        group_points = points[member_mask]
        group_tangent_vectors = tangents[member_mask]
        group_member_indices = np.flatnonzero(member_mask).astype(np.int32)
        if group_points.shape[0] < int(params["segment_min_points"]):
            continue

        run_groups = split_direction_group_into_runs(
            coords=group_points,
            tangents=group_tangent_vectors,
            params=params,
        )

        for run_indices in run_groups:
            run_indices = run_indices.astype(np.int32)
            subgroup_points = group_points[run_indices]
            subgroup_tangents = group_tangent_vectors[run_indices]
            subgroup_member_indices = group_member_indices[run_indices]
            if subgroup_points.shape[0] < int(params["segment_min_points"]):
                continue
            metrics = compute_subgroup_metrics(
                points=subgroup_points,
                tangents=subgroup_tangents,
            )
            if metrics is None:
                continue
            quality = classify_trigger_subgroup(metrics, cluster_spacing, params)
            candidate_groups.append(
                {
                    "subgroup_id": int(subgroup_counter),
                    "member_indices": subgroup_member_indices,
                    "tangents": subgroup_tangents.astype(np.float32),
                    "quality": quality,
                    "metrics": metrics,
                    "direction_group": int(direction_group),
                }
            )
            subgroup_counter += 1

    main_groups = [group for group in candidate_groups if group["quality"] == "main"]
    fragment_groups = [group for group in candidate_groups if group["quality"] == "fragment"]
    bad_groups = [group for group in candidate_groups if group["quality"] == "bad"]

    if not main_groups and fragment_groups:
        best_fragment = max(
            fragment_groups,
            key=lambda group: (
                group["metrics"]["point_count"],
                group["metrics"]["linearity"],
                group["metrics"]["length"],
            ),
        )
        best_fragment["quality"] = "main"
        main_groups = [best_fragment]
        fragment_groups = [group for group in fragment_groups if group["subgroup_id"] != best_fragment["subgroup_id"]]

    main_bundles = [
        {
            "seed_subgroup_id": int(group["subgroup_id"]),
            "groups": [group],
            "merged_fragment_ids": [],
            "merged_main_ids": [],
            "direction": group["metrics"]["direction"],
            "line_support": group["metrics"]["line_support"],
        }
        for group in main_groups
    ]
    for bundle in main_bundles:
        refresh_main_bundle_geometry(bundle, points)

    attach_dist_th = float(params["trigger_fragment_attach_dist_scale"]) * cluster_spacing
    attach_along_gap_th = float(params["trigger_fragment_attach_gap_scale"]) * cluster_spacing
    attach_cos_th = float(params["trigger_fragment_attach_cos_th"])

    for fragment in fragment_groups:
        best_bundle = try_attach_group_to_main_bundle(
            group=fragment,
            main_bundles=main_bundles,
            points=points,
            attach_cos_th=attach_cos_th,
            attach_dist_th=attach_dist_th,
            attach_along_gap_th=attach_along_gap_th,
        )
        if best_bundle is not None:
            best_bundle["groups"].append(fragment)
            best_bundle["merged_fragment_ids"].append(int(fragment["subgroup_id"]))
            refresh_main_bundle_geometry(best_bundle, points)

    merge_direction_cos_th = float(params["trigger_main_merge_cos_th"])
    merge_line_dist_th = float(params["trigger_main_merge_dist_scale"]) * cluster_spacing
    merge_gap_th = float(params["trigger_main_merge_gap_scale"]) * cluster_spacing
    merge_lateral_th = float(params["trigger_main_merge_lateral_scale"]) * cluster_spacing

    changed = True
    while changed:
        changed = False
        for left in range(len(main_bundles)):
            if changed:
                break
            for right in range(left + 1, len(main_bundles)):
                if not should_merge_main_bundles(
                    bundle_a=main_bundles[left],
                    bundle_b=main_bundles[right],
                    points=points,
                    direction_cos_th=merge_direction_cos_th,
                    line_dist_th=merge_line_dist_th,
                    along_gap_th=merge_gap_th,
                    lateral_offset_th=merge_lateral_th,
                ):
                    continue

                main_bundles[left]["groups"].extend(main_bundles[right]["groups"])
                main_bundles[left]["merged_fragment_ids"].extend(main_bundles[right]["merged_fragment_ids"])
                main_bundles[left]["merged_main_ids"].append(int(main_bundles[right]["seed_subgroup_id"]))
                main_bundles[left]["merged_main_ids"].extend(main_bundles[right]["merged_main_ids"])
                refresh_main_bundle_geometry(main_bundles[left], points)
                del main_bundles[right]
                changed = True
                break

    final_bundles = []
    for bundle in main_bundles:
        member_indices = merge_subgroup_points(bundle["groups"])
        if member_indices.shape[0] < int(params["segment_min_points"]):
            continue
        final_bundles.append(
            {
                "seed_subgroup_id": int(bundle["seed_subgroup_id"]),
                "member_indices": member_indices,
            }
        )

    if not final_bundles:
        final_bundles = [
            {
                "seed_subgroup_id": -1,
                "member_indices": np.arange(points.shape[0], dtype=np.int32),
            }
        ]

    unused_bad_indices = merge_subgroup_points(bad_groups)

    visualization_rows = []
    merge_target_by_fragment = {}
    merge_target_by_main = {}
    for bundle in main_bundles:
        for fragment_id in bundle["merged_fragment_ids"]:
            merge_target_by_fragment[int(fragment_id)] = int(bundle["seed_subgroup_id"])
        for main_id in bundle["merged_main_ids"]:
            merge_target_by_main[int(main_id)] = int(bundle["seed_subgroup_id"])

    for group in candidate_groups:
        members = points[group["member_indices"]]
        color = classify_color(group["quality"])
        cluster_col = np.full((members.shape[0], 1), float(cluster_record["cluster_id"]), dtype=np.float32)
        subgroup_col = np.full((members.shape[0], 1), float(group["subgroup_id"]), dtype=np.float32)
        if group["quality"] == "main":
            class_id = 1.0
        elif group["quality"] == "fragment":
            class_id = 2.0
        else:
            class_id = 3.0
        class_col = np.full((members.shape[0], 1), class_id, dtype=np.float32)
        merge_target = float(
            merge_target_by_fragment.get(
                int(group["subgroup_id"]),
                merge_target_by_main.get(int(group["subgroup_id"]), -1),
            )
        )
        merge_col = np.full((members.shape[0], 1), merge_target, dtype=np.float32)
        visualization_rows.append(
            np.concatenate(
                [members.astype(np.float32), np.tile(color[None, :], (members.shape[0], 1)), cluster_col, subgroup_col, class_col, merge_col],
                axis=1,
            )
        )

    return final_bundles, visualization_rows, unused_bad_indices


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


def export_npz(path: Path, payload: dict) -> None:
    """Export npz."""
    save_npz(path, payload)


def sample_segment_geometry(start: np.ndarray, end: np.ndarray, step: float) -> np.ndarray:
    """Sample one segment densely enough for visualization."""
    length = float(np.linalg.norm(end - start))
    num = max(int(np.ceil(length / max(step, 1e-4))) + 1, 2)
    alpha = np.linspace(0.0, 1.0, num=num, dtype=np.float32)
    return ((1.0 - alpha[:, None]) * start[None, :] + alpha[:, None] * end[None, :]).astype(np.float32)


def export_support_geometry_xyz(supports_payload: dict, output_dir: Path, sample_step: float = 0.02) -> None:
    """Export support geometry sampling for CloudCompare-style inspection."""
    rows = []
    for support_id in range(int(supports_payload["support_id"].shape[0])):
        support_type = int(supports_payload["support_type"][support_id])
        offset = int(supports_payload["segment_offset"][support_id])
        length = int(supports_payload["segment_length"][support_id])

        seed = (support_id * 1103515245 + 12345) & 0x7FFFFFFF
        color = np.array(
            [64 + (seed & 127), 64 + ((seed >> 8) & 127), 64 + ((seed >> 16) & 127)],
            dtype=np.float32,
        )

        for seg_idx in range(offset, offset + length):
            points = sample_segment_geometry(
                start=supports_payload["segment_start"][seg_idx],
                end=supports_payload["segment_end"][seg_idx],
                step=sample_step,
            )
            support_col = np.full((points.shape[0], 1), float(support_id), dtype=np.float32)
            type_col = np.full((points.shape[0], 1), float(support_type), dtype=np.float32)
            rows.append(
                np.concatenate(
                    [points, np.tile(color[None, :], (points.shape[0], 1)), support_col, type_col],
                    axis=1,
                )
            )

    if rows:
        data = np.concatenate(rows, axis=0).astype(np.float32)
    else:
        data = np.empty((0, 8), dtype=np.float32)

    save_xyz(
        output_dir / "support_geometry.xyz",
        data,
        ["%.6f", "%.6f", "%.6f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f"],
    )


def export_trigger_group_classes_xyz(debug_payload: dict, output_dir: Path) -> None:
    """Export trigger candidate subgroup classes for manual inspection."""
    data = debug_payload["trigger_group_visualization"]
    save_xyz(
        output_dir / "trigger_group_classes.xyz",
        data,
        ["%.6f", "%.6f", "%.6f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f"],
    )
