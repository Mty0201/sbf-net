"""
Build fine-grained local clusters for BF Edge v3.

Bottom-up micro-cluster merge algorithm:
1. Per semantic_pair: DBSCAN with small eps -> many tight micro-clusters
2. Direction-aware merge: adjacent micro-clusters with compatible tangents
   are merged via union-find
3. Post-merge rescue: noise points assigned to nearest merged cluster

This replaces the previous split-based pipeline (large DBSCAN -> direction
grouping -> spatial splitting -> outlier pruning) which suffered from
cross-edge contamination when splitting large connected clusters.

Phase 6 redesign: bottom-up merge structurally prevents cross-edge supports.
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


# -------------------------------------------------------------------------
# Shared utilities
# -------------------------------------------------------------------------


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


def group_tangents(
    tangents: np.ndarray,
    direction_cos_th: float,
) -> np.ndarray:
    """Simple sign-invariant tangent grouping.

    Groups tangent vectors by direction similarity (sign-invariant cosine).
    Each group contains tangents whose pairwise |cos(angle)| >= direction_cos_th.

    Retained for use by validate_cluster_contract().
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


# -------------------------------------------------------------------------
# Bottom-up micro-cluster merge
# -------------------------------------------------------------------------


def _lateral_bimodal_split_1d(
    lat_1d: np.ndarray,
    split_threshold: float,
) -> float | None:
    """Detect split point in a 1D lateral projection via largest gap.

    Finds the largest gap in sorted lateral positions. If the gap exceeds
    ``split_threshold``, returns the midpoint as the split coordinate.

    This is simpler and more robust than Otsu for the parallel-edge case:
    two edges separated by a physical gap always produce a large gap in
    the sorted lateral projection, regardless of point-count asymmetry.

    Returns the split coordinate, or ``None`` if no gap exceeds threshold.
    """
    n = len(lat_1d)
    if n < 2:
        return None
    sorted_lat = np.sort(lat_1d)
    gaps = np.diff(sorted_lat)
    max_gap_idx = int(np.argmax(gaps))
    max_gap = float(gaps[max_gap_idx])
    if max_gap < split_threshold:
        return None
    return float((sorted_lat[max_gap_idx] + sorted_lat[max_gap_idx + 1]) / 2)


def _recursive_lateral_split(
    g_local: np.ndarray,
    g_pts: np.ndarray,
    g_tgts: np.ndarray,
    split_threshold: float,
    min_pts: int,
) -> list[np.ndarray]:
    """Recursively split a direction group at lateral gaps.

    Finds the largest lateral gap (perpendicular to the group's mean
    tangent) and splits there.  Recurses on each half so that 3+ parallel
    lines are fully separated, not just bisected once.

    Returns a list of local-index arrays (indices into the parent
    micro-cluster's point array, same semantics as ``g_local``).
    """
    if g_pts.shape[0] < 2 * min_pts:
        return [g_local]

    # Mean tangent (sign-aligned)
    ref = g_tgts[0]
    dots = g_tgts @ ref
    signs = np.where(dots >= 0, 1.0, -1.0)
    mt = (g_tgts * signs[:, None]).mean(axis=0)
    mt_norm = np.linalg.norm(mt)
    if mt_norm < 1e-9:
        return [g_local]
    mt = mt / mt_norm

    # Lateral projection perpendicular to this tangent
    centroid = g_pts.mean(axis=0)
    diffs = g_pts - centroid
    lateral_proj = diffs - (diffs @ mt)[:, None] * mt[None, :]
    lat_max = np.linalg.norm(lateral_proj, axis=1).max()
    if lat_max < split_threshold:
        return [g_local]

    # 1D lateral axis via SVD
    _, _, vh_lat = np.linalg.svd(lateral_proj, full_matrices=False)
    lat_1d = lateral_proj @ vh_lat[0]

    sp = _lateral_bimodal_split_1d(lat_1d, split_threshold)
    if sp is None:
        return [g_local]

    left_mask = lat_1d < sp
    right_mask = ~left_mask
    left_local = g_local[left_mask]
    right_local = g_local[right_mask]

    fragments: list[np.ndarray] = []
    for frag_local, frag_mask in ((left_local, left_mask), (right_local, right_mask)):
        if frag_local.shape[0] < min_pts:
            continue
        fragments.extend(
            _recursive_lateral_split(
                frag_local, g_pts[frag_mask], g_tgts[frag_mask],
                split_threshold, min_pts,
            )
        )
    return fragments if fragments else [g_local]


def _split_bimodal_clusters(
    pair_coords: np.ndarray,
    pair_tangents: np.ndarray,
    labels: np.ndarray,
    micro_ids: list[int],
    global_median_spacing: float,
    config: Stage2Config,
) -> tuple[np.ndarray, list[int]]:
    """Split micro-clusters that contain parallel edges.

    A single DBSCAN micro-cluster can span an entire rectangular frame
    (e.g. window/door) because the horizontal edge physically bridges
    two parallel vertical edges, making them one connected component.

    Strategy: for each micro-cluster, group points by tangent direction.
    Each direction group is recursively checked for lateral gaps and split
    until no fragment contains a gap exceeding the threshold.  This handles
    3+ parallel lines (e.g. window mullions) that a single binary split
    would leave partially merged.

    Returns updated (labels, micro_ids) with new IDs for split fragments.
    """
    split_threshold = config.split_lateral_threshold_scale * global_median_spacing
    cos_th = config.merge_direction_cos_th
    min_pts = config.min_cluster_points
    next_id = max(micro_ids) + 1 if micro_ids else 0

    new_labels = labels.copy()
    new_micro_ids = list(micro_ids)

    for mid in list(micro_ids):
        mask = new_labels == mid
        global_idx = np.where(mask)[0]
        pts = pair_coords[global_idx]
        tgts = pair_tangents[global_idx]
        n = pts.shape[0]
        if n < 2 * min_pts:
            continue

        # Group by tangent direction
        dir_labels = group_tangents(tgts, cos_th)
        dir_ids = sorted(set(int(x) for x in dir_labels if x >= 0))

        # Check if any direction group has bimodal lateral distribution
        any_split = False
        fragments: list[np.ndarray] = []  # list of local index arrays

        for gid in dir_ids:
            g_local = np.where(dir_labels == gid)[0]
            g_pts = pts[g_local]
            g_tgts = tgts[g_local]

            sub_fragments = _recursive_lateral_split(
                g_local, g_pts, g_tgts, split_threshold, min_pts,
            )
            if len(sub_fragments) > 1:
                any_split = True
            fragments.extend(sub_fragments)

        if not any_split or len(fragments) < 2:
            continue

        # Reassign: keep original mid for first fragment, new IDs for rest.
        # First, mark all points in this micro-cluster as unassigned.
        new_labels[global_idx] = -1

        for fi, frag_local in enumerate(fragments):
            frag_global = global_idx[frag_local]
            if frag_global.shape[0] < min_pts:
                continue  # too small, stays as noise (will be rescued)
            if fi == 0:
                new_labels[frag_global] = mid
            else:
                new_labels[frag_global] = next_id
                new_micro_ids.append(next_id)
                next_id += 1

    return new_labels, new_micro_ids


def _compute_micro_cluster_tangents(
    pair_tangents: np.ndarray,
    labels: np.ndarray,
    micro_ids: list[int],
) -> np.ndarray:
    """Compute sign-invariant mean tangent per micro-cluster.

    Returns (len(micro_ids), 3) float64 array of unit tangent vectors.
    """
    result = np.zeros((len(micro_ids), 3), dtype=np.float64)
    for i, mid in enumerate(micro_ids):
        mask = labels == mid
        tgts = normalize_rows(pair_tangents[mask])
        if tgts.shape[0] == 0:
            continue
        ref = tgts[0]
        dots = tgts @ ref
        signs = np.where(dots >= 0, 1.0, -1.0)
        aligned = tgts * signs[:, None]
        mt = aligned.mean(axis=0)
        n = np.linalg.norm(mt)
        if n > 1e-9:
            result[i] = mt / n
    return result


def _merged_has_lateral_gap(
    coords_list: list[np.ndarray],
    tangents_list: list[np.ndarray],
    cos_th: float,
    split_threshold: float,
    min_pts: int,
) -> bool:
    """Check if merging micro-clusters would create a lateral gap.

    Simulates the merged point cloud: groups by tangent direction, then
    checks each direction group for a lateral gap exceeding split_threshold.
    Returns True if a gap is found (merge should be rejected).
    """
    all_pts = np.concatenate(coords_list, axis=0)
    all_tgts = np.concatenate(tangents_list, axis=0)
    if all_pts.shape[0] < 2 * min_pts:
        return False

    dir_labels = group_tangents(all_tgts, cos_th)
    for gid in set(int(x) for x in dir_labels if x >= 0):
        g_mask = dir_labels == gid
        g_pts = all_pts[g_mask]
        g_tgts = all_tgts[g_mask]
        if g_pts.shape[0] < 2 * min_pts:
            continue
        # Mean tangent for direction group
        ref = g_tgts[0]
        dots = g_tgts @ ref
        signs = np.where(dots >= 0, 1.0, -1.0)
        mt = (g_tgts * signs[:, None]).mean(axis=0)
        mt_norm = np.linalg.norm(mt)
        if mt_norm < 1e-9:
            continue
        mt = mt / mt_norm
        # Lateral projection
        centroid = g_pts.mean(axis=0)
        diffs = g_pts - centroid
        lateral_proj = diffs - (diffs @ mt)[:, None] * mt[None, :]
        if np.linalg.norm(lateral_proj, axis=1).max() < split_threshold:
            continue
        _, _, vh = np.linalg.svd(lateral_proj, full_matrices=False)
        lat_1d = lateral_proj @ vh[0]
        if _lateral_bimodal_split_1d(lat_1d, split_threshold) is not None:
            return True
    return False


def _merge_micro_clusters(
    centroids: np.ndarray,
    mean_tangents: np.ndarray,
    merge_radius: float,
    cos_th: float,
    lateral_max: float,
    micro_coords: list[np.ndarray],
    micro_tangents: list[np.ndarray],
    split_threshold: float,
    min_pts: int,
) -> np.ndarray:
    """Direction-aware merge of micro-clusters via union-find.

    Two micro-clusters merge only if:
    1. Centroid distance <= merge_radius
    2. Direction compatibility: |cos(angle)| >= cos_th
    3. Lateral offset <= lateral_max (perpendicular to shared tangent)
    4. Merged point cloud has no lateral gap (prevents re-merging split fragments)

    Condition 4 simulates the merged component's point cloud and rejects
    the merge if any direction group would exhibit a bimodal lateral gap,
    which is exactly the condition that _split_bimodal_clusters split on.

    Returns array of root IDs (one per micro-cluster).
    """
    n = centroids.shape[0]
    parent = np.arange(n, dtype=np.int32)
    # Track which micro-cluster indices belong to each component root
    component_members: dict[int, list[int]] = {i: [i] for i in range(n)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            component_members[rb] = component_members[rb] + component_members.pop(ra)

    # Build radius neighbor graph
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(radius=merge_radius).fit(centroids)
    _, adj_indices = nn.radius_neighbors(centroids)

    for i in range(n):
        for j in adj_indices[i]:
            j = int(j)
            if j <= i:
                continue
            if find(i) == find(j):
                continue
            cos_sim = abs(float(np.dot(mean_tangents[i], mean_tangents[j])))
            if cos_sim < cos_th:
                continue
            # Lateral offset check: project centroid difference onto
            # the plane perpendicular to the shared tangent direction.
            avg_tangent = mean_tangents[i] + mean_tangents[j]
            tn = np.linalg.norm(avg_tangent)
            if tn < 1e-9:
                continue
            avg_tangent = avg_tangent / tn
            diff = centroids[j] - centroids[i]
            along = float(np.dot(diff, avg_tangent))
            lateral = float(np.sqrt(max(np.dot(diff, diff) - along * along, 0.0)))
            if lateral > lateral_max:
                continue
            # Lateral gap check on simulated merged component
            ri, rj = find(i), find(j)
            merged_indices = component_members[ri] + component_members[rj]
            if _merged_has_lateral_gap(
                [micro_coords[k] for k in merged_indices],
                [micro_tangents[k] for k in merged_indices],
                cos_th, split_threshold, min_pts,
            ):
                continue
            union(i, j)

    # Normalize all parents to roots
    roots = np.array([find(i) for i in range(n)], dtype=np.int32)
    return roots


def _cluster_one_pair(
    pair_coords: np.ndarray,
    pair_tangents: np.ndarray,
    config: Stage2Config,
    global_median_spacing: float,
) -> tuple[np.ndarray, int]:
    """Run micro-cluster + merge + rescue for one semantic pair.

    Returns:
        merged_labels: (N,) int32, -1 for unassigned noise
        num_rescued: number of noise points rescued
    """
    n = pair_coords.shape[0]
    eps_micro = config.micro_eps_scale * global_median_spacing
    merge_radius = config.merge_radius_scale * global_median_spacing
    rescue_radius = config.rescue_radius_scale * global_median_spacing
    cos_th = config.merge_direction_cos_th

    # Step 1: Micro-cluster
    labels = DBSCAN(eps=eps_micro, min_samples=config.micro_min_samples).fit_predict(pair_coords)
    labels = labels.astype(np.int32)
    micro_ids = sorted(set(int(x) for x in labels if x >= 0))

    if not micro_ids:
        return np.full(n, -1, dtype=np.int32), 0

    # Step 1.5: Split micro-clusters with bimodal lateral distribution
    labels, micro_ids = _split_bimodal_clusters(
        pair_coords, pair_tangents, labels, micro_ids, global_median_spacing, config,
    )

    # Prune micro-cluster IDs that lost all points during the split
    micro_ids = [m for m in micro_ids if np.any(labels == m)]
    if not micro_ids:
        return np.full(n, -1, dtype=np.int32), 0

    # Step 2: Compute centroids, mean tangents, and per-micro-cluster point arrays
    centroids = np.array([pair_coords[labels == m].mean(axis=0) for m in micro_ids])
    mean_tangents = _compute_micro_cluster_tangents(pair_tangents, labels, micro_ids)
    micro_coords = [pair_coords[labels == m] for m in micro_ids]
    micro_tgts = [pair_tangents[labels == m] for m in micro_ids]

    # Step 3: Direction-aware merge with lateral offset + lateral gap guard
    lateral_max = config.merge_lateral_scale * global_median_spacing
    split_threshold = config.split_lateral_threshold_scale * global_median_spacing
    roots = _merge_micro_clusters(
        centroids, mean_tangents, merge_radius, cos_th, lateral_max,
        micro_coords, micro_tgts, split_threshold, config.min_cluster_points,
    )

    # Map root IDs to contiguous merged cluster IDs
    unique_roots = np.unique(roots)
    root_to_merged = {int(r): i for i, r in enumerate(unique_roots)}

    # Assign merged labels to points
    merged_labels = np.full(n, -1, dtype=np.int32)
    for i, mid in enumerate(micro_ids):
        mask = labels == mid
        merged_labels[mask] = root_to_merged[int(roots[i])]

    # Step 4: Post-merge rescue
    num_rescued = 0
    noise_idx = np.flatnonzero(merged_labels == -1)
    if noise_idx.size > 0:
        cluster_mask = merged_labels >= 0
        if cluster_mask.any():
            tree = cKDTree(pair_coords[cluster_mask])
            cluster_ids_flat = merged_labels[cluster_mask]
            dists, nn_idx = tree.query(pair_coords[noise_idx], k=1)
            for ni_local, (d, ci) in enumerate(zip(dists, nn_idx)):
                if float(d) <= rescue_radius:
                    merged_labels[noise_idx[ni_local]] = int(cluster_ids_flat[ci])
                    num_rescued += 1

    return merged_labels, num_rescued


# -------------------------------------------------------------------------
# Main clustering function
# -------------------------------------------------------------------------


def cluster_boundary_centers(
    boundary_centers: dict,
    config: Stage2Config,
) -> tuple[dict, dict]:
    """Bottom-up micro-cluster merge: micro-cluster -> merge -> rescue.

    Each output cluster is a spatially-contiguous, direction-consistent
    group of boundary centers that directly satisfies Stage 3's fitter
    assumptions by construction.
    """
    center_coord = boundary_centers["center_coord"]
    center_tangent = boundary_centers["center_tangent"]
    semantic_pair = boundary_centers["semantic_pair"]

    num_centers = center_coord.shape[0]

    # Global median 1-NN spacing drives all adaptive thresholds.
    # Uses 1-NN (not k-NN mean) to match the micro-cluster scale.
    if num_centers >= 2:
        tree = cKDTree(center_coord)
        d1, _ = tree.query(center_coord, k=2, workers=-1)
        global_median_spacing = float(np.median(d1[:, 1]))
    else:
        global_median_spacing = 0.01  # fallback

    num_rescued = 0
    run_records: list[tuple[np.ndarray, np.ndarray]] = []

    unique_pairs = np.unique(semantic_pair, axis=0)
    for pair in unique_pairs:
        pair_mask = np.all(semantic_pair == pair[None, :], axis=1)
        pair_indices = np.where(pair_mask)[0].astype(np.int32)
        pair_coords = center_coord[pair_indices]
        pair_tangents = center_tangent[pair_indices]

        if pair_indices.size < config.micro_min_samples:
            continue

        merged_labels, pair_rescued = _cluster_one_pair(
            pair_coords, pair_tangents, config, global_median_spacing,
        )
        num_rescued += pair_rescued

        # Collect runs (one per merged cluster)
        for cid in np.unique(merged_labels):
            if cid < 0:
                continue
            cmask = merged_labels == int(cid)
            global_indices = pair_indices[cmask]
            if global_indices.shape[0] >= config.min_cluster_points:
                run_records.append((pair.astype(np.int32), global_indices.astype(np.int32)))

    # Build output payload (same format as before)
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
        "params": {
            "micro_eps_scale": float(config.micro_eps_scale),
            "micro_min_samples": int(config.micro_min_samples),
            "split_lateral_threshold_scale": float(config.split_lateral_threshold_scale),
            "merge_radius_scale": float(config.merge_radius_scale),
            "merge_direction_angle_deg": float(config.merge_direction_angle_deg),
            "merge_lateral_scale": float(config.merge_lateral_scale),
            "rescue_radius_scale": float(config.rescue_radius_scale),
            "min_cluster_points": int(config.min_cluster_points),
        },
    }
    return payload, meta


# -------------------------------------------------------------------------
# Visualization helpers (unchanged)
# -------------------------------------------------------------------------


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
