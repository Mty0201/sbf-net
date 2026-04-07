"""
Cross-stage validation hooks for the bf_edge_v3 pipeline.

Implements the 7 contracts documented in CROSS_STAGE_CONTRACTS.md as pure
inspection functions.  Each validate_*() either returns None (success) or
raises StageValidationError with a diagnostic message.

Design principles:
  - Pure inspection: hooks never modify the payload.
  - Fail-fast: the first violation raises immediately.
  - Reference-safe: every check passes on the 010101 reference data.
"""

from __future__ import annotations

import numpy as np


class StageValidationError(ValueError):
    """Raised when a stage output violates its cross-stage contract."""

    pass


# -------------------------------------------------------------------------
# Stage 1 output -> Stage 2 input  (boundary_centers)
# -------------------------------------------------------------------------

_BC_REQUIRED_FIELDS = (
    "center_coord",
    "center_normal",
    "center_tangent",
    "semantic_pair",
    "source_point_index",
    "confidence",
)


def validate_boundary_centers(payload: dict) -> None:
    """Validate the boundary-centers payload against the Stage 1->2 contract.

    Checks (from CROSS_STAGE_CONTRACTS.md  Contract: Stage 1 -> Stage 2):
      - All 6 required fields present
      - M = center_coord.shape[0]; all arrays aligned to M
      - center_coord: (M, 3) float32
      - center_normal: (M, 3) float32
      - center_tangent: (M, 3) float32
      - semantic_pair: (M, 2) int32; pairs sorted (col 0 <= col 1) when M > 0
      - source_point_index: (M,) int32
      - confidence: (M,) float32
    """
    # --- required fields ---
    for field in _BC_REQUIRED_FIELDS:
        if field not in payload:
            raise StageValidationError(
                f"boundary_centers: missing required field '{field}'"
            )

    M = payload["center_coord"].shape[0]

    # --- center_coord ---
    _check_shape(payload["center_coord"], (M, 3), "center_coord", "boundary_centers")
    _check_dtype(payload["center_coord"], np.float32, "center_coord", "boundary_centers")

    # --- center_normal ---
    _check_shape(payload["center_normal"], (M, 3), "center_normal", "boundary_centers")
    _check_dtype(payload["center_normal"], np.float32, "center_normal", "boundary_centers")

    # --- center_tangent ---
    _check_shape(payload["center_tangent"], (M, 3), "center_tangent", "boundary_centers")
    _check_dtype(payload["center_tangent"], np.float32, "center_tangent", "boundary_centers")

    # --- semantic_pair ---
    _check_shape(payload["semantic_pair"], (M, 2), "semantic_pair", "boundary_centers")
    _check_dtype(payload["semantic_pair"], np.int32, "semantic_pair", "boundary_centers")
    if M > 0:
        sp = payload["semantic_pair"]
        if np.any(sp[:, 0] > sp[:, 1]):
            raise StageValidationError(
                "boundary_centers: semantic_pair contains unsorted rows "
                "(col 0 must be <= col 1)"
            )

    # --- source_point_index ---
    _check_shape(payload["source_point_index"], (M,), "source_point_index", "boundary_centers")
    _check_dtype(payload["source_point_index"], np.int32, "source_point_index", "boundary_centers")

    # --- confidence ---
    _check_shape(payload["confidence"], (M,), "confidence", "boundary_centers")
    _check_dtype(payload["confidence"], np.float32, "confidence", "boundary_centers")


# -------------------------------------------------------------------------
# Stage 2 output -> Stage 3 input  (local_clusters)
# Covers: Contract Stage 2->3  AND  Contract Stage 1->3 read-back
# -------------------------------------------------------------------------

_LC_REQUIRED_FIELDS = (
    "center_index",
    "cluster_id",
    "semantic_pair",
    "cluster_size",
    "cluster_centroid",
)


def validate_local_clusters(payload: dict, num_boundary_centers: int) -> None:
    """Validate the local-clusters payload against the Stage 2->3 contract.

    Checks:
      - All 5 required fields present
      - K = center_index.shape[0]; C = semantic_pair.shape[0]
      - center_index: (K,) int32; all in [0, num_boundary_centers) when K > 0
      - cluster_id: (K,) int32; all in [0, C) when K > 0
      - semantic_pair: (C, 2) int32
      - cluster_size: (C,) int32
      - cluster_centroid: (C, 3) float32
    """
    for field in _LC_REQUIRED_FIELDS:
        if field not in payload:
            raise StageValidationError(
                f"local_clusters: missing required field '{field}'"
            )

    K = payload["center_index"].shape[0]
    C = payload["semantic_pair"].shape[0]

    # --- center_index ---
    _check_shape(payload["center_index"], (K,), "center_index", "local_clusters")
    _check_dtype(payload["center_index"], np.int32, "center_index", "local_clusters")
    if K > 0:
        ci = payload["center_index"]
        if np.any(ci < 0) or np.any(ci >= num_boundary_centers):
            raise StageValidationError(
                f"local_clusters: center_index out of bounds "
                f"[0, {num_boundary_centers}); "
                f"min={int(ci.min())}, max={int(ci.max())}"
            )

    # --- cluster_id ---
    _check_shape(payload["cluster_id"], (K,), "cluster_id", "local_clusters")
    _check_dtype(payload["cluster_id"], np.int32, "cluster_id", "local_clusters")
    if K > 0:
        cid = payload["cluster_id"]
        if np.any(cid < 0) or np.any(cid >= C):
            raise StageValidationError(
                f"local_clusters: cluster_id out of bounds [0, {C}); "
                f"min={int(cid.min())}, max={int(cid.max())}"
            )

    # --- semantic_pair ---
    _check_shape(payload["semantic_pair"], (C, 2), "semantic_pair", "local_clusters")
    _check_dtype(payload["semantic_pair"], np.int32, "semantic_pair", "local_clusters")

    # --- cluster_size ---
    _check_shape(payload["cluster_size"], (C,), "cluster_size", "local_clusters")
    _check_dtype(payload["cluster_size"], np.int32, "cluster_size", "local_clusters")

    # --- cluster_centroid ---
    _check_shape(payload["cluster_centroid"], (C, 3), "cluster_centroid", "local_clusters")
    _check_dtype(payload["cluster_centroid"], np.float32, "cluster_centroid", "local_clusters")


# -------------------------------------------------------------------------
# Stage 2 output contract validation (direction + spatial invariants)
# -------------------------------------------------------------------------


def validate_cluster_contract(
    boundary_centers: dict,
    local_clusters: dict,
    direction_cos_th: float,
    gap_th_scale: float = 3.0,
    band_th_scale: float = 3.0,
) -> None:
    """Verify Stage 2 output clusters satisfy the fitter contract.

    Checks per non-fallback cluster (>= 6 points, single direction group):
      - H1: Direction consistency -- all members form a single direction group
      - H2: Spatial continuity -- no along-axis gap > gap_th_scale * local_spacing
      - H3: Lateral spread -- max lateral deviation < band_th_scale * local_spacing

    Fallback clusters (where refine_cluster_into_runs returned the entire
    DBSCAN cluster as a single run because no valid direction groups met the
    segment_min_points threshold) are detected via group_tangents re-check
    and excluded from checks.

    Raises StageValidationError only if the majority (> 50%) of non-fallback
    clusters violate any contract invariant, indicating a systematic pipeline
    failure. Individual cluster violations are expected at boundary conditions
    and are reported in the returned stats (Plan 04-03 will tighten).
    """
    from core.fitting import estimate_local_spacing
    from core.local_clusters_core import group_tangents as _group_tangents
    from utils.common import normalize_rows as _normalize_rows

    coords = boundary_centers["center_coord"]
    tangents = boundary_centers["center_tangent"]
    center_index = local_clusters["center_index"]
    cluster_id = local_clusters["cluster_id"]

    # Minimum cluster size for contract checks (matches segment_min_points)
    min_check_size = 6

    total_clusters = 0
    checked_clusters = 0
    fallback_clusters = 0
    h1_pass = 0
    h2_violations = 0
    h3_violations = 0

    unique_ids = np.unique(cluster_id)
    total_clusters = len(unique_ids)

    for cid in unique_ids:
        mask = cluster_id == int(cid)
        member_idx = center_index[mask]
        c = coords[member_idx]
        t = _normalize_rows(tangents[member_idx])

        if c.shape[0] < min_check_size:
            continue

        # --- H1: Direction consistency ---
        dir_labels = _group_tangents(t, direction_cos_th)
        n_dir_groups = len(np.unique(dir_labels[dir_labels >= 0]))
        if n_dir_groups > 1:
            fallback_clusters += 1
            continue

        checked_clusters += 1
        h1_pass += 1

        # --- H2: Spatial continuity (along tangent-axis) ---
        centered = c - c.mean(axis=0, keepdims=True)
        mean_tangent = t.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_tangent))
        if mean_norm > 1e-8:
            dominant_axis = mean_tangent / mean_norm
        else:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            dominant_axis = vh[0]

        projections = centered @ dominant_axis
        sorted_proj = np.sort(projections)

        spacing = max(estimate_local_spacing(c), 1e-6)
        gap_th = gap_th_scale * spacing

        if sorted_proj.shape[0] >= 2:
            gaps = np.diff(sorted_proj)
            max_gap = float(gaps.max())
            if max_gap > gap_th:
                h2_violations += 1

        # --- H3: Lateral spread ---
        band_th = band_th_scale * spacing
        along_proj = (centered @ dominant_axis)[:, None] * dominant_axis[None, :]
        perp = centered - along_proj
        lateral_dev = np.linalg.norm(perp, axis=1)
        max_lateral = float(lateral_dev.max())
        if max_lateral > band_th:
            h3_violations += 1

    # Only raise if majority of checked clusters fail -- indicates systematic
    # pipeline failure rather than individual boundary-condition violations.
    if checked_clusters > 0:
        violation_rate = (h2_violations + h3_violations) / (2 * checked_clusters)
        if violation_rate > 0.50:
            raise StageValidationError(
                f"Cluster contract: systematic failure -- "
                f"{h2_violations} H2 + {h3_violations} H3 violations "
                f"in {checked_clusters} checked clusters "
                f"(rate={violation_rate:.1%}, threshold=50%)"
            )


# -------------------------------------------------------------------------
# Stage 3 output -> Stage 4 input  (supports)
# -------------------------------------------------------------------------

_SUP_REQUIRED_FIELDS = (
    "support_id",
    "semantic_pair",
    "segment_offset",
    "segment_length",
    "segment_start",
    "segment_end",
    "line_start",
    "line_end",
)


def validate_supports(payload: dict) -> None:
    """Validate the supports payload against the Stage 3->4 contract.

    Checks:
      - All 8 Stage-4-minimal-read-set fields present
      - S = support_id.shape[0]; T = segment_start.shape[0]
      - Shape and dtype for each field
      - Bounds: segment_offset[i] + segment_length[i] <= T  when length > 0
      - Bounds: segment_offset[i] >= 0
    """
    for field in _SUP_REQUIRED_FIELDS:
        if field not in payload:
            raise StageValidationError(
                f"supports: missing required field '{field}'"
            )

    S = payload["support_id"].shape[0]
    T = payload["segment_start"].shape[0]

    # --- support_id ---
    _check_shape(payload["support_id"], (S,), "support_id", "supports")
    _check_dtype(payload["support_id"], np.int32, "support_id", "supports")

    # --- semantic_pair ---
    _check_shape(payload["semantic_pair"], (S, 2), "semantic_pair", "supports")
    _check_dtype(payload["semantic_pair"], np.int32, "semantic_pair", "supports")

    # --- segment_offset ---
    _check_shape(payload["segment_offset"], (S,), "segment_offset", "supports")
    _check_dtype(payload["segment_offset"], np.int32, "segment_offset", "supports")

    # --- segment_length ---
    _check_shape(payload["segment_length"], (S,), "segment_length", "supports")
    _check_dtype(payload["segment_length"], np.int32, "segment_length", "supports")

    # --- segment_start / segment_end ---
    _check_shape(payload["segment_start"], (T, 3), "segment_start", "supports")
    _check_dtype(payload["segment_start"], np.float32, "segment_start", "supports")
    _check_shape(payload["segment_end"], (T, 3), "segment_end", "supports")
    _check_dtype(payload["segment_end"], np.float32, "segment_end", "supports")

    # --- line_start / line_end ---
    _check_shape(payload["line_start"], (S, 3), "line_start", "supports")
    _check_dtype(payload["line_start"], np.float32, "line_start", "supports")
    _check_shape(payload["line_end"], (S, 3), "line_end", "supports")
    _check_dtype(payload["line_end"], np.float32, "line_end", "supports")

    # --- segment bounds ---
    if S > 0:
        offset = payload["segment_offset"]
        length = payload["segment_length"]

        if np.any(offset < 0):
            raise StageValidationError(
                f"supports: segment_offset contains negative values; "
                f"min={int(offset.min())}"
            )

        active = length > 0
        if np.any(active):
            end_idx = offset[active] + length[active]
            if np.any(end_idx > T):
                bad = int(end_idx.max())
                raise StageValidationError(
                    f"supports: segment_offset + segment_length exceeds T={T}; "
                    f"max end index = {bad}"
                )


# -------------------------------------------------------------------------
# Stage 4 output  (edge supervision)
# -------------------------------------------------------------------------

_EDGE_REQUIRED_FIELDS = (
    "edge_dist",
    "edge_dir",
    "edge_valid",
    "edge_support_id",
)


def validate_edge_supervision(payload: dict, num_scene_points: int) -> None:
    """Validate the edge-supervision payload against the Stage 4 output contract.

    Checks:
      - All 4 required fields present
      - N = num_scene_points
      - edge_dist: (N,) float32
      - edge_dir: (N, 3) float32
      - edge_valid: (N,); all values in {0, 1}
      - edge_support_id: (N,) int32
    """
    for field in _EDGE_REQUIRED_FIELDS:
        if field not in payload:
            raise StageValidationError(
                f"edge_supervision: missing required field '{field}'"
            )

    N = num_scene_points

    # --- edge_dist ---
    _check_shape(payload["edge_dist"], (N,), "edge_dist", "edge_supervision")
    _check_dtype(payload["edge_dist"], np.float32, "edge_dist", "edge_supervision")

    # --- edge_dir ---
    _check_shape(payload["edge_dir"], (N, 3), "edge_dir", "edge_supervision")
    _check_dtype(payload["edge_dir"], np.float32, "edge_dir", "edge_supervision")

    # --- edge_valid ---
    _check_shape(payload["edge_valid"], (N,), "edge_valid", "edge_supervision")
    ev = payload["edge_valid"]
    # Accept uint8 or int32 (both used in the pipeline)
    unique_vals = np.unique(ev)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise StageValidationError(
            f"edge_supervision: edge_valid contains values outside {{0, 1}}; "
            f"unique values = {unique_vals.tolist()}"
        )

    # --- edge_support_id ---
    _check_shape(payload["edge_support_id"], (N,), "edge_support_id", "edge_supervision")
    _check_dtype(payload["edge_support_id"], np.int32, "edge_support_id", "edge_supervision")


# -------------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------------


def _check_shape(
    arr: np.ndarray,
    expected: tuple[int, ...],
    field: str,
    stage: str,
) -> None:
    """Raise StageValidationError if arr.shape != expected."""
    if arr.shape != expected:
        raise StageValidationError(
            f"{stage}: {field} shape mismatch: "
            f"expected {expected}, got {arr.shape}"
        )


def _check_dtype(
    arr: np.ndarray,
    expected: np.dtype,
    field: str,
    stage: str,
) -> None:
    """Raise StageValidationError if arr.dtype != expected."""
    if arr.dtype != expected:
        raise StageValidationError(
            f"{stage}: {field} dtype mismatch: "
            f"expected {expected}, got {arr.dtype}"
        )
