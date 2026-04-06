"""
Centralized parameter definitions for the bf_edge_v3 pipeline.

All values are the exact original defaults -- no behavioral changes.
Extracted during Phase 2 refactor to prepare for Phase 3 config injection (REF-04).

Parameter sections:
  1. Stage 3 fit parameters (DEFAULT_FIT_PARAMS)
  2. Stage 2 trigger parameters (DEFAULT_TRIGGER_PARAMS)
  3. Stage 2 denoise parameters (DEFAULT_DENOISE_PARAMS)
  4. Stage 4 parameters (documented as comments -- CLI args, not in-code dicts)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Section 1 -- Stage 3 fit parameters
# ---------------------------------------------------------------------------
# These control the trigger regrouping path in Stage 3 (supports_core.py).
# Angle parameters are stored in degrees; runtime conversion to cosine
# thresholds happens in build_runtime_params() (fit_local_supports.py /
# build_support_dataset_v3.py).

DEFAULT_FIT_PARAMS: dict[str, float | int] = {
    # -- Run splitting parameters --
    "segment_direction_angle_deg": 20.0,        # max angle between tangents for same direction group
    "segment_run_gap_scale": 3.0,               # along-axis gap threshold = scale * local_spacing
    "segment_run_lateral_gap_scale": 2.5,        # lateral gap threshold = scale * local_spacing
    "segment_run_lateral_band_scale": 3.0,       # lateral band width threshold = scale * local_spacing
    "segment_min_points": 6,                     # minimum points per run/subgroup to keep
    # -- Trigger main subgroup classification --
    "trigger_main_min_points": 12,               # main subgroup: minimum point count
    "trigger_main_linearity_th": 0.88,           # main subgroup: minimum PCA linearity
    "trigger_main_tangent_angle_deg": 20.0,      # main subgroup: max tangent angle deviation
    "trigger_main_length_scale": 6.0,            # main subgroup: minimum length = scale * spacing
    "trigger_main_lateral_scale": 2.5,           # main subgroup: max lateral spread = scale * spacing
    # -- Trigger fragment subgroup classification --
    "trigger_fragment_min_points": 6,            # fragment subgroup: minimum point count
    "trigger_fragment_linearity_th": 0.78,       # fragment subgroup: minimum PCA linearity
    "trigger_fragment_tangent_angle_deg": 28.0,  # fragment subgroup: max tangent angle deviation
    "trigger_fragment_lateral_scale": 3.5,       # fragment subgroup: max lateral spread = scale * spacing
    # -- Fragment attachment to main bundles --
    "trigger_fragment_attach_dist_scale": 2.5,   # max distance to main = scale * spacing
    "trigger_fragment_attach_gap_scale": 4.0,    # max along-axis gap = scale * spacing
    "trigger_fragment_attach_angle_deg": 20.0,   # max angle to main direction
    # -- Main bundle merging --
    "trigger_main_merge_angle_deg": 10.0,        # max angle between main bundle directions
    "trigger_main_merge_dist_scale": 1.5,        # max line-to-centroid distance = scale * spacing
    "trigger_main_merge_gap_scale": 3.0,         # max along-axis endpoint gap = scale * spacing
    "trigger_main_merge_lateral_scale": 1.4,     # max lateral offset between centroids = scale * spacing
    # -- Endpoint absorption --
    "trigger_endpoint_absorb_dist_scale": 2.2,           # max endpoint distance = scale * spacing
    "trigger_endpoint_absorb_line_dist_scale": 1.6,      # max line distance = scale * spacing
    "trigger_endpoint_absorb_proj_scale": 2.6,           # max projection distance = scale * spacing
    "trigger_endpoint_absorb_max_points_per_end": 12,    # max bad points absorbed per endpoint
}

# ---------------------------------------------------------------------------
# Section 2 -- Stage 2 trigger parameters
# ---------------------------------------------------------------------------
# These control the trigger-split decision in Stage 2 (local_clusters_core.py).
# The trigger fires when ALL three conditions are met (AND logic) plus minimum
# cluster size.
#
# trigger_min_cluster_size is computed at runtime as:
#   max(min_samples * trigger_min_cluster_size_factor, trigger_min_cluster_size_floor)
# where min_samples is a CLI parameter (default 8).

DEFAULT_TRIGGER_PARAMS: dict[str, float | int] = {
    "trigger_min_cluster_size_factor": 6,    # trigger_min_cluster_size = max(min_samples * factor, floor)
    "trigger_min_cluster_size_floor": 48,    # absolute minimum cluster size for trigger
    "linearity_th": 0.85,                    # cluster PCA linearity below this triggers
    "tangent_coherence_th": 0.88,            # cluster tangent coherence below this triggers
    "bbox_anisotropy_th": 6.0,               # cluster bbox anisotropy below this triggers (not a long strip)
}

# ---------------------------------------------------------------------------
# Section 3 -- Stage 2 denoise parameters
# ---------------------------------------------------------------------------
# These are safeguards for the lightweight_denoise_cluster() function in
# local_clusters_core.py. They prevent over-aggressive denoising.
#
# min_keep_points is computed at runtime as:
#   max(min_samples * min_keep_points_factor, min_keep_points_floor)
# where min_samples is a CLI parameter (default 8).

DEFAULT_DENOISE_PARAMS: dict[str, float | int] = {
    "max_remove_ratio": 0.20,               # cancel denoise if >20% of cluster would be removed
    "min_keep_points_factor": 1,             # min_keep_points = max(min_samples * factor, floor)
    "min_keep_points_floor": 6,              # absolute minimum points to keep after denoise
}

# ---------------------------------------------------------------------------
# Section 4 -- Stage 4 parameters (reference only)
# ---------------------------------------------------------------------------
# Stage 4 parameters (currently CLI args in build_pointwise_edge_supervision.py):
# support_radius = 0.08 (default)
# sigma = support_radius / 2.0 = 0.04
# These are managed via argparse, not extracted here.
# Phase 3 config injection (REF-04) will unify them.
