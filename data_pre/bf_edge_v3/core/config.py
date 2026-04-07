"""
Per-stage frozen dataclass configuration for the bf_edge_v3 pipeline.

Unifies the 5 scattered parameter sources (params.py dicts, CLI argparse
defaults, build_runtime_params() angle-to-cosine conversion, inline computed
thresholds) into one frozen dataclass per stage.

Design principles:
  - frozen=True: immutable after construction (T-03-01 mitigation)
  - Defaults reproduce current pipeline behavior exactly (Part A rule)
  - Angle params stored in degrees; cosine thresholds exposed as @property
  - to_runtime_dict() on Stage3Config produces the flat dict that
    build_supports_payload() expects (transition method)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# -------------------------------------------------------------------------
# Stage 1: Boundary center detection
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage1Config:
    """Boundary center detection parameters."""

    k: int = 32
    min_cross_ratio: float = 0.15
    min_side_points: int = 4
    ignore_index: int = -1


# -------------------------------------------------------------------------
# Stage 2: DBSCAN clustering + denoise + trigger
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage2Config:
    """DBSCAN clustering, denoising, direction/spatial splitting, and rescue parameters."""

    # CLI-level
    eps: float = 0.08
    min_samples: int = 8
    denoise_knn: int = 8
    sparse_distance_ratio: float = 1.75
    sparse_mad_scale: float = 3.0

    # Denoise params (from DEFAULT_DENOISE_PARAMS)
    max_remove_ratio: float = 0.20
    min_keep_points_factor: int = 1
    min_keep_points_floor: int = 6

    # Direction + spatial run splitting (moved from Stage3Config)
    segment_direction_angle_deg: float = 20.0
    segment_run_gap_scale: float = 3.0
    segment_run_lateral_gap_scale: float = 2.5
    segment_run_lateral_band_scale: float = 3.0
    segment_min_points: int = 6

    # Density-adaptive rescue
    rescue_knn: int = 8
    rescue_distance_scale: float = 2.0

    @property
    def segment_direction_cos_th(self) -> float:
        """Cosine threshold for direction grouping (sign-invariant)."""
        return float(np.cos(np.deg2rad(float(self.segment_direction_angle_deg))))

    @property
    def min_keep_points(self) -> int:
        """Minimum points to keep after denoising.

        ``max(min_samples * min_keep_points_factor,
              min_keep_points_floor)``
        """
        return max(
            self.min_samples * self.min_keep_points_factor,
            self.min_keep_points_floor,
        )


# -------------------------------------------------------------------------
# Stage 3: Support fitting
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage3Config:
    """Support fitting parameters (3 CLI + 25 DEFAULT_FIT_PARAMS)."""

    # CLI-level
    line_residual_th: float = 0.01
    min_cluster_size: int = 8
    max_polyline_vertices: int = 32

    # -- Run splitting parameters --
    segment_direction_angle_deg: float = 20.0
    segment_run_gap_scale: float = 3.0
    segment_run_lateral_gap_scale: float = 2.5
    segment_run_lateral_band_scale: float = 3.0
    segment_min_points: int = 6

    # -- Trigger main subgroup classification --
    trigger_main_min_points: int = 12
    trigger_main_linearity_th: float = 0.88
    trigger_main_tangent_angle_deg: float = 20.0
    trigger_main_length_scale: float = 6.0
    trigger_main_lateral_scale: float = 2.5

    # -- Trigger fragment subgroup classification --
    trigger_fragment_min_points: int = 6
    trigger_fragment_linearity_th: float = 0.78
    trigger_fragment_tangent_angle_deg: float = 28.0
    trigger_fragment_lateral_scale: float = 3.5

    # -- Fragment attachment to main bundles --
    trigger_fragment_attach_dist_scale: float = 2.5
    trigger_fragment_attach_gap_scale: float = 4.0
    trigger_fragment_attach_angle_deg: float = 20.0

    # -- Main bundle merging --
    trigger_main_merge_angle_deg: float = 10.0
    trigger_main_merge_dist_scale: float = 1.5
    trigger_main_merge_gap_scale: float = 3.0
    trigger_main_merge_lateral_scale: float = 1.4

    # -- Endpoint absorption --
    trigger_endpoint_absorb_dist_scale: float = 2.2
    trigger_endpoint_absorb_line_dist_scale: float = 1.6
    trigger_endpoint_absorb_proj_scale: float = 2.6
    trigger_endpoint_absorb_max_points_per_end: int = 12

    # -- Cosine-threshold derived properties --

    @property
    def segment_direction_cos_th(self) -> float:
        return float(np.cos(np.deg2rad(float(self.segment_direction_angle_deg))))

    @property
    def trigger_main_tangent_cos_th(self) -> float:
        return float(np.cos(np.deg2rad(float(self.trigger_main_tangent_angle_deg))))

    @property
    def trigger_fragment_tangent_cos_th(self) -> float:
        return float(np.cos(np.deg2rad(float(self.trigger_fragment_tangent_angle_deg))))

    @property
    def trigger_fragment_attach_cos_th(self) -> float:
        return float(np.cos(np.deg2rad(float(self.trigger_fragment_attach_angle_deg))))

    @property
    def trigger_main_merge_cos_th(self) -> float:
        return float(np.cos(np.deg2rad(float(self.trigger_main_merge_angle_deg))))

    def to_runtime_dict(self) -> dict:
        """Produce the flat dict that ``build_supports_payload()`` expects.

        This is the EXACT dict previously built by ``build_runtime_params()``
        in ``fit_local_supports.py`` (and the identical
        ``build_support_runtime_params()`` in ``build_support_dataset_v3.py``).

        Transition method: lets core functions continue taking a flat dict
        while the config system is established.
        """
        return {
            "line_residual_th": float(self.line_residual_th),
            "min_cluster_size": int(self.min_cluster_size),
            "max_polyline_vertices": int(self.max_polyline_vertices),
            "segment_direction_cos_th": self.segment_direction_cos_th,
            "segment_run_gap_scale": float(self.segment_run_gap_scale),
            "segment_run_lateral_gap_scale": float(self.segment_run_lateral_gap_scale),
            "segment_run_lateral_band_scale": float(self.segment_run_lateral_band_scale),
            "segment_min_points": int(self.segment_min_points),
            "trigger_main_min_points": int(self.trigger_main_min_points),
            "trigger_main_linearity_th": float(self.trigger_main_linearity_th),
            "trigger_main_tangent_cos_th": self.trigger_main_tangent_cos_th,
            "trigger_main_length_scale": float(self.trigger_main_length_scale),
            "trigger_main_lateral_scale": float(self.trigger_main_lateral_scale),
            "trigger_fragment_min_points": int(self.trigger_fragment_min_points),
            "trigger_fragment_linearity_th": float(self.trigger_fragment_linearity_th),
            "trigger_fragment_tangent_cos_th": self.trigger_fragment_tangent_cos_th,
            "trigger_fragment_lateral_scale": float(self.trigger_fragment_lateral_scale),
            "trigger_fragment_attach_dist_scale": float(self.trigger_fragment_attach_dist_scale),
            "trigger_fragment_attach_gap_scale": float(self.trigger_fragment_attach_gap_scale),
            "trigger_fragment_attach_cos_th": self.trigger_fragment_attach_cos_th,
            "trigger_main_merge_cos_th": self.trigger_main_merge_cos_th,
            "trigger_main_merge_dist_scale": float(self.trigger_main_merge_dist_scale),
            "trigger_main_merge_gap_scale": float(self.trigger_main_merge_gap_scale),
            "trigger_main_merge_lateral_scale": float(self.trigger_main_merge_lateral_scale),
            "trigger_endpoint_absorb_dist_scale": float(self.trigger_endpoint_absorb_dist_scale),
            "trigger_endpoint_absorb_line_dist_scale": float(self.trigger_endpoint_absorb_line_dist_scale),
            "trigger_endpoint_absorb_proj_scale": float(self.trigger_endpoint_absorb_proj_scale),
            "trigger_endpoint_absorb_max_points_per_end": int(self.trigger_endpoint_absorb_max_points_per_end),
        }


# -------------------------------------------------------------------------
# Stage 4: Pointwise edge supervision
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage4Config:
    """Pointwise edge supervision parameters."""

    support_radius: float = 0.08
    ignore_index: int = -1

    @property
    def sigma(self) -> float:
        """Gaussian decay sigma. Currently ``support_radius / 2``."""
        return max(self.support_radius / 2.0, 1e-8)
