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
    min_samples: int = 5
    denoise_knn: int = 8
    sparse_distance_ratio: float = 1.75
    sparse_mad_scale: float = 3.0

    # Denoise params (from DEFAULT_DENOISE_PARAMS)
    max_remove_ratio: float = 0.20
    min_keep_points_factor: int = 1
    min_keep_points_floor: int = 6

    # Direction + spatial run splitting (moved from Stage3Config)
    segment_direction_angle_deg: float = 45.0
    segment_run_gap_scale: float = 3.0
    segment_run_lateral_gap_scale: float = 2.5
    segment_run_lateral_band_scale: float = 3.0
    segment_min_points: int = 4

    # Density-adaptive rescue
    rescue_knn: int = 8
    rescue_distance_scale: float = 3.0

    # Density-conditional denoise: skip denoise for sparse clusters
    # Clusters with spacing > threshold * global_median skip denoise
    # 0.5 restricts denoise to only tightly-packed clusters (<=0.5x global median)
    denoise_density_threshold: float = 0.5

    @property
    def segment_direction_cos_th(self) -> float:
        """Cosine threshold for direction grouping (sign-invariant)."""
        return float(np.cos(np.deg2rad(float(self.segment_direction_angle_deg))))

    @property
    def min_keep_points(self) -> int:
        """Minimum points to keep after denoising.

        ``max(min_samples * min_keep_points_factor,
              min_keep_points_floor)``

        With defaults: ``max(5 * 1, 6) = 6``.
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
    """Support fitting parameters (3 CLI + 4 endpoint absorption)."""

    # CLI-level
    line_residual_th: float = 0.01
    min_cluster_size: int = 4
    max_polyline_vertices: int = 32

    # -- Endpoint absorption --
    trigger_endpoint_absorb_dist_scale: float = 2.2
    trigger_endpoint_absorb_line_dist_scale: float = 1.6
    trigger_endpoint_absorb_proj_scale: float = 2.6
    trigger_endpoint_absorb_max_points_per_end: int = 12

    def to_runtime_dict(self) -> dict:
        """Produce the flat dict that ``build_supports_payload()`` expects.

        Contains exactly 7 keys: 3 CLI params + 4 endpoint absorption params.
        Transition method: lets core functions continue taking a flat dict
        while the config system is established.
        """
        return {
            "line_residual_th": float(self.line_residual_th),
            "min_cluster_size": int(self.min_cluster_size),
            "max_polyline_vertices": int(self.max_polyline_vertices),
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
