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
# Stage 2: Bottom-up micro-cluster merge
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage2Config:
    """Bottom-up micro-cluster merge parameters.

    Algorithm: micro-cluster (small eps DBSCAN) -> bimodal lateral split
    -> direction-aware merge (union-find on adjacent clusters with compatible
    tangents) -> post-merge rescue (assign noise to nearest merged cluster).

    Replaces the previous split-based pipeline (large DBSCAN -> direction
    grouping -> spatial splitting -> outlier pruning) which suffered from
    cross-edge contamination when splitting large connected clusters.
    """

    # -- Micro-clustering --
    # eps = micro_eps_scale * global_median_spacing
    micro_eps_scale: float = 3.5
    micro_min_samples: int = 3

    # -- Bimodal lateral split --
    # Splits micro-clusters whose per-direction-group lateral distribution
    # has a gap > split_lateral_threshold_scale * gms, indicating parallel
    # edges bridged by DBSCAN through connecting edges (e.g. window frames).
    split_lateral_threshold_scale: float = 5.0

    # -- Direction-aware merge --
    # merge_radius = merge_radius_scale * global_median_spacing
    merge_radius_scale: float = 8.0
    merge_direction_angle_deg: float = 45.0
    # lateral_max = merge_lateral_scale * global_median_spacing
    # Prevents merging parallel edges that are laterally offset.
    merge_lateral_scale: float = 5.0

    # -- Post-merge rescue --
    # rescue_radius = rescue_radius_scale * global_median_spacing
    rescue_radius_scale: float = 10.0

    # -- Output filtering --
    min_cluster_points: int = 4

    @property
    def merge_direction_cos_th(self) -> float:
        """Cosine threshold for direction-compatible merge (sign-invariant)."""
        return float(np.cos(np.deg2rad(float(self.merge_direction_angle_deg))))


# -------------------------------------------------------------------------
# Stage 3: Support fitting
# -------------------------------------------------------------------------

@dataclass(frozen=True)
class Stage3Config:
    """Support fitting parameters (3 CLI + 4 endpoint absorption + 2 quality gates)."""

    # CLI-level
    line_residual_th: float = 0.01
    min_cluster_size: int = 4
    max_polyline_vertices: int = 32

    # -- Quality gates --
    polyline_residual_th: float = 0.04
    min_cluster_density: float = 15.0

    # -- Endpoint absorption --
    trigger_endpoint_absorb_dist_scale: float = 2.2
    trigger_endpoint_absorb_line_dist_scale: float = 1.6
    trigger_endpoint_absorb_proj_scale: float = 2.6
    trigger_endpoint_absorb_max_points_per_end: int = 12

    def to_runtime_dict(self) -> dict:
        """Produce the flat dict that ``build_supports_payload()`` expects.

        Contains exactly 9 keys: 3 CLI params + 2 quality gates + 4 endpoint absorption params.
        Transition method: lets core functions continue taking a flat dict
        while the config system is established.
        """
        return {
            "line_residual_th": float(self.line_residual_th),
            "min_cluster_size": int(self.min_cluster_size),
            "max_polyline_vertices": int(self.max_polyline_vertices),
            "polyline_residual_th": float(self.polyline_residual_th),
            "min_cluster_density": float(self.min_cluster_density),
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
