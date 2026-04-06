"""Tests for the per-stage frozen dataclass configuration system (REF-04).

Verifies:
  - Default values match current hardcoded parameters exactly
  - Derived properties (angle -> cosine, factor/floor -> computed) are correct
  - Stage3Config.to_runtime_dict() produces the same dict as build_runtime_params()
  - Frozen semantics prevent post-construction modification
  - Custom values override defaults while keeping other fields intact
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from core.config import Stage1Config, Stage2Config, Stage3Config, Stage4Config


# ---------------------------------------------------------------------------
# Stage 1 defaults
# ---------------------------------------------------------------------------

class TestStage1Defaults:
    def test_stage1_defaults(self) -> None:
        cfg = Stage1Config()
        assert cfg.k == 32
        assert cfg.min_cross_ratio == 0.15
        assert cfg.min_side_points == 4
        assert cfg.ignore_index == -1


# ---------------------------------------------------------------------------
# Stage 2 defaults and derived
# ---------------------------------------------------------------------------

class TestStage2Defaults:
    def test_stage2_defaults(self) -> None:
        cfg = Stage2Config()
        # CLI-level
        assert cfg.eps == 0.08
        assert cfg.min_samples == 8
        assert cfg.denoise_knn == 8
        assert cfg.sparse_distance_ratio == 1.75
        assert cfg.sparse_mad_scale == 3.0
        # Trigger params (from DEFAULT_TRIGGER_PARAMS)
        assert cfg.trigger_min_cluster_size_factor == 6
        assert cfg.trigger_min_cluster_size_floor == 48
        assert cfg.linearity_th == 0.85
        assert cfg.tangent_coherence_th == 0.88
        assert cfg.bbox_anisotropy_th == 6.0
        # Denoise params (from DEFAULT_DENOISE_PARAMS)
        assert cfg.max_remove_ratio == 0.20
        assert cfg.min_keep_points_factor == 1
        assert cfg.min_keep_points_floor == 6

    def test_stage2_derived(self) -> None:
        cfg = Stage2Config()
        # trigger_min_cluster_size = max(8*6, 48) = max(48, 48) = 48
        assert cfg.trigger_min_cluster_size == 48
        # min_keep_points = max(8*1, 6) = max(8, 6) = 8
        assert cfg.min_keep_points == 8


# ---------------------------------------------------------------------------
# Stage 3 defaults, cosine properties, and to_runtime_dict
# ---------------------------------------------------------------------------

class TestStage3Defaults:
    def test_stage3_defaults(self) -> None:
        cfg = Stage3Config()
        # CLI-level
        assert cfg.line_residual_th == 0.01
        assert cfg.min_cluster_size == 8
        assert cfg.max_polyline_vertices == 32
        # All 25 DEFAULT_FIT_PARAMS fields
        assert cfg.segment_direction_angle_deg == 20.0
        assert cfg.segment_run_gap_scale == 3.0
        assert cfg.segment_run_lateral_gap_scale == 2.5
        assert cfg.segment_run_lateral_band_scale == 3.0
        assert cfg.segment_min_points == 6
        assert cfg.trigger_main_min_points == 12
        assert cfg.trigger_main_linearity_th == 0.88
        assert cfg.trigger_main_tangent_angle_deg == 20.0
        assert cfg.trigger_main_length_scale == 6.0
        assert cfg.trigger_main_lateral_scale == 2.5
        assert cfg.trigger_fragment_min_points == 6
        assert cfg.trigger_fragment_linearity_th == 0.78
        assert cfg.trigger_fragment_tangent_angle_deg == 28.0
        assert cfg.trigger_fragment_lateral_scale == 3.5
        assert cfg.trigger_fragment_attach_dist_scale == 2.5
        assert cfg.trigger_fragment_attach_gap_scale == 4.0
        assert cfg.trigger_fragment_attach_angle_deg == 20.0
        assert cfg.trigger_main_merge_angle_deg == 10.0
        assert cfg.trigger_main_merge_dist_scale == 1.5
        assert cfg.trigger_main_merge_gap_scale == 3.0
        assert cfg.trigger_main_merge_lateral_scale == 1.4
        assert cfg.trigger_endpoint_absorb_dist_scale == 2.2
        assert cfg.trigger_endpoint_absorb_line_dist_scale == 1.6
        assert cfg.trigger_endpoint_absorb_proj_scale == 2.6
        assert cfg.trigger_endpoint_absorb_max_points_per_end == 12

    def test_stage3_cosine_properties(self) -> None:
        cfg = Stage3Config()
        # segment_direction_cos_th from 20.0 deg
        assert cfg.segment_direction_cos_th == float(np.cos(np.deg2rad(20.0)))
        # trigger_main_tangent_cos_th from 20.0 deg
        assert cfg.trigger_main_tangent_cos_th == float(np.cos(np.deg2rad(20.0)))
        # trigger_fragment_tangent_cos_th from 28.0 deg
        assert cfg.trigger_fragment_tangent_cos_th == float(np.cos(np.deg2rad(28.0)))
        # trigger_fragment_attach_cos_th from 20.0 deg
        assert cfg.trigger_fragment_attach_cos_th == float(np.cos(np.deg2rad(20.0)))
        # trigger_main_merge_cos_th from 10.0 deg
        assert cfg.trigger_main_merge_cos_th == float(np.cos(np.deg2rad(10.0)))

    def test_stage3_to_runtime_dict(self) -> None:
        """Verify to_runtime_dict() produces the EXACT dict that
        build_runtime_params(default_args) produces in fit_local_supports.py.
        """
        cfg = Stage3Config()
        result = cfg.to_runtime_dict()

        # This is the verbatim expected dict from build_runtime_params()
        # with all defaults applied.
        expected = {
            "line_residual_th": 0.01,
            "min_cluster_size": 8,
            "max_polyline_vertices": 32,
            "segment_direction_cos_th": float(np.cos(np.deg2rad(20.0))),
            "segment_run_gap_scale": 3.0,
            "segment_run_lateral_gap_scale": 2.5,
            "segment_run_lateral_band_scale": 3.0,
            "segment_min_points": 6,
            "trigger_main_min_points": 12,
            "trigger_main_linearity_th": 0.88,
            "trigger_main_tangent_cos_th": float(np.cos(np.deg2rad(20.0))),
            "trigger_main_length_scale": 6.0,
            "trigger_main_lateral_scale": 2.5,
            "trigger_fragment_min_points": 6,
            "trigger_fragment_linearity_th": 0.78,
            "trigger_fragment_tangent_cos_th": float(np.cos(np.deg2rad(28.0))),
            "trigger_fragment_lateral_scale": 3.5,
            "trigger_fragment_attach_dist_scale": 2.5,
            "trigger_fragment_attach_gap_scale": 4.0,
            "trigger_fragment_attach_cos_th": float(np.cos(np.deg2rad(20.0))),
            "trigger_main_merge_cos_th": float(np.cos(np.deg2rad(10.0))),
            "trigger_main_merge_dist_scale": 1.5,
            "trigger_main_merge_gap_scale": 3.0,
            "trigger_main_merge_lateral_scale": 1.4,
            "trigger_endpoint_absorb_dist_scale": 2.2,
            "trigger_endpoint_absorb_line_dist_scale": 1.6,
            "trigger_endpoint_absorb_proj_scale": 2.6,
            "trigger_endpoint_absorb_max_points_per_end": 12,
        }

        assert set(result.keys()) == set(expected.keys()), (
            f"Key mismatch:\n"
            f"  extra: {set(result.keys()) - set(expected.keys())}\n"
            f"  missing: {set(expected.keys()) - set(result.keys())}"
        )
        for key in expected:
            assert result[key] == expected[key], (
                f"Value mismatch for {key!r}: {result[key]!r} != {expected[key]!r}"
            )
            assert type(result[key]) is type(expected[key]), (
                f"Type mismatch for {key!r}: {type(result[key]).__name__} != {type(expected[key]).__name__}"
            )


# ---------------------------------------------------------------------------
# Stage 4 defaults and sigma
# ---------------------------------------------------------------------------

class TestStage4Defaults:
    def test_stage4_defaults(self) -> None:
        cfg = Stage4Config()
        assert cfg.support_radius == 0.08
        assert cfg.ignore_index == -1

    def test_stage4_sigma(self) -> None:
        cfg = Stage4Config()
        assert cfg.sigma == 0.04


# ---------------------------------------------------------------------------
# Frozen semantics
# ---------------------------------------------------------------------------

class TestFrozen:
    def test_frozen(self) -> None:
        """Assigning to any config field must raise FrozenInstanceError."""
        for cfg in (Stage1Config(), Stage2Config(), Stage3Config(), Stage4Config()):
            with pytest.raises(dataclasses.FrozenInstanceError):
                object.__setattr__  # just to confirm the exception type is correct
                cfg.__class__.__dataclass_fields__  # access for iteration
                # Pick the first field name and try to set it
                first_field = next(iter(dataclasses.fields(cfg))).name
                setattr(cfg, first_field, 999)


# ---------------------------------------------------------------------------
# Custom values
# ---------------------------------------------------------------------------

class TestCustomValues:
    def test_custom_values(self) -> None:
        cfg = Stage2Config(eps=0.12)
        assert cfg.eps == 0.12
        # All other fields remain at defaults
        assert cfg.min_samples == 8
        assert cfg.denoise_knn == 8
        assert cfg.sparse_distance_ratio == 1.75
        assert cfg.sparse_mad_scale == 3.0
        assert cfg.trigger_min_cluster_size_factor == 6
        assert cfg.trigger_min_cluster_size_floor == 48
        assert cfg.linearity_th == 0.85
        assert cfg.tangent_coherence_th == 0.88
        assert cfg.bbox_anisotropy_th == 6.0
        assert cfg.max_remove_ratio == 0.20
        assert cfg.min_keep_points_factor == 1
        assert cfg.min_keep_points_floor == 6
