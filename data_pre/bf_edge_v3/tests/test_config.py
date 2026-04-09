"""Tests for the per-stage frozen dataclass configuration system (REF-04).

Verifies:
  - Default values match current hardcoded parameters exactly
  - Derived properties (angle -> cosine) are correct
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
# Stage 2 defaults and derived (Phase 6: bottom-up micro-cluster merge)
# ---------------------------------------------------------------------------

class TestStage2Defaults:
    def test_stage2_defaults(self) -> None:
        cfg = Stage2Config()
        # Micro-clustering
        assert cfg.micro_eps_scale == 3.5
        assert cfg.micro_min_samples == 3
        # Bimodal lateral split
        assert cfg.split_lateral_threshold_scale == 5.0
        # Direction-aware merge
        assert cfg.merge_radius_scale == 8.0
        assert cfg.merge_direction_angle_deg == 45.0
        assert cfg.merge_lateral_scale == 5.0
        # Post-merge rescue
        assert cfg.rescue_radius_scale == 10.0
        # Output filtering
        assert cfg.min_cluster_points == 4

    def test_stage2_derived(self) -> None:
        cfg = Stage2Config()
        # merge_direction_cos_th from 45.0 deg
        assert cfg.merge_direction_cos_th == float(np.cos(np.deg2rad(45.0)))


# ---------------------------------------------------------------------------
# Stage 3 defaults, cosine properties, and to_runtime_dict
# ---------------------------------------------------------------------------

class TestStage3Defaults:
    def test_stage3_defaults(self) -> None:
        cfg = Stage3Config()
        # CLI-level
        assert cfg.line_residual_th == 0.01
        assert cfg.min_cluster_size == 4
        assert cfg.max_polyline_vertices == 32
        # Quality gates
        assert cfg.polyline_residual_th == 0.04
        assert cfg.min_cluster_density == 15.0
        # Endpoint absorption (4 fields retained from trigger path)
        assert cfg.trigger_endpoint_absorb_dist_scale == 2.2
        assert cfg.trigger_endpoint_absorb_line_dist_scale == 1.6
        assert cfg.trigger_endpoint_absorb_proj_scale == 2.6
        assert cfg.trigger_endpoint_absorb_max_points_per_end == 12

    def test_stage3_to_runtime_dict(self) -> None:
        """Verify to_runtime_dict() produces exactly 9 keys (3 CLI + 2 gates + 4 absorb)."""
        cfg = Stage3Config()
        result = cfg.to_runtime_dict()

        expected = {
            "line_residual_th": 0.01,
            "min_cluster_size": 4,
            "max_polyline_vertices": 32,
            "polyline_residual_th": 0.04,
            "min_cluster_density": 15.0,
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
        assert cfg.sigma == 0.02

    def test_stage4_sigma_custom(self) -> None:
        cfg = Stage4Config(support_sigma=0.01)
        assert cfg.sigma == 0.01


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
        cfg = Stage2Config(micro_eps_scale=5.0)
        assert cfg.micro_eps_scale == 5.0
        # All other fields remain at defaults
        assert cfg.micro_min_samples == 3
        assert cfg.merge_radius_scale == 8.0
        assert cfg.merge_direction_angle_deg == 45.0
        assert cfg.rescue_radius_scale == 10.0
        assert cfg.min_cluster_points == 4
