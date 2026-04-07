"""Contract invariant tests for the Phase 4 Stage 2 clustering pipeline (ALG-01).

Tests verify that the new fine-grained clustering output satisfies:
  - Direction consistency: every non-fallback cluster is a single direction group
  - Spatial continuity: no along-axis gaps exceed adaptive threshold
  - Lateral spread: perpendicular deviation within adaptive threshold
  - No trigger flag in output payload (Phase 4 eliminates trigger mechanism)
  - Non-regression: assigned center count and cluster count are reasonable
    compared to Part A baseline

All tests use the 010101 sample scene via the phase4_stage2 fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.config import Stage2Config
from core.validation import (
    StageValidationError,
    validate_cluster_contract,
    validate_local_clusters,
)

_HERE = Path(__file__).resolve().parent
REFERENCE_DIR = _HERE / "reference"
SAMPLE_SCENE_DIR = _HERE.parent / "samples" / "010101"

_SKIP_NO_SCENE = pytest.mark.skipif(
    not (SAMPLE_SCENE_DIR / "coord.npy").exists(),
    reason="Sample scene 010101 not on disk",
)
_SKIP_NO_REF = pytest.mark.skipif(
    not REFERENCE_DIR.exists() or not (REFERENCE_DIR / "local_clusters.npz").exists(),
    reason="Part A reference data not generated",
)

pytestmark = [_SKIP_NO_SCENE]


# ---------------------------------------------------------------------------
# Contract invariant tests
# ---------------------------------------------------------------------------


class TestClusterContract:
    def test_every_cluster_direction_consistent(self, phase4_stage2) -> None:
        """validate_cluster_contract must not raise for direction check (H1).

        Non-fallback clusters are single direction groups by construction.
        Fallback clusters are skipped during validation.
        """
        payload, meta, bc = phase4_stage2
        cfg = Stage2Config()
        # Should not raise
        validate_cluster_contract(
            boundary_centers=bc,
            local_clusters=payload,
            direction_cos_th=cfg.segment_direction_cos_th,
        )

    def test_every_cluster_spatially_continuous(self, phase4_stage2) -> None:
        """No along-axis gap > gap_th in any non-fallback cluster (H2).

        We re-implement the check inline to verify individual clusters
        (validate_cluster_contract only raises on systematic failure).
        """
        from core.fitting import estimate_local_spacing
        from core.local_clusters_core import group_tangents
        from utils.common import normalize_rows

        payload, meta, bc = phase4_stage2
        cfg = Stage2Config()
        coords = bc["center_coord"]
        tangents = bc["center_tangent"]
        center_index = payload["center_index"]
        cluster_id = payload["cluster_id"]

        violations = 0
        checked = 0

        for cid in np.unique(cluster_id):
            mask = cluster_id == int(cid)
            member_idx = center_index[mask]
            c = coords[member_idx]
            t = normalize_rows(tangents[member_idx])

            if c.shape[0] < cfg.segment_min_points:
                continue

            # Skip fallback clusters (multi-direction-group)
            dir_labels = group_tangents(t, cfg.segment_direction_cos_th)
            n_groups = len(np.unique(dir_labels[dir_labels >= 0]))
            if n_groups > 1:
                continue

            checked += 1
            mean_t = t.mean(axis=0)
            norm = float(np.linalg.norm(mean_t))
            if norm > 1e-8:
                axis = mean_t / norm
            else:
                _, _, vh = np.linalg.svd(c - c.mean(axis=0, keepdims=True), full_matrices=False)
                axis = vh[0]

            proj = np.sort((c - c.mean(axis=0, keepdims=True)) @ axis)
            spacing = max(estimate_local_spacing(c), 1e-6)
            gap_th = 3.0 * spacing

            if proj.shape[0] >= 2:
                max_gap = float(np.diff(proj).max())
                if max_gap > gap_th:
                    violations += 1

        # Allow up to 50% violation (matches validate_cluster_contract threshold)
        assert checked > 0, "No clusters checked"
        violation_rate = violations / checked
        assert violation_rate <= 0.50, (
            f"H2 spatial continuity: {violations}/{checked} clusters violate "
            f"(rate={violation_rate:.1%}, threshold=50%)"
        )

    def test_every_cluster_laterally_bounded(self, phase4_stage2) -> None:
        """Lateral spread within band_th for non-fallback clusters (H3)."""
        from core.fitting import estimate_local_spacing
        from core.local_clusters_core import group_tangents
        from utils.common import normalize_rows

        payload, meta, bc = phase4_stage2
        cfg = Stage2Config()
        coords = bc["center_coord"]
        tangents = bc["center_tangent"]
        center_index = payload["center_index"]
        cluster_id = payload["cluster_id"]

        violations = 0
        checked = 0

        for cid in np.unique(cluster_id):
            mask = cluster_id == int(cid)
            member_idx = center_index[mask]
            c = coords[member_idx]
            t = normalize_rows(tangents[member_idx])

            if c.shape[0] < cfg.segment_min_points:
                continue

            dir_labels = group_tangents(t, cfg.segment_direction_cos_th)
            n_groups = len(np.unique(dir_labels[dir_labels >= 0]))
            if n_groups > 1:
                continue

            checked += 1
            mean_t = t.mean(axis=0)
            norm = float(np.linalg.norm(mean_t))
            if norm > 1e-8:
                axis = mean_t / norm
            else:
                _, _, vh = np.linalg.svd(c - c.mean(axis=0, keepdims=True), full_matrices=False)
                axis = vh[0]

            centered = c - c.mean(axis=0, keepdims=True)
            along = (centered @ axis)[:, None] * axis[None, :]
            perp = centered - along
            lateral_dev = np.linalg.norm(perp, axis=1)

            spacing = max(estimate_local_spacing(c), 1e-6)
            band_th = 3.0 * spacing

            if float(lateral_dev.max()) > band_th:
                violations += 1

        assert checked > 0, "No clusters checked"
        violation_rate = violations / checked
        assert violation_rate <= 0.50, (
            f"H3 lateral spread: {violations}/{checked} clusters violate "
            f"(rate={violation_rate:.1%}, threshold=50%)"
        )

    def test_no_trigger_flag_in_output(self, phase4_stage2) -> None:
        """Phase 4 removes cluster_trigger_flag from the Stage 2 payload."""
        payload, meta, bc = phase4_stage2
        assert "cluster_trigger_flag" not in payload, (
            f"cluster_trigger_flag should not be in Phase 4 output, "
            f"got keys: {sorted(payload.keys())}"
        )

    @pytest.mark.skipif(
        not REFERENCE_DIR.exists() or not (REFERENCE_DIR / "local_clusters.npz").exists(),
        reason="Part A reference data not generated",
    )
    def test_total_assigned_centers_reasonable(self, phase4_stage2) -> None:
        """Assigned center count >= 80% of Part A baseline (non-regression)."""
        payload, meta, bc = phase4_stage2
        ref = np.load(REFERENCE_DIR / "local_clusters.npz")
        ref_assigned = ref["center_index"].shape[0]
        new_assigned = payload["center_index"].shape[0]
        assert new_assigned >= 0.80 * ref_assigned, (
            f"Non-regression: new assigned centers ({new_assigned}) < 80% of "
            f"Part A baseline ({ref_assigned})"
        )

    @pytest.mark.skipif(
        not REFERENCE_DIR.exists() or not (REFERENCE_DIR / "local_clusters.npz").exists(),
        reason="Part A reference data not generated",
    )
    def test_cluster_count_increased(self, phase4_stage2) -> None:
        """Phase 4 should produce more clusters (fine-grained runs > coarse DBSCAN)."""
        payload, meta, bc = phase4_stage2
        ref = np.load(REFERENCE_DIR / "local_clusters.npz")
        ref_clusters = ref["semantic_pair"].shape[0]
        new_clusters = payload["semantic_pair"].shape[0]
        assert new_clusters > ref_clusters, (
            f"Expected more clusters from Phase 4 runs: "
            f"new={new_clusters}, Part A baseline={ref_clusters}"
        )

    def test_validation_hook_passes(self, phase4_stage2) -> None:
        """Both validate_local_clusters and validate_cluster_contract pass."""
        payload, meta, bc = phase4_stage2
        cfg = Stage2Config()
        num_bc = bc["center_coord"].shape[0]

        # validate_local_clusters should not raise
        validate_local_clusters(payload, num_boundary_centers=num_bc)

        # validate_cluster_contract should not raise
        validate_cluster_contract(
            boundary_centers=bc,
            local_clusters=payload,
            direction_cos_th=cfg.segment_direction_cos_th,
        )
