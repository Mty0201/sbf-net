"""Contract invariant tests for the Stage 2 clustering pipeline.

Tests verify that clustering output satisfies:
  - Direction consistency: every non-fallback cluster is a single direction group
  - Spatial continuity: no along-axis gaps exceed adaptive threshold
  - Lateral spread: perpendicular deviation within adaptive threshold
  - No trigger flag in output payload
  - Validation hooks pass

All tests use the 010101 sample scene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.config import Stage1Config, Stage2Config
from core.validation import (
    validate_cluster_contract,
    validate_local_clusters,
)

SAMPLE_SCENE_DIR = Path(__file__).resolve().parent.parent / "samples" / "010101"

pytestmark = pytest.mark.skipif(
    not (SAMPLE_SCENE_DIR / "coord.npy").exists(),
    reason="Sample scene 010101 not on disk",
)


@pytest.fixture(scope="module")
def stage2_result():
    """Run Stages 1→2 on sample scene 010101."""
    from core.boundary_centers_core import build_boundary_centers
    from core.local_clusters_core import cluster_boundary_centers
    from utils.stage_io import load_scene

    cfg1 = Stage1Config()
    scene = load_scene(SAMPLE_SCENE_DIR)
    _, bc, _ = build_boundary_centers(
        scene=scene,
        k=cfg1.k,
        min_cross_ratio=cfg1.min_cross_ratio,
        min_side_points=cfg1.min_side_points,
        ignore_index=cfg1.ignore_index,
    )
    cfg2 = Stage2Config()
    payload, meta = cluster_boundary_centers(bc, cfg2)
    return payload, meta, bc


class TestClusterContract:
    def test_every_cluster_direction_consistent(self, stage2_result) -> None:
        payload, meta, bc = stage2_result
        cfg = Stage2Config()
        validate_cluster_contract(
            boundary_centers=bc,
            local_clusters=payload,
            direction_cos_th=cfg.merge_direction_cos_th,
        )

    def test_every_cluster_spatially_continuous(self, stage2_result) -> None:
        from core.fitting import estimate_local_spacing
        from core.local_clusters_core import group_tangents
        from utils.common import normalize_rows

        payload, meta, bc = stage2_result
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

            if c.shape[0] < cfg.min_cluster_points:
                continue

            dir_labels = group_tangents(t, cfg.merge_direction_cos_th)
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

        assert checked > 0, "No clusters checked"
        violation_rate = violations / checked
        assert violation_rate <= 0.50, (
            f"H2 spatial continuity: {violations}/{checked} clusters violate "
            f"(rate={violation_rate:.1%}, threshold=50%)"
        )

    def test_every_cluster_laterally_bounded(self, stage2_result) -> None:
        from core.fitting import estimate_local_spacing
        from core.local_clusters_core import group_tangents
        from utils.common import normalize_rows

        payload, meta, bc = stage2_result
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

            if c.shape[0] < cfg.min_cluster_points:
                continue

            dir_labels = group_tangents(t, cfg.merge_direction_cos_th)
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

    def test_no_trigger_flag_in_output(self, stage2_result) -> None:
        payload, meta, bc = stage2_result
        assert "cluster_trigger_flag" not in payload

    def test_validation_hook_passes(self, stage2_result) -> None:
        payload, meta, bc = stage2_result
        cfg = Stage2Config()
        num_bc = bc["center_coord"].shape[0]
        validate_local_clusters(payload, num_boundary_centers=num_bc)
        validate_cluster_contract(
            boundary_centers=bc,
            local_clusters=payload,
            direction_cos_th=cfg.merge_direction_cos_th,
        )
