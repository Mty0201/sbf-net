"""
Equivalence gate: end-to-end pipeline validation and in-memory path consistency.

Reference-data equivalence tests were removed during cleanup — the pipeline
has evolved through multiple phases and frozen snapshots are no longer
meaningful. The surviving tests verify:
  1. All validation hooks pass on current pipeline output.
  2. The in-memory path (Stages 1→2→3 in one process) is bit-identical
     to the per-stage path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
SAMPLE_SCENE_DIR = _HERE.parent / "samples" / "010101"

_SKIP_NO_SCENE = pytest.mark.skipif(
    not (SAMPLE_SCENE_DIR / "coord.npy").exists(),
    reason="Sample scene 010101 not on disk",
)

pytestmark = [_SKIP_NO_SCENE]


# ---------------------------------------------------------------------------
# Module-scoped fixtures: run pipeline once, reuse across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stage1(sample_scene_dir):
    """Run Stage 1 with default Stage1Config on scene 010101."""
    from core.config import Stage1Config
    from core.boundary_centers_core import build_boundary_centers
    from utils.stage_io import load_scene

    cfg = Stage1Config()
    scene = load_scene(sample_scene_dir)
    _, payload, _ = build_boundary_centers(
        scene=scene,
        k=cfg.k,
        min_cross_ratio=cfg.min_cross_ratio,
        min_side_points=cfg.min_side_points,
        ignore_index=cfg.ignore_index,
    )
    return payload


@pytest.fixture(scope="module")
def stage2(stage1):
    """Run Stage 2 with default config."""
    from core.config import Stage2Config
    from core.local_clusters_core import cluster_boundary_centers

    cfg = Stage2Config()
    payload, _ = cluster_boundary_centers(
        boundary_centers=stage1,
        config=cfg,
    )
    return payload


@pytest.fixture(scope="module")
def stage3(stage1, stage2):
    """Run Stage 3 with default config."""
    from core.config import Stage3Config
    from core.supports_core import build_supports_payload

    cfg = Stage3Config()
    payload, _, _ = build_supports_payload(
        boundary_centers=stage1,
        local_clusters=stage2,
        params=cfg.to_runtime_dict(),
    )
    return payload


@pytest.fixture(scope="module")
def stage4(sample_scene_dir, stage3):
    """Run Stage 4 with default config."""
    from core.config import Stage4Config
    from core.pointwise_core import build_pointwise_edge_supervision
    from utils.stage_io import load_scene

    cfg = Stage4Config()
    scene = load_scene(sample_scene_dir)
    payload, _ = build_pointwise_edge_supervision(
        scene=scene,
        supports=stage3,
        support_radius=cfg.support_radius,
        ignore_index=cfg.ignore_index,
    )
    return payload


# ---------------------------------------------------------------------------
# Validation hooks pass on current pipeline output
# ---------------------------------------------------------------------------


def test_validation_passes_on_pipeline_output(
    stage1, stage2, stage3, stage4, sample_scene_dir,
):
    from core.config import Stage2Config
    from core.validation import (
        validate_boundary_centers,
        validate_cluster_contract,
        validate_local_clusters,
        validate_supports,
        validate_edge_supervision,
    )
    from utils.stage_io import load_scene

    validate_boundary_centers(stage1)
    validate_local_clusters(
        stage2,
        num_boundary_centers=stage1["center_coord"].shape[0],
    )
    cfg2 = Stage2Config()
    validate_cluster_contract(
        boundary_centers=stage1,
        local_clusters=stage2,
        direction_cos_th=cfg2.merge_direction_cos_th,
    )
    validate_supports(stage3)
    scene = load_scene(sample_scene_dir)
    validate_edge_supervision(stage4, num_scene_points=scene["coord"].shape[0])


# ---------------------------------------------------------------------------
# In-memory path: Stages 1→2→3 in sequence matches per-stage results
# ---------------------------------------------------------------------------


def test_inmemory_path_matches_perstage(sample_scene_dir, stage3):
    """Run Stages 1→2→3 in-memory and compare Stage 3 output against
    the per-stage fixture result."""
    from core.config import Stage1Config, Stage2Config, Stage3Config
    from core.boundary_centers_core import build_boundary_centers
    from core.local_clusters_core import cluster_boundary_centers
    from core.supports_core import build_supports_payload
    from utils.stage_io import load_scene

    cfg1 = Stage1Config()
    cfg2 = Stage2Config()
    cfg3 = Stage3Config()

    scene = load_scene(sample_scene_dir)
    _, bc_payload, _ = build_boundary_centers(
        scene=scene,
        k=cfg1.k,
        min_cross_ratio=cfg1.min_cross_ratio,
        min_side_points=cfg1.min_side_points,
        ignore_index=cfg1.ignore_index,
    )
    lc_payload, _ = cluster_boundary_centers(
        boundary_centers=bc_payload,
        config=cfg2,
    )
    sup_payload, _, _ = build_supports_payload(
        boundary_centers=bc_payload,
        local_clusters=lc_payload,
        params=cfg3.to_runtime_dict(),
    )

    ref_keys = sorted(stage3.keys())
    mem_keys = sorted(sup_payload.keys())
    assert ref_keys == mem_keys, f"Key mismatch: per-stage={ref_keys}, in-memory={mem_keys}"
    for key in ref_keys:
        assert np.array_equal(stage3[key], sup_payload[key]), (
            f"In-memory path differs from per-stage at supports[{key}]"
        )
