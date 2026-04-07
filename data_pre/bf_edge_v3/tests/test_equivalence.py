"""
Equivalence gate: bit-identical comparison of pipeline output against reference data.

Phase 5 strategy:
  - Stage 1: still compared against Part A reference (tests/reference/) since
    Stage 1 is unchanged by Phase 4 or Phase 5. Uses np.array_equal (strict bit-identical).
  - Stages 2-4 (Phase 4): @pytest.mark.skip-ed. Phase 5 intentionally changes
    Stage 2 behavior (density-conditional denoise, lowered thresholds).
  - Stages 2-4 (Phase 5): compared against Phase 5 reference data (tests/reference_v3/).
  - Cross-checks: validation hooks and in-memory path use Phase 5 pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
REFERENCE_DIR = _HERE / "reference"
REFERENCE_V2_DIR = _HERE / "reference_v2"
REFERENCE_V3_DIR = _HERE / "reference_v3"
SAMPLE_SCENE_DIR = _HERE.parent / "samples" / "010101"

_SKIP_NO_SCENE = pytest.mark.skipif(
    not (SAMPLE_SCENE_DIR / "coord.npy").exists(),
    reason="Sample scene 010101 not on disk",
)
_SKIP_NO_REF = pytest.mark.skipif(
    not REFERENCE_DIR.exists() or not (REFERENCE_DIR / "supports.npz").exists(),
    reason="Reference data not generated -- run Plan 03-01 Task 1 first",
)
_SKIP_NO_REF_V2 = pytest.mark.skipif(
    not REFERENCE_V2_DIR.exists() or not (REFERENCE_V2_DIR / "supports.npz").exists(),
    reason="Phase 4 reference data not generated",
)
_SKIP_NO_REF_V3 = pytest.mark.skipif(
    not REFERENCE_V3_DIR.exists() or not (REFERENCE_V3_DIR / "supports.npz").exists(),
    reason="Phase 5 reference data not generated -- run Plan 05-02 Task 1 first",
)

pytestmark = [_SKIP_NO_SCENE]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_npz_identical(ref_path: Path, actual: dict, label: str) -> None:
    """Assert every field in a reference NPZ matches the actual dict."""
    ref = np.load(ref_path)
    ref_keys = sorted(ref.files)
    actual_keys = sorted(actual.keys())
    assert ref_keys == actual_keys, (
        f"{label}: key mismatch -- ref={ref_keys}, actual={actual_keys}"
    )
    for key in ref_keys:
        r, a = ref[key], actual[key]
        if not np.array_equal(r, a):
            diff_idx = np.where(r != a)
            sample = tuple(idx[:5] for idx in diff_idx) if diff_idx[0].size else ()
            pytest.fail(
                f"{label}[{key}]: not bit-identical\n"
                f"  ref shape={r.shape} dtype={r.dtype}\n"
                f"  actual shape={a.shape} dtype={a.dtype}\n"
                f"  first diffs at: {sample}\n"
                f"  max |diff|={np.max(np.abs(r.astype(float) - a.astype(float)))}"
            )


def _assert_npy_identical(ref_path: Path, actual: np.ndarray, label: str) -> None:
    """Assert a reference .npy array matches the actual array."""
    ref = np.load(ref_path)
    if not np.array_equal(ref, actual):
        diff_idx = np.where(ref != actual)
        sample = tuple(idx[:5] for idx in diff_idx) if diff_idx[0].size else ()
        pytest.fail(
            f"{label}: not bit-identical\n"
            f"  ref shape={ref.shape} dtype={ref.dtype}\n"
            f"  actual shape={actual.shape} dtype={actual.dtype}\n"
            f"  first diffs at: {sample}\n"
            f"  max |diff|={np.max(np.abs(ref.astype(float) - actual.astype(float)))}"
        )


# ---------------------------------------------------------------------------
# Module-scoped fixtures: run pipeline once, reuse across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def refactored_stage1(sample_scene_dir):
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


# -- Phase 5 fixtures (current defaults) --


@pytest.fixture(scope="module")
def phase5_stage2(refactored_stage1):
    """Run Stage 2 with Phase 5 config (density-conditional denoise)."""
    from core.config import Stage2Config
    from core.local_clusters_core import cluster_boundary_centers

    cfg = Stage2Config()
    payload, _ = cluster_boundary_centers(
        boundary_centers=refactored_stage1,
        config=cfg,
    )
    return payload


@pytest.fixture(scope="module")
def phase5_stage3(refactored_stage1, phase5_stage2):
    """Run Stage 3 with Phase 5 config."""
    from core.config import Stage3Config
    from core.supports_core import build_supports_payload

    cfg = Stage3Config()
    payload, _, _ = build_supports_payload(
        boundary_centers=refactored_stage1,
        local_clusters=phase5_stage2,
        params=cfg.to_runtime_dict(),
    )
    return payload


@pytest.fixture(scope="module")
def phase5_stage4(sample_scene_dir, phase5_stage3):
    """Run Stage 4 with Phase 5 config."""
    from core.config import Stage4Config
    from core.pointwise_core import build_pointwise_edge_supervision
    from utils.stage_io import load_scene

    cfg = Stage4Config()
    scene = load_scene(sample_scene_dir)
    payload, _ = build_pointwise_edge_supervision(
        scene=scene,
        supports=phase5_stage3,
        support_radius=cfg.support_radius,
        ignore_index=cfg.ignore_index,
    )
    return payload


# ---------------------------------------------------------------------------
# Stage 1: boundary centers (unchanged by Phase 4 or Phase 5)
# ---------------------------------------------------------------------------


@_SKIP_NO_REF
def test_stage1_boundary_centers_identical(refactored_stage1, reference_dir):
    _assert_npz_identical(
        reference_dir / "boundary_centers.npz",
        refactored_stage1,
        "Stage 1 boundary_centers",
    )


# ---------------------------------------------------------------------------
# Stage 2: local clusters -- Part A baseline (SKIPPED: Phase 4 changes output)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Phase 4 intentionally changes Stages 2-4 output. "
    "Part A reference preserved in tests/reference/ for comparison."
)
def test_stage2_local_clusters_identical():
    """[ARCHIVED] Part A equivalence test for Stage 2."""
    pass


# ---------------------------------------------------------------------------
# Stage 3: supports -- Part A baseline (SKIPPED)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Phase 4 intentionally changes Stages 2-4 output. "
    "Part A reference preserved in tests/reference/ for comparison."
)
def test_stage3_supports_identical():
    """[ARCHIVED] Part A equivalence test for Stage 3."""
    pass


# ---------------------------------------------------------------------------
# Stage 4: pointwise edge supervision -- Part A baseline (SKIPPED)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Phase 4 intentionally changes Stages 2-4 output. "
    "Part A reference preserved in tests/reference/ for comparison."
)
def test_stage4_edge_dist_identical():
    """[ARCHIVED] Part A equivalence test for Stage 4 edge_dist."""
    pass


@pytest.mark.skip(
    reason="Phase 4 intentionally changes Stages 2-4 output. "
    "Part A reference preserved in tests/reference/ for comparison."
)
def test_stage4_edge_dir_identical():
    """[ARCHIVED] Part A equivalence test for Stage 4 edge_dir."""
    pass


@pytest.mark.skip(
    reason="Phase 4 intentionally changes Stages 2-4 output. "
    "Part A reference preserved in tests/reference/ for comparison."
)
def test_stage4_edge_valid_identical():
    """[ARCHIVED] Part A equivalence test for Stage 4 edge_valid."""
    pass


@pytest.mark.skip(
    reason="Phase 4 intentionally changes Stages 2-4 output. "
    "Part A reference preserved in tests/reference/ for comparison."
)
def test_stage4_edge_support_id_identical():
    """[ARCHIVED] Part A equivalence test for Stage 4 edge_support_id."""
    pass


# ---------------------------------------------------------------------------
# Phase 4 equivalence: Stages 2-4 against reference_v2/ (SKIPPED: Phase 5)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Phase 5 intentionally changes Stages 2-4 output. "
    "Phase 4 reference preserved in tests/reference_v2/ for comparison."
)
def test_stage2_local_clusters_v2_identical():
    """[ARCHIVED] Phase 4 equivalence test for Stage 2."""
    pass


@pytest.mark.skip(
    reason="Phase 5 intentionally changes Stages 2-4 output. "
    "Phase 4 reference preserved in tests/reference_v2/ for comparison."
)
def test_stage3_supports_v2_identical():
    """[ARCHIVED] Phase 4 equivalence test for Stage 3."""
    pass


@pytest.mark.skip(
    reason="Phase 5 intentionally changes Stages 2-4 output. "
    "Phase 4 reference preserved in tests/reference_v2/ for comparison."
)
def test_stage4_edge_dist_v2_identical():
    """[ARCHIVED] Phase 4 equivalence test for Stage 4 edge_dist."""
    pass


@pytest.mark.skip(
    reason="Phase 5 intentionally changes Stages 2-4 output. "
    "Phase 4 reference preserved in tests/reference_v2/ for comparison."
)
def test_stage4_edge_dir_v2_identical():
    """[ARCHIVED] Phase 4 equivalence test for Stage 4 edge_dir."""
    pass


@pytest.mark.skip(
    reason="Phase 5 intentionally changes Stages 2-4 output. "
    "Phase 4 reference preserved in tests/reference_v2/ for comparison."
)
def test_stage4_edge_valid_v2_identical():
    """[ARCHIVED] Phase 4 equivalence test for Stage 4 edge_valid."""
    pass


@pytest.mark.skip(
    reason="Phase 5 intentionally changes Stages 2-4 output. "
    "Phase 4 reference preserved in tests/reference_v2/ for comparison."
)
def test_stage4_edge_support_id_v2_identical():
    """[ARCHIVED] Phase 4 equivalence test for Stage 4 edge_support_id."""
    pass


# ---------------------------------------------------------------------------
# Phase 5 equivalence: Stages 2-4 against reference_v3/
# ---------------------------------------------------------------------------


@_SKIP_NO_REF_V3
def test_stage2_local_clusters_v3_identical(phase5_stage2):
    _assert_npz_identical(
        REFERENCE_V3_DIR / "local_clusters.npz",
        phase5_stage2,
        "Stage 2 local_clusters (Phase 5)",
    )


@_SKIP_NO_REF_V3
def test_stage3_supports_v3_identical(phase5_stage3):
    _assert_npz_identical(
        REFERENCE_V3_DIR / "supports.npz",
        phase5_stage3,
        "Stage 3 supports (Phase 5)",
    )


@_SKIP_NO_REF_V3
def test_stage4_edge_dist_v3_identical(phase5_stage4):
    _assert_npy_identical(
        REFERENCE_V3_DIR / "edge_dist.npy",
        phase5_stage4["edge_dist"],
        "Stage 4 edge_dist (Phase 5)",
    )


@_SKIP_NO_REF_V3
def test_stage4_edge_dir_v3_identical(phase5_stage4):
    _assert_npy_identical(
        REFERENCE_V3_DIR / "edge_dir.npy",
        phase5_stage4["edge_dir"],
        "Stage 4 edge_dir (Phase 5)",
    )


@_SKIP_NO_REF_V3
def test_stage4_edge_valid_v3_identical(phase5_stage4):
    _assert_npy_identical(
        REFERENCE_V3_DIR / "edge_valid.npy",
        phase5_stage4["edge_valid"],
        "Stage 4 edge_valid (Phase 5)",
    )


@_SKIP_NO_REF_V3
def test_stage4_edge_support_id_v3_identical(phase5_stage4):
    _assert_npy_identical(
        REFERENCE_V3_DIR / "edge_support_id.npy",
        phase5_stage4["edge_support_id"],
        "Stage 4 edge_support_id (Phase 5)",
    )


# ---------------------------------------------------------------------------
# Cross-check: validation hooks pass on Phase 5 output
# ---------------------------------------------------------------------------


def test_validation_passes_on_phase5_output(
    refactored_stage1,
    phase5_stage2,
    phase5_stage3,
    phase5_stage4,
    sample_scene_dir,
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

    validate_boundary_centers(refactored_stage1)
    validate_local_clusters(
        phase5_stage2,
        num_boundary_centers=refactored_stage1["center_coord"].shape[0],
    )
    cfg2 = Stage2Config()
    validate_cluster_contract(
        boundary_centers=refactored_stage1,
        local_clusters=phase5_stage2,
        direction_cos_th=cfg2.segment_direction_cos_th,
    )
    validate_supports(phase5_stage3)
    scene = load_scene(sample_scene_dir)
    validate_edge_supervision(phase5_stage4, num_scene_points=scene["coord"].shape[0])


# ---------------------------------------------------------------------------
# In-memory path: Stages 1->2->3 in sequence matches per-stage results
# ---------------------------------------------------------------------------


def test_inmemory_path_matches_perstage(sample_scene_dir, phase5_stage3):
    """Run Stages 1->2->3 in-memory (build_support_dataset_v3 pattern) and
    compare Stage 3 output against the per-stage fixture result."""
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

    # Compare every field in supports payload
    ref_keys = sorted(phase5_stage3.keys())
    mem_keys = sorted(sup_payload.keys())
    assert ref_keys == mem_keys, f"Key mismatch: per-stage={ref_keys}, in-memory={mem_keys}"
    for key in ref_keys:
        assert np.array_equal(phase5_stage3[key], sup_payload[key]), (
            f"In-memory path differs from per-stage at supports[{key}]"
        )
