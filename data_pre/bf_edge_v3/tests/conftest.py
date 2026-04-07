"""Shared pytest configuration and fixtures for bf_edge_v3 tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Mirror _bootstrap.py: insert bf_edge_v3 root at position 0 so that
# ``from core.xxx import ...`` works identically to the scripts/ layer.
_BF_EDGE_V3_ROOT = Path(__file__).resolve().parents[1]
if str(_BF_EDGE_V3_ROOT) not in sys.path:
    sys.path.insert(0, str(_BF_EDGE_V3_ROOT))

SAMPLE_SCENE_DIR = _BF_EDGE_V3_ROOT / "samples" / "010101"
REFERENCE_DIR = _BF_EDGE_V3_ROOT / "tests" / "reference"
REFERENCE_V2_DIR = _BF_EDGE_V3_ROOT / "tests" / "reference_v2"


@pytest.fixture(scope="session")
def sample_scene_dir() -> Path:
    """Return the sample scene directory, skipping if it does not exist."""
    if not (SAMPLE_SCENE_DIR / "coord.npy").exists():
        pytest.skip("Sample scene 010101 not found on disk")
    return SAMPLE_SCENE_DIR


@pytest.fixture(scope="session")
def reference_dir() -> Path:
    """Return the reference data directory."""
    return REFERENCE_DIR


@pytest.fixture(scope="session")
def reference_v2_dir() -> Path:
    """Return the Phase 4 reference data directory."""
    return REFERENCE_V2_DIR


@pytest.fixture(scope="module")
def phase4_stage2(sample_scene_dir):
    """Run Stage 2 with Phase 4 config (rescue + direction + spatial splitting)."""
    from core.config import Stage1Config, Stage2Config
    from core.boundary_centers_core import build_boundary_centers
    from core.local_clusters_core import cluster_boundary_centers
    from utils.stage_io import load_scene

    cfg1 = Stage1Config()
    scene = load_scene(sample_scene_dir)
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
