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
