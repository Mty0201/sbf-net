"""Tests for cross-stage validation hooks (REF-05).

Verifies:
  - 4 validate_* functions pass on valid 010101 reference data
  - Each rejects specific malformed payloads (wrong shapes, bad dtypes,
    out-of-bounds indices, unsorted pairs)
  - StageValidationError is raised for all violations

Reference data layout in tests/reference/:
  - boundary_centers.npz  (Stage 1 output)
  - local_clusters.npz    (Stage 2 output)
  - supports.npz          (Stage 3 output)
  - edge_*.npy            (Stage 4 output arrays)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.validation import (
    StageValidationError,
    validate_boundary_centers,
    validate_edge_supervision,
    validate_local_clusters,
    validate_supports,
)

# Resolve reference and sample directories relative to this file,
# mirroring the constants defined in conftest.py.
_BF_EDGE_V3_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_DIR = _BF_EDGE_V3_ROOT / "tests" / "reference"
_SAMPLE_SCENE_DIR = _BF_EDGE_V3_ROOT / "samples" / "010101"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_bc_payload() -> dict:
    """Load boundary centers reference payload as a plain dict."""
    npz = np.load(_REFERENCE_DIR / "boundary_centers.npz")
    return {k: npz[k] for k in npz.files}


def _load_lc_payload() -> tuple[dict, int]:
    """Load local clusters reference payload and num_boundary_centers."""
    npz = np.load(_REFERENCE_DIR / "local_clusters.npz")
    bc_npz = np.load(_REFERENCE_DIR / "boundary_centers.npz")
    num_bc = bc_npz["center_coord"].shape[0]
    return {k: npz[k] for k in npz.files}, num_bc


def _load_supports_payload() -> dict:
    """Load supports reference payload as a plain dict."""
    npz = np.load(_REFERENCE_DIR / "supports.npz")
    return {k: npz[k] for k in npz.files}


def _load_edge_payload() -> tuple[dict, int]:
    """Load edge supervision reference payload and num_scene_points."""
    payload = {
        "edge_dist": np.load(_REFERENCE_DIR / "edge_dist.npy"),
        "edge_dir": np.load(_REFERENCE_DIR / "edge_dir.npy"),
        "edge_valid": np.load(_REFERENCE_DIR / "edge_valid.npy"),
        "edge_support_id": np.load(_REFERENCE_DIR / "edge_support_id.npy"),
    }
    coord = np.load(_SAMPLE_SCENE_DIR / "coord.npy")
    return payload, coord.shape[0]


# ---------------------------------------------------------------------------
# validate_boundary_centers
# ---------------------------------------------------------------------------

class TestValidateBoundaryCenters:
    def test_validate_bc_passes_valid(self) -> None:
        """Reference 010101 boundary centers must pass validation."""
        payload = _load_bc_payload()
        validate_boundary_centers(payload)  # no exception = pass

    def test_validate_bc_rejects_missing_field(self) -> None:
        """Removing a required field must raise StageValidationError."""
        payload = _load_bc_payload()
        del payload["center_coord"]
        with pytest.raises(StageValidationError):
            validate_boundary_centers(payload)

    def test_validate_bc_rejects_bad_shape(self) -> None:
        """center_coord with shape (M, 4) must raise StageValidationError."""
        payload = _load_bc_payload()
        M = payload["center_coord"].shape[0]
        payload["center_coord"] = np.zeros((M, 4), dtype=np.float32)
        with pytest.raises(StageValidationError):
            validate_boundary_centers(payload)

    def test_validate_bc_rejects_unsorted_pair(self) -> None:
        """semantic_pair where col0 > col1 must raise StageValidationError."""
        payload = _load_bc_payload()
        bad_pair = payload["semantic_pair"].copy()
        bad_pair[0] = [10, 2]  # unsorted: 10 > 2
        payload["semantic_pair"] = bad_pair
        with pytest.raises(StageValidationError):
            validate_boundary_centers(payload)


# ---------------------------------------------------------------------------
# validate_local_clusters
# ---------------------------------------------------------------------------

class TestValidateLocalClusters:
    def test_validate_lc_passes_valid(self) -> None:
        """Reference 010101 local clusters must pass validation."""
        payload, num_bc = _load_lc_payload()
        validate_local_clusters(payload, num_boundary_centers=num_bc)

    def test_validate_lc_rejects_oob_center_index(self) -> None:
        """center_index with value >= M must raise StageValidationError."""
        payload, num_bc = _load_lc_payload()
        bad = payload["center_index"].copy()
        bad[0] = num_bc  # out of bounds: == M
        payload["center_index"] = bad
        with pytest.raises(StageValidationError):
            validate_local_clusters(payload, num_boundary_centers=num_bc)

    def test_validate_lc_rejects_oob_cluster_id(self) -> None:
        """cluster_id with value >= C must raise StageValidationError."""
        payload, num_bc = _load_lc_payload()
        C = payload["semantic_pair"].shape[0]
        bad = payload["cluster_id"].copy()
        bad[0] = C  # out of bounds: == C
        payload["cluster_id"] = bad
        with pytest.raises(StageValidationError):
            validate_local_clusters(payload, num_boundary_centers=num_bc)


# ---------------------------------------------------------------------------
# validate_supports
# ---------------------------------------------------------------------------

class TestValidateSupports:
    def test_validate_supports_passes_valid(self) -> None:
        """Reference 010101 supports must pass validation."""
        payload = _load_supports_payload()
        validate_supports(payload)

    def test_validate_supports_rejects_oob_segment(self) -> None:
        """segment_offset[i]+segment_length[i] > T must raise StageValidationError."""
        payload = _load_supports_payload()
        T = payload["segment_start"].shape[0]
        bad_offset = payload["segment_offset"].copy()
        bad_length = payload["segment_length"].copy()
        # Force an out-of-bounds: set first support to offset=T, length=1
        bad_offset[0] = T
        bad_length[0] = 1
        payload["segment_offset"] = bad_offset
        payload["segment_length"] = bad_length
        with pytest.raises(StageValidationError):
            validate_supports(payload)


# ---------------------------------------------------------------------------
# validate_edge_supervision
# ---------------------------------------------------------------------------

class TestValidateEdgeSupervision:
    def test_validate_edge_passes_valid(self) -> None:
        """Reference 010101 edge supervision must pass validation."""
        payload, N = _load_edge_payload()
        validate_edge_supervision(payload, num_scene_points=N)

    def test_validate_edge_rejects_wrong_length(self) -> None:
        """edge_dist with wrong length must raise StageValidationError."""
        payload, N = _load_edge_payload()
        payload["edge_dist"] = np.zeros((N + 100,), dtype=np.float32)
        with pytest.raises(StageValidationError):
            validate_edge_supervision(payload, num_scene_points=N)

    def test_validate_edge_rejects_bad_valid_values(self) -> None:
        """edge_valid with value=2 must raise StageValidationError."""
        payload, N = _load_edge_payload()
        bad = payload["edge_valid"].copy()
        bad[0] = 2  # not in {0, 1}
        payload["edge_valid"] = bad
        with pytest.raises(StageValidationError):
            validate_edge_supervision(payload, num_scene_points=N)
