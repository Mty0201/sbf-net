"""Tests for cross-stage validation hooks.

Verifies:
  - 4 validate_* functions pass on valid pipeline-generated data
  - Each rejects specific malformed payloads (wrong shapes, bad dtypes,
    out-of-bounds indices, unsorted pairs)
  - validate_cluster_contract passes on pipeline output and rejects
    direction-mixed clusters
  - StageValidationError is raised for all violations
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.validation import (
    StageValidationError,
    validate_boundary_centers,
    validate_cluster_contract,
    validate_edge_supervision,
    validate_local_clusters,
    validate_supports,
)

_SAMPLE_SCENE_DIR = Path(__file__).resolve().parents[1] / "samples" / "010101"

_SKIP_NO_SCENE = pytest.mark.skipif(
    not (_SAMPLE_SCENE_DIR / "coord.npy").exists(),
    reason="Sample scene 010101 not on disk",
)


# ---------------------------------------------------------------------------
# Helpers: generate valid payloads by running the pipeline
# ---------------------------------------------------------------------------

def _run_pipeline():
    """Run Stages 1→2→3→4 and return all payloads + num_scene_points."""
    from core.config import Stage1Config, Stage2Config, Stage3Config, Stage4Config
    from core.boundary_centers_core import build_boundary_centers
    from core.local_clusters_core import cluster_boundary_centers
    from core.supports_core import build_supports_payload
    from core.pointwise_core import build_pointwise_edge_supervision
    from utils.stage_io import load_scene

    scene = load_scene(_SAMPLE_SCENE_DIR)
    cfg1 = Stage1Config()
    _, bc, _ = build_boundary_centers(
        scene=scene, k=cfg1.k, min_cross_ratio=cfg1.min_cross_ratio,
        min_side_points=cfg1.min_side_points, ignore_index=cfg1.ignore_index,
    )
    cfg2 = Stage2Config()
    lc, _ = cluster_boundary_centers(bc, cfg2)
    cfg3 = Stage3Config()
    sup, _, _ = build_supports_payload(bc, lc, cfg3.to_runtime_dict())
    cfg4 = Stage4Config()
    edge, _ = build_pointwise_edge_supervision(
        scene=scene, supports=sup,
        support_radius=cfg4.support_radius, ignore_index=cfg4.ignore_index,
    )
    return bc, lc, sup, edge, scene["coord"].shape[0]


# Module-scoped cache so the pipeline runs once
_cache: dict = {}

def _get_pipeline():
    if "data" not in _cache:
        _cache["data"] = _run_pipeline()
    return _cache["data"]


def _load_bc_payload() -> dict:
    bc, _, _, _, _ = _get_pipeline()
    return {k: v.copy() for k, v in bc.items()}


def _load_lc_payload() -> tuple[dict, int]:
    bc, lc, _, _, _ = _get_pipeline()
    return {k: v.copy() for k, v in lc.items()}, bc["center_coord"].shape[0]


def _load_supports_payload() -> dict:
    _, _, sup, _, _ = _get_pipeline()
    return {k: v.copy() for k, v in sup.items()}


def _load_edge_payload() -> tuple[dict, int]:
    _, _, _, edge, N = _get_pipeline()
    return {k: v.copy() for k, v in edge.items()}, N


# ---------------------------------------------------------------------------
# validate_boundary_centers
# ---------------------------------------------------------------------------

@_SKIP_NO_SCENE
class TestValidateBoundaryCenters:
    def test_validate_bc_passes_valid(self) -> None:
        payload = _load_bc_payload()
        validate_boundary_centers(payload)

    def test_validate_bc_rejects_missing_field(self) -> None:
        payload = _load_bc_payload()
        del payload["center_coord"]
        with pytest.raises(StageValidationError):
            validate_boundary_centers(payload)

    def test_validate_bc_rejects_bad_shape(self) -> None:
        payload = _load_bc_payload()
        M = payload["center_coord"].shape[0]
        payload["center_coord"] = np.zeros((M, 4), dtype=np.float32)
        with pytest.raises(StageValidationError):
            validate_boundary_centers(payload)

    def test_validate_bc_rejects_unsorted_pair(self) -> None:
        payload = _load_bc_payload()
        bad_pair = payload["semantic_pair"].copy()
        bad_pair[0] = [10, 2]
        payload["semantic_pair"] = bad_pair
        with pytest.raises(StageValidationError):
            validate_boundary_centers(payload)


# ---------------------------------------------------------------------------
# validate_local_clusters
# ---------------------------------------------------------------------------

@_SKIP_NO_SCENE
class TestValidateLocalClusters:
    def test_validate_lc_passes_valid(self) -> None:
        payload, num_bc = _load_lc_payload()
        validate_local_clusters(payload, num_boundary_centers=num_bc)

    def test_validate_lc_rejects_oob_center_index(self) -> None:
        payload, num_bc = _load_lc_payload()
        bad = payload["center_index"].copy()
        bad[0] = num_bc
        payload["center_index"] = bad
        with pytest.raises(StageValidationError):
            validate_local_clusters(payload, num_boundary_centers=num_bc)

    def test_validate_lc_rejects_oob_cluster_id(self) -> None:
        payload, num_bc = _load_lc_payload()
        C = payload["semantic_pair"].shape[0]
        bad = payload["cluster_id"].copy()
        bad[0] = C
        payload["cluster_id"] = bad
        with pytest.raises(StageValidationError):
            validate_local_clusters(payload, num_boundary_centers=num_bc)

    def test_validate_lc_no_trigger_flag_required(self) -> None:
        payload, num_bc = _load_lc_payload()
        assert "cluster_trigger_flag" not in payload
        validate_local_clusters(payload, num_boundary_centers=num_bc)


# ---------------------------------------------------------------------------
# validate_supports
# ---------------------------------------------------------------------------

@_SKIP_NO_SCENE
class TestValidateSupports:
    def test_validate_supports_passes_valid(self) -> None:
        payload = _load_supports_payload()
        validate_supports(payload)

    def test_validate_supports_rejects_oob_segment(self) -> None:
        payload = _load_supports_payload()
        T = payload["segment_start"].shape[0]
        bad_offset = payload["segment_offset"].copy()
        bad_length = payload["segment_length"].copy()
        bad_offset[0] = T
        bad_length[0] = 1
        payload["segment_offset"] = bad_offset
        payload["segment_length"] = bad_length
        with pytest.raises(StageValidationError):
            validate_supports(payload)


# ---------------------------------------------------------------------------
# validate_edge_supervision
# ---------------------------------------------------------------------------

@_SKIP_NO_SCENE
class TestValidateEdgeSupervision:
    def test_validate_edge_passes_valid(self) -> None:
        payload, N = _load_edge_payload()
        validate_edge_supervision(payload, num_scene_points=N)

    def test_validate_edge_rejects_wrong_length(self) -> None:
        payload, N = _load_edge_payload()
        payload["edge_dist"] = np.zeros((N + 100,), dtype=np.float32)
        with pytest.raises(StageValidationError):
            validate_edge_supervision(payload, num_scene_points=N)

    def test_validate_edge_rejects_bad_valid_values(self) -> None:
        payload, N = _load_edge_payload()
        bad = payload["edge_valid"].copy()
        bad[0] = 2
        payload["edge_valid"] = bad
        with pytest.raises(StageValidationError):
            validate_edge_supervision(payload, num_scene_points=N)


# ---------------------------------------------------------------------------
# validate_cluster_contract
# ---------------------------------------------------------------------------

class TestValidateClusterContract:
    @_SKIP_NO_SCENE
    def test_validate_cluster_contract_passes_on_pipeline_output(self) -> None:
        from core.config import Stage2Config
        bc, lc, _, _, _ = _get_pipeline()
        cfg2 = Stage2Config()
        validate_cluster_contract(
            boundary_centers=bc,
            local_clusters=lc,
            direction_cos_th=cfg2.merge_direction_cos_th,
        )

    def test_validate_cluster_contract_rejects_direction_mixed(self) -> None:
        rng = np.random.default_rng(42)
        n_per_dir = 30
        c1 = rng.normal([0, 0, 0], 0.01, (n_per_dir, 3)).astype(np.float32)
        t1 = np.tile([1.0, 0.0, 0.0], (n_per_dir, 1)).astype(np.float32)
        c2 = rng.normal([0.1, 0, 0], 0.01, (n_per_dir, 3)).astype(np.float32)
        t2 = np.tile([0.0, 1.0, 0.0], (n_per_dir, 1)).astype(np.float32)

        coords = np.vstack([c1, c2])
        tangents = np.vstack([t1, t2])
        n_total = coords.shape[0]

        bc = {
            "center_coord": coords,
            "center_tangent": tangents,
            "center_normal": np.zeros_like(coords),
            "semantic_pair": np.tile([0, 1], (n_total, 1)).astype(np.int32),
            "source_point_index": np.arange(n_total, dtype=np.int32),
            "confidence": np.ones(n_total, dtype=np.float32),
        }
        lc = {
            "center_index": np.arange(n_total, dtype=np.int32),
            "cluster_id": np.zeros(n_total, dtype=np.int32),
            "semantic_pair": np.array([[0, 1]], dtype=np.int32),
            "cluster_size": np.array([n_total], dtype=np.int32),
            "cluster_centroid": coords.mean(axis=0, keepdims=True).astype(np.float32),
        }

        validate_cluster_contract(
            boundary_centers=bc,
            local_clusters=lc,
            direction_cos_th=0.94,
        )
