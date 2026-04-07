"""DEN-02 and DEN-03 verification: density-adaptive supervision gap tests.

Tests:
  - Programmatic gap measurement on 020101 and 020102 scenes
  - Dense-region non-regression check
  - Synthetic density-conditional denoise behavior
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup: bf_edge_v3 root must be on sys.path for core.* imports.
# conftest.py handles this, but we also need scripts/tools/ for diagnose funcs.
# ---------------------------------------------------------------------------

_BF_EDGE_V3_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _BF_EDGE_V3_ROOT / "scripts" / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from diagnose_net01 import (
    compute_scene_knn_distances,
    assign_density_buckets,
    analyze_stage2,
)

from core.config import Stage2Config
from core.local_clusters_core import cluster_boundary_centers
from utils.stage_io import load_boundary_centers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEN_02_GAP_THRESHOLD = 0.05  # 5 percentage points
DEN_03_DENSE_RATE_MIN = 0.99

# parents[1] = bf_edge_v3, parents[2] = data_pre, parents[3] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
SCENE_020101 = _REPO_ROOT / "samples" / "training" / "020101"
SCENE_020102 = _REPO_ROOT / "samples" / "validation" / "020102"

_SKIP_NO_020101 = pytest.mark.skipif(
    not (SCENE_020101 / "coord.npy").exists()
    or not (SCENE_020101 / "boundary_centers.npz").exists(),
    reason="Scene 020101 not on disk (requires samples/training/020101/)",
)
_SKIP_NO_020102 = pytest.mark.skipif(
    not (SCENE_020102 / "coord.npy").exists()
    or not (SCENE_020102 / "boundary_centers.npz").exists(),
    reason="Scene 020102 not on disk (requires samples/validation/020102/)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _measure_gap(scene_dir: Path) -> dict:
    """Run Phase 5 clustering and measure density gap.

    Returns dict with keys: dense_rate, sparse_rate, gap, num_denoise_skipped.
    Uses the same P25/P75 bucketing methodology as diagnose_net01.py for
    consistent measurement (T-05-05 mitigation).
    """
    coord = np.load(scene_dir / "coord.npy").astype(np.float32)
    bc = load_boundary_centers(scene_dir)

    # Run Phase 5 clustering with default config
    cfg = Stage2Config()
    lc_payload, meta = cluster_boundary_centers(bc, cfg)

    # Compute scene-wide kNN density (k=10)
    mean_knn_dist = compute_scene_knn_distances(coord, k=10)

    # Density buckets for boundary centers
    source_idx = bc["source_point_index"]
    center_densities = mean_knn_dist[source_idx]
    p25 = float(np.percentile(center_densities, 25))
    p75 = float(np.percentile(center_densities, 75))
    center_buckets = assign_density_buckets(center_densities, p25, p75)

    # Stage 2 survival analysis per bucket
    stage2 = analyze_stage2(bc, lc_payload, center_buckets)

    dense_rate = stage2["dense"]["rate"]
    sparse_rate = stage2["sparse"]["rate"]
    gap = dense_rate - sparse_rate

    return {
        "dense_rate": dense_rate,
        "sparse_rate": sparse_rate,
        "gap": gap,
        "num_denoise_skipped": meta["num_denoise_skipped"],
        "num_clusters": meta["num_clusters"],
        "num_assigned": meta["num_assigned"],
    }


# ---------------------------------------------------------------------------
# DEN-02: Sparse-to-dense gap < 5pp
# ---------------------------------------------------------------------------


@_SKIP_NO_020101
def test_den02_gap_020101():
    """DEN-02: Stage 2 survival gap < 5pp on 020101."""
    result = _measure_gap(SCENE_020101)
    gap = result["gap"]
    assert gap < DEN_02_GAP_THRESHOLD, (
        f"DEN-02 FAIL on 020101: gap={gap:.4f} ({gap*100:.1f}pp) >= {DEN_02_GAP_THRESHOLD}"
        f"\n  dense_rate={result['dense_rate']:.4f}, sparse_rate={result['sparse_rate']:.4f}"
        f"\n  clusters={result['num_clusters']}, assigned={result['num_assigned']}"
    )


@_SKIP_NO_020102
def test_den02_gap_020102():
    """DEN-02: Stage 2 survival gap < 5pp on 020102."""
    result = _measure_gap(SCENE_020102)
    gap = result["gap"]
    assert gap < DEN_02_GAP_THRESHOLD, (
        f"DEN-02 FAIL on 020102: gap={gap:.4f} ({gap*100:.1f}pp) >= {DEN_02_GAP_THRESHOLD}"
        f"\n  dense_rate={result['dense_rate']:.4f}, sparse_rate={result['sparse_rate']:.4f}"
        f"\n  clusters={result['num_clusters']}, assigned={result['num_assigned']}"
    )


# ---------------------------------------------------------------------------
# DEN-03: Dense-region non-regression (>= 0.99)
# ---------------------------------------------------------------------------


@_SKIP_NO_020101
def test_den03_dense_rate_020101():
    """DEN-03: Dense-region survival rate >= 0.99 on 020101."""
    result = _measure_gap(SCENE_020101)
    dense_rate = result["dense_rate"]
    assert dense_rate >= DEN_03_DENSE_RATE_MIN, (
        f"DEN-03 FAIL on 020101: dense_rate={dense_rate:.4f} < {DEN_03_DENSE_RATE_MIN}"
    )


@_SKIP_NO_020102
def test_den03_dense_rate_020102():
    """DEN-03: Dense-region survival rate >= 0.99 on 020102."""
    result = _measure_gap(SCENE_020102)
    dense_rate = result["dense_rate"]
    assert dense_rate >= DEN_03_DENSE_RATE_MIN, (
        f"DEN-03 FAIL on 020102: dense_rate={dense_rate:.4f} < {DEN_03_DENSE_RATE_MIN}"
    )


# ---------------------------------------------------------------------------
# Synthetic: density-conditional denoise behavior
# ---------------------------------------------------------------------------


def test_density_conditional_denoise_synthetic():
    """Verify that sparse clusters skip denoise while dense clusters do not.

    Creates synthetic boundary centers with two semantic pairs:
    - Pair (1,2): dense cluster with tight spacing (~0.005)
    - Pair (3,4): sparse cluster with looser spacing (~0.03)

    Both spacings are within DBSCAN eps=0.08 for clustering, but the sparse
    cluster's internal spacing exceeds the denoise_density_threshold * global
    median, so denoise is skipped for it.

    Checks that num_denoise_skipped > 0 (sparse cluster skipped denoise)
    and that the sparse cluster retains all its points.
    """
    rng = np.random.RandomState(42)

    # Dense cluster: 30 points in a tight line, spacing ~0.005 (well within eps=0.08)
    n_dense = 30
    dense_x = np.arange(n_dense, dtype=np.float32) * 0.005
    dense_coords = np.column_stack([
        dense_x,
        rng.normal(0, 0.0005, n_dense).astype(np.float32),
        rng.normal(0, 0.0005, n_dense).astype(np.float32),
    ])
    dense_tangents = np.tile([1.0, 0.0, 0.0], (n_dense, 1)).astype(np.float32)
    dense_pairs = np.tile([1, 2], (n_dense, 1)).astype(np.int32)

    # Sparse cluster: 30 points in a looser line, spacing ~0.03 (still within eps=0.08)
    # Needs min_samples=5 neighbors within eps, so 0.03 spacing gives ~2 neighbors
    # within eps=0.08 on each side = 5+ total neighbors per point.
    n_sparse = 30
    sparse_x = np.arange(n_sparse, dtype=np.float32) * 0.03
    sparse_coords = np.column_stack([
        sparse_x + 10.0,  # offset so DBSCAN doesn't merge with dense
        rng.normal(0, 0.001, n_sparse).astype(np.float32),
        rng.normal(0, 0.001, n_sparse).astype(np.float32),
    ])
    sparse_tangents = np.tile([1.0, 0.0, 0.0], (n_sparse, 1)).astype(np.float32)
    sparse_pairs = np.tile([3, 4], (n_sparse, 1)).astype(np.int32)

    # Combine into boundary_centers dict
    center_coord = np.concatenate([dense_coords, sparse_coords], axis=0)
    center_tangent = np.concatenate([dense_tangents, sparse_tangents], axis=0)
    semantic_pair = np.concatenate([dense_pairs, sparse_pairs], axis=0)
    n_total = center_coord.shape[0]

    boundary_centers = {
        "center_coord": center_coord.astype(np.float32),
        "center_tangent": center_tangent.astype(np.float32),
        "semantic_pair": semantic_pair.astype(np.int32),
        "source_point_index": np.arange(n_total, dtype=np.int32),
        "confidence": np.ones(n_total, dtype=np.float32),
    }

    cfg = Stage2Config()
    payload, meta = cluster_boundary_centers(boundary_centers, cfg)

    # The sparse cluster should have triggered a denoise skip
    # (sparse spacing ~0.03 > 0.5 * global_median where global_median is
    # dominated by the mixed spacing of both clusters)
    assert meta["num_denoise_skipped"] >= 1, (
        f"Expected at least 1 denoise skip for sparse cluster, got {meta['num_denoise_skipped']}"
    )

    # Sparse cluster points (indices n_dense..n_total-1) should mostly survive
    survived_set = set(payload["center_index"].tolist())
    sparse_indices = set(range(n_dense, n_total))
    sparse_survived = sparse_indices & survived_set
    sparse_survival_rate = len(sparse_survived) / len(sparse_indices)

    # With denoise skipped, sparse survival should be high (only losses from
    # DBSCAN noise and min_points filtering at cluster edges)
    assert sparse_survival_rate >= 0.8, (
        f"Sparse cluster survival too low: {sparse_survival_rate:.2f} "
        f"({len(sparse_survived)}/{len(sparse_indices)})"
    )
