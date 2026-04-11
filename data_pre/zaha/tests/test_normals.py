"""Tests for ``data_pre.zaha.utils.normals`` — DS-ZAHA-P1-05.

GREEN tests matching the behaviour contract in 01-03-PLAN.md Task 3a:
- test_unit_length: on a 20x20 flat plane fixture, all estimated normals
  have ``‖n‖ > 0.99`` (effectively unit length up to float32 precision).
- test_no_nan: on the same fixture, ``np.isfinite(normals).all()``.
- test_degenerate_fallback: a 5-point collinear chunk is rejected (< knn=30)
  with a clear ValueError; a 50-point collinear chunk is either accepted
  with valid normals or rejected with a clear ValueError — it MUST NEVER
  return NaN.

Target impl module: ``data_pre.zaha.utils.normals`` (created in Plan 03).
Behaviour references: CONTEXT.md D-17 (adaptive-radius PCA, k ≈ 30 default),
D-18 (acceptance bar: unit-length float32 (N, 3), no NaN; "thinning the
wall" aspirational, not blocking), D-19 (per-chunk, not whole-building),
VALIDATION.md rows ``P01-normal-01..03``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _load():
    """Guarded import — open3d is required for this module."""
    return pytest.importorskip("data_pre.zaha.utils.normals")


def _flat_plane(n_side: int = 20) -> np.ndarray:
    """20x20 = 400-point fixture on z ≈ 0 with sub-mm gaussian noise."""
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(
        np.linspace(0.0, 1.0, n_side),
        np.linspace(0.0, 1.0, n_side),
    )
    z = 0.001 * rng.standard_normal(gx.shape)
    return np.stack([gx.ravel(), gy.ravel(), z.ravel()], axis=1).astype(
        np.float32
    )


def test_unit_length() -> None:
    """P01-normal-01 — all estimated normals have ``‖n‖ > 0.99`` (D-18)."""
    normals_mod = _load()
    xyz = _flat_plane()
    cfg = normals_mod.NormalConfig(knn=30)
    normals = normals_mod.estimate_normals(xyz, cfg)

    assert normals.shape == xyz.shape, (
        f"expected {xyz.shape}, got {normals.shape}"
    )
    assert normals.dtype == np.float32, (
        f"expected float32, got {normals.dtype}"
    )
    norms = np.linalg.norm(normals, axis=1)
    assert norms.min() > 0.99, f"min norm {norms.min()}"
    assert norms.max() < 1.01, f"max norm {norms.max()}"


def test_no_nan() -> None:
    """P01-normal-02 — no NaN or inf in the output normals (D-18)."""
    normals_mod = _load()
    xyz = _flat_plane()
    cfg = normals_mod.NormalConfig(knn=30)
    normals = normals_mod.estimate_normals(xyz, cfg)
    assert np.isfinite(normals).all(), "normals must be finite everywhere"


def test_degenerate_fallback() -> None:
    """P01-normal-03 — degenerate input raises clear errors, never NaN.

    Two sub-cases:

    1. **5-point collinear.** n=5 is less than knn=30, so the implementation
       must raise ``ValueError`` with a message naming the ``n < knn`` case
       (so the caller knows to widen knn or drop the chunk).
    2. **50-point collinear.** n=50 is >= knn=30, so the estimator can
       actually run. It may either (a) return valid unit-length normals
       (PCA picks an arbitrary perpendicular) or (b) raise ValueError naming
       the degeneracy. It MUST NOT return NaN or non-unit normals.
    """
    normals_mod = _load()

    # Sub-case 1: 5-point collinear is below knn and must raise.
    xyz_small = np.stack(
        [
            np.linspace(0.0, 1.0, 5),
            np.zeros(5),
            np.zeros(5),
        ],
        axis=1,
    ).astype(np.float32)
    cfg = normals_mod.NormalConfig(knn=30)
    with pytest.raises(ValueError):
        normals_mod.estimate_normals(xyz_small, cfg)

    # Sub-case 2: 50-point collinear — the estimator has enough points for
    # KNN but the neighbourhood is rank-deficient. Either path is valid so
    # long as NaN never leaks.
    xyz_big = np.stack(
        [
            np.linspace(0.0, 10.0, 50),
            np.zeros(50),
            np.zeros(50),
        ],
        axis=1,
    ).astype(np.float32)

    try:
        normals = normals_mod.estimate_normals(xyz_big, cfg)
    except ValueError as exc:
        msg = str(exc).lower()
        allowed = ("degeneracy", "degenerate", "collinear", "non-finite", "||n||")
        assert any(keyword in msg for keyword in allowed), (
            f"degenerate error must mention reason, got: {exc}"
        )
    else:
        assert np.isfinite(normals).all(), (
            "estimator returned non-finite normals on degenerate input"
        )
        norms = np.linalg.norm(normals, axis=1)
        assert norms.min() > 0.99, (
            f"estimator returned sub-unit normals on degenerate input: "
            f"min={norms.min()}"
        )
