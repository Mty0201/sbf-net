"""Tests for ``data_pre.zaha.utils.denoise`` — DS-ZAHA-P1-03.

GREEN tests matching the behaviour contract in 01-03-PLAN.md Task 1:
- test_drop_cap: SOR drops between 1% and 10% of a synthetic cube+outliers cloud (D-12).
- test_determinism: two sequential calls on the same input produce bitwise-equal output.

Target impl module: ``data_pre.zaha.utils.denoise`` (created in Plan 03).
Behaviour references: CONTEXT.md D-11 (3 candidates), D-12 (≤10% drop cap,
visible stripe reduction), D-13 (pre-chunking on whole-building cloud),
VALIDATION.md rows ``P01-denoise-01..02``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _load_denoise():
    """Guarded import — skip if open3d is missing from the test env."""
    return pytest.importorskip("data_pre.zaha.utils.denoise")


def _build_cube_plus_outliers() -> tuple[np.ndarray, np.ndarray]:
    """1000-point 10×10×10 grid at 0.05 m spacing + 100 uniform-noise outliers."""
    rng = np.random.default_rng(0)
    gx, gy, gz = np.mgrid[0:10, 0:10, 0:10]
    cube = (
        np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
        * 0.05
    )
    outliers = rng.uniform(-1.0, 1.0, size=(100, 3)).astype(np.float32)
    xyz = np.concatenate([cube, outliers], axis=0)
    seg = np.zeros(len(xyz), dtype=np.int32)
    return xyz, seg


def test_drop_cap() -> None:
    """P01-denoise-01 — SOR drops between 1% and 10% of points on the fixture (D-12).

    Hard cap from CONTEXT D-12: Phase 1 does not ship if any sample loses more
    than 10 %. The low bound (1 %) makes sure the filter is actually doing
    something (not a no-op).
    """
    denoise = _load_denoise()
    xyz, seg = _build_cube_plus_outliers()
    cfg = denoise.DenoiseConfig(
        method="sor", params={"nb_neighbors": 20, "std_ratio": 2.0}
    )
    result = denoise.denoise_cloud(xyz, seg, cfg)
    assert 0.01 <= result.drop_frac <= 0.10, (
        f"drop_frac={result.drop_frac} outside [0.01, 0.10] — "
        f"n_in={result.n_in} n_out={result.n_out}"
    )
    assert result.xyz.dtype == np.float32
    assert result.segment.dtype == np.int32
    assert len(result.xyz) == len(result.segment)


def test_determinism() -> None:
    """P01-denoise-02 — two runs on the same fixture are bitwise equal.

    open3d's CPU implementation is deterministic for fixed inputs and fixed
    parameters; the test confirms the wrapper does not leak entropy via
    wall-clock seeds or similar.
    """
    denoise = _load_denoise()
    xyz, seg = _build_cube_plus_outliers()
    cfg = denoise.DenoiseConfig(
        method="sor", params={"nb_neighbors": 20, "std_ratio": 2.0}
    )
    r1 = denoise.denoise_cloud(xyz, seg, cfg)
    r2 = denoise.denoise_cloud(xyz, seg, cfg)
    assert np.array_equal(r1.xyz, r2.xyz)
    assert np.array_equal(r1.segment, r2.segment)
    assert r1.n_out == r2.n_out
