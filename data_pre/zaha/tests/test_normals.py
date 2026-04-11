"""Tests for ``data_pre.zaha.utils.normals`` — DS-ZAHA-P1-05.

RED stubs until Plan 03 lands the adaptive-radius PCA normal estimator.

Target impl module: ``data_pre.zaha.utils.normals`` (created in Plan 03).
Behaviour references: CONTEXT.md D-17 (adaptive-radius PCA, k ≈ 30 default),
D-18 (acceptance bar: unit-length float32 (N, 3), no NaN; "thinning the
wall" aspirational, not blocking), D-19 (per-chunk, not whole-building),
VALIDATION.md rows ``P01-normal-01..03``.
"""

from __future__ import annotations

import pytest


def test_unit_length() -> None:
    """P01-normal-01 — all estimated normals have ``‖n‖ > 0.99`` (D-18).

    Expected behaviour (Plan 03 task 4):
        on a synthetic flat-plane point cloud, every estimated normal vector
        has magnitude greater than 0.99 (effectively unit length up to
        float32 precision). No NaN, no zero-length, no sign flip.
    """
    pytest.fail("not yet implemented — PLAN 03 task 4")


def test_no_nan() -> None:
    """P01-normal-02 — no NaN or inf in the output normals array (D-18).

    Expected behaviour (Plan 03 task 4):
        ``np.isfinite(normals).all() == True`` on the synthetic fixture. The
        acceptance bar in CONTEXT D-18 is explicit: unit-length float32
        ``(N, 3)`` with correct shape and no NaN.
    """
    pytest.fail("not yet implemented — PLAN 03 task 4")


def test_degenerate_fallback() -> None:
    """P01-normal-03 — degenerate 5-point collinear input does not produce NaN.

    Expected behaviour (Plan 03 task 4):
        a 5-point collinear chunk (PCA covariance is rank-deficient) must
        either return a valid fallback normal (e.g. any unit vector in the
        null space) or raise an explicit error. It MUST NOT silently emit
        NaN values — downstream D-21 sanity checks reject NaN on hard-fail.
    """
    pytest.fail("not yet implemented — PLAN 03 task 4")
