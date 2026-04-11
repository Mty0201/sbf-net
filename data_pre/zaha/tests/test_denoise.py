"""Tests for ``data_pre.zaha.utils.denoise`` — DS-ZAHA-P1-03.

RED stubs until Plan 03 lands the denoising method selection (3-candidate
research gate: SOR + radius outlier + {bilateral, MLS, RANSAC residual}).

Target impl module: ``data_pre.zaha.utils.denoise`` (created in Plan 03).
Behaviour references: CONTEXT.md D-11 (3 candidates), D-12 (≤10% drop cap,
visible stripe reduction), D-13 (pre-chunking on whole-building cloud),
VALIDATION.md rows ``P01-denoise-01..02``.
"""

from __future__ import annotations

import pytest


def test_drop_cap() -> None:
    """P01-denoise-01 — denoiser drops between 1% and 10% of points (D-12).

    Expected behaviour (Plan 03 task 2):
        on a synthetic 1000-point cube with 100 outliers, the chosen denoiser
        must remove between 1% and 10% of points (hard cap from CONTEXT D-12).
        ≤ 10% is the critical bound — the acceptance rubric says Phase 1 does
        not ship if any sample loses more than 10%.
    """
    pytest.fail("not yet implemented — PLAN 03 task 2")


def test_determinism() -> None:
    """P01-denoise-02 — two runs on the same fixture are bitwise equal.

    Expected behaviour (Plan 03 task 2):
        ``denoise(pts)`` is deterministic: two sequential runs on the same
        input produce bitwise-equal output arrays (same indices kept, same
        coordinate bytes).
    """
    pytest.fail("not yet implemented — PLAN 03 task 2")
