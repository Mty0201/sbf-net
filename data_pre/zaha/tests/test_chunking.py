"""Tests for ``data_pre.zaha.utils.chunking`` — DS-ZAHA-P1-04.

RED stubs until Plan 03 lands the deterministic axis-aligned box-grid chunker
with ≥2 m overlap and ≤ 0.6 M pts/chunk budget.

Target impl module: ``data_pre.zaha.utils.chunking`` (created in Plan 03).
Behaviour references: CONTEXT.md D-06 (axis-aligned box grid, row-major order,
deterministic origin = bbox.min), D-07 (≤ 0.6 M pts/chunk), D-08 (fixed XY
tile size across dataset), D-10 (chunk directory naming), VALIDATION.md rows
``P01-chunk-01..03``.
"""

from __future__ import annotations

import pytest


def test_deterministic() -> None:
    """P01-chunk-01 — same building → same chunk IDs, bboxes, ordering (D-06).

    Expected behaviour (Plan 03 task 3):
        two invocations on the same input produce identical chunk_idx lists,
        identical bbox tuples, and row-major (x-outer, y-inner) ordering —
        the deterministic audit hook behind the manifest's ``commit_hash`` field.
    """
    pytest.fail("not yet implemented — PLAN 03 task 3")


def test_budget() -> None:
    """P01-chunk-02 — no chunk exceeds the 600,000-point budget (D-07).

    Expected behaviour (Plan 03 task 3):
        on a synthetic building cloud whose post-0.02 density is within the
        worst-case ZAHA envelope, every produced chunk must satisfy
        ``len(coord) <= 600_000``. Hard cap from CONTEXT D-07 (tighter than
        ROADMAP's 1.0 M initial target).
    """
    pytest.fail("not yet implemented — PLAN 03 task 3")


def test_overlap() -> None:
    """P01-chunk-03 — adjacent chunks share ≥ 2 m overlap in xy (D-06).

    Expected behaviour (Plan 03 task 3):
        tile_stride = tile_size − overlap, with overlap ≥ 2.0 m in both x and
        y. Overlap regions may carry the same point in two chunks — Phase 1
        does NOT dedup (D-09). Training-time sphere-crop doesn't care; eval
        dedup is a Phase 1b/4 concern.
    """
    pytest.fail("not yet implemented — PLAN 03 task 3")
