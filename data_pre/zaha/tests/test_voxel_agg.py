"""Tests for ``data_pre.zaha.utils.voxel_agg`` — DS-ZAHA-P1-02 + VOID drop order.

RED stubs until Plan 02 lands the hash-partitioned external-sort voxel
aggregator (see ``01-RESEARCH.md §B.2`` and CONTEXT D-14/D-15 supersession).

Target impl module: ``data_pre.zaha.utils.voxel_agg`` (created in Plan 02).
Behaviour references: CONTEXT.md D-01 (VOID drop order), D-02 (remap 1..16 →
0..15), D-16 (smallest-class-ID tie-break), RESEARCH.md §B (aggregation
strategy), VALIDATION.md rows ``P01-voxel-01..04``.
"""

from __future__ import annotations

import pytest


def test_determinism(synthetic_pcd_fixture) -> None:
    """P01-voxel-01 — two aggregation runs on the same fixture are bitwise equal.

    Expected behaviour (Plan 02 task 2):
        ``voxel_agg(points)`` is deterministic: ``out1 == out2`` element-wise
        for ``out1 = voxel_agg(points); out2 = voxel_agg(points)``.
    """
    pytest.fail("not yet implemented — PLAN 02 task 2")


def test_tie_break() -> None:
    """P01-voxel-02 — majority-vote tie broken by smallest raw class ID (D-16).

    Expected behaviour (Plan 02 task 2):
        hand-built voxel with class histogram ``{3: 2, 5: 2}`` resolves to
        winner ``3`` (the smallest id wins). Ties on raw space, before the
        1..16 → 0..15 remap.
    """
    pytest.fail("not yet implemented — PLAN 02 task 2")


def test_hash_partition() -> None:
    """P01-voxel-03 — two voxel keys with known values land in the same bin.

    Expected behaviour (Plan 02 task 2):
        with ``K = 16`` disk bins and a fixed partition hash, two distinct
        voxel keys that hash to the same ``key % K`` residue must be routed
        to the same bin. Tests the hash-partitioned external-sort plan
        recommended by RESEARCH §B.2.
    """
    pytest.fail("not yet implemented — PLAN 02 task 2")


def test_void_drop_order() -> None:
    """P01-voxel-04 — VOID drop ordering matches CONTEXT D-01 / D-02 / D-03.

    Expected behaviour (Plan 02 task 2):
        voxel with mixture ``{0: 2, 1: 1}`` → majority VOID → DROP (no output).
        voxel with mixture ``{0: 1, 1: 2}`` → majority Wall → KEEP and remap
        class 1 → 0 (remapped LoFG3 space). Downsample runs on the full cloud
        first (majority vote sees VOID honestly); any voxel whose winning label
        is 0 is removed post-vote.
    """
    pytest.fail("not yet implemented — PLAN 02 task 2")
