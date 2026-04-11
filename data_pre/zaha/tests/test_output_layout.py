"""Tests for the NPY output layout — DS-ZAHA-P1-06.

RED stubs until Plan 04 runs the orchestrator script and produces the chunked
NPY layout.

Target impl: ``data_pre.zaha.scripts.build_zaha_chunks`` (Plan 04).
Behaviour references: CONTEXT.md D-02 (segment.npy values in ``[0, 15]``
after remap), D-20 (single entry-point script + manifest), D-21 (sanity gates
include segment range + shape match), VALIDATION.md rows ``P01-layout-01..02``.
"""

from __future__ import annotations

import pytest


def test_file_structure() -> None:
    """P01-layout-01 — ``<root>/<split>/<sample>__c000/{coord,segment,normal}.npy``.

    Expected behaviour (Plan 04 task 1):
        after a build run on a mini fixture, the expected directory tree
        ``<root>/<split>/<sample>__c000/{coord,segment,normal}.npy`` exists
        and each file loads as a non-empty numpy array. Skipped until Plan 04
        has executed once on a mini fixture.
    """
    pytest.fail("not yet implemented — PLAN 04 task 1")


def test_segment_range() -> None:
    """P01-layout-02 — ``segment.npy`` values strictly in ``[0, 15]`` (D-02).

    Expected behaviour (Plan 04 task 1):
        every written ``segment.npy`` satisfies
        ``seg.min() >= 0 and seg.max() <= 15``, per CONTEXT D-02. VOID (raw
        ID 0) was dropped at Phase 1's voxel step; the remaining raw IDs
        1..16 were remapped to 0..15. No value ``< 0`` or ``>= 16`` may
        appear — CONTEXT D-21 sanity gate (d).
    """
    pytest.fail("not yet implemented — PLAN 04 task 1")
