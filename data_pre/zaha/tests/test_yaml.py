"""Tests for ``lofg3_to_lofg2.yaml`` — DS-ZAHA-P1-06 + D-04 OuterCeilingSurface.

RED stub until the structural assertions are wired in Plan 02/03.

Target artifact: ``<phase_dir>/lofg3_to_lofg2.yaml`` (authored in Plan 01 Task 3,
this plan).
Behaviour references: CONTEXT.md D-04 (OuterCeilingSurface → other_el LOCKED),
D-05 (YAML keys are remapped 0..15 space, NOT raw 1..16 space), RESEARCH.md
§G.3 (schema version + field list), VALIDATION.md row ``P01-yaml-01``.
"""

from __future__ import annotations

import pytest


def test_schema(phase_dir) -> None:
    """P01-yaml-01 — schema is valid, 16 entries, D-04 OuterCeilingSurface locked.

    Expected behaviour (Plan 02+ task — test body):
        ``yaml.safe_load(phase_dir / 'lofg3_to_lofg2.yaml')`` returns a dict
        with keys
        ``{schema_version, num_lofg3_classes, num_lofg2_classes, lofg2_buckets, lofg3_to_lofg2, sources}``.
        ``lofg3_to_lofg2`` is a dict with integer keys in ``[0, 15]``.
        Entry ``13`` maps to bucket ``4`` (OuterCeilingSurface → other_el,
        LOCKED by CONTEXT D-04). 11 HIGH-confidence entries cite paper Fig. 3;
        4 LOW-confidence entries are annotated ``ASSUMED`` per Plan 01 Task 3.
    """
    pytest.fail("not yet implemented — PLAN 02 task test body")
