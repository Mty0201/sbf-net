"""Tests for ``lofg3_to_lofg2.yaml`` — DS-ZAHA-P1-06 + D-04 OuterCeilingSurface.

GREEN in Plan 04 Task 1 — asserts schema version, entry count, and D-04 lock.

Target artifact: ``<phase_dir>/lofg3_to_lofg2.yaml`` (authored in Plan 01 Task 3).
Behaviour references: CONTEXT.md D-04 (OuterCeilingSurface → other_el LOCKED),
D-05 (YAML keys are remapped 0..15 space, NOT raw 1..16 space), RESEARCH.md
§G.3 (schema version + field list), VALIDATION.md row ``P01-yaml-01``.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def test_schema(phase_dir: Path) -> None:
    """P01-yaml-01 — schema is valid, 16 entries, D-04 OuterCeilingSurface locked.

    ``yaml.safe_load(phase_dir / 'lofg3_to_lofg2.yaml')`` returns a dict with
    keys
    ``{schema_version, num_lofg3_classes, num_lofg2_classes, lofg2_buckets, lofg3_to_lofg2, sources}``.
    ``lofg3_to_lofg2`` is a dict with integer keys in ``[0, 15]``. Entry ``13``
    maps to bucket ``4`` (OuterCeilingSurface → other_el, LOCKED by
    CONTEXT D-04). The ``sources[13]`` entry must reference ``D-04``.
    """
    yaml_path = phase_dir / "lofg3_to_lofg2.yaml"
    assert yaml_path.exists(), f"{yaml_path} missing — Plan 01 Task 3 not run?"
    d = yaml.safe_load(yaml_path.read_text())
    # Required top-level keys per RESEARCH §G.3.
    for key in (
        "schema_version",
        "num_lofg3_classes",
        "num_lofg2_classes",
        "lofg2_buckets",
        "lofg3_to_lofg2",
        "sources",
    ):
        assert key in d, f"yaml missing required key {key!r}"
    assert d["schema_version"] == 1
    assert d["num_lofg3_classes"] == 16
    assert d["num_lofg2_classes"] == 5
    assert len(d["lofg2_buckets"]) == 5
    # D-05: keys are in the remapped [0, 15] space.
    assert len(d["lofg3_to_lofg2"]) == 16
    assert set(d["lofg3_to_lofg2"].keys()) == set(range(16))
    # D-04 lock: OuterCeilingSurface (remapped index 13) → other_el (bucket 4).
    assert d["lofg3_to_lofg2"][13] == 4
    # Source annotation for entry 13 must reference D-04 so auditors can trace.
    assert "D-04" in d["sources"][13], (
        f"sources[13] must reference D-04, got: {d['sources'][13]!r}"
    )
