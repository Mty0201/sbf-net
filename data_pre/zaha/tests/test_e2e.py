"""End-to-end tests for the ZAHA orchestrator — DS-ZAHA-P1-07.

RED stubs until Plan 04 lands the ``build_zaha_chunks.py`` entry point +
``manifest.json`` + golden file.

Target impl: ``data_pre.zaha.scripts.build_zaha_chunks`` (Plan 04).
Behaviour references: CONTEXT.md D-20 (manifest schema), D-21 (sanity gates),
D-22 (hard-failure policy), RESEARCH.md §H (orchestration), VALIDATION.md rows
``P01-e2e-01..04`` + ``P01-e2e-golden``.
"""

from __future__ import annotations

import pytest


def test_smallest_sample() -> None:
    """P01-e2e-01 — full pipeline runs on the smallest sample without error.

    Expected behaviour (Plan 04 task 2):
        ``python data_pre/zaha/scripts/build_zaha_chunks.py --samples
        DEBY_LOD2_4907179 --output /tmp/zaha_e2e_smoke`` exits with code 0
        and writes at least one chunk. Skipped until Plan 04 has run once
        on a real sample.
    """
    pytest.fail("not yet implemented — PLAN 04 task 2")


def test_manifest_schema() -> None:
    """P01-e2e-02 — ``manifest.json`` has required top-level keys (D-20).

    Expected behaviour (Plan 04 task 2):
        loads ``/tmp/zaha_e2e_smoke/manifest.json``, asserts dict contains
        at minimum the CONTEXT D-20 keys: ``sample, chunk_idx, bbox,
        point_count, denoising_method, denoising_params, normal_method,
        normal_params, source_pcd, commit_hash`` per chunk entry.
    """
    pytest.fail("not yet implemented — PLAN 04 task 2")


def test_sanity_gates() -> None:
    """P01-e2e-03 — class histogram drift computation runs without exception (D-21).

    Expected behaviour (Plan 04 task 2):
        loads the manifest and runs the CONTEXT D-21 sanity gate (a): total
        per-class point histogram across chunks roughly matches raw-PCD
        class histogram within VOID-drop + downsampling variance. The test
        asserts the computation completes without raising; the actual
        threshold is a Plan 04 decision.
    """
    pytest.fail("not yet implemented — PLAN 04 task 2")


def test_determinism() -> None:
    """P01-e2e-04 — two pipeline runs on the same sample are bitwise equal.

    Expected behaviour (Plan 04 task 2):
        run the full pipeline twice on the same input sample; all produced
        ``coord.npy``, ``segment.npy``, ``normal.npy`` files across all
        chunks must be byte-equal. This is the deterministic audit hook
        behind the manifest's ``commit_hash`` field (CONTEXT D-20).
    """
    pytest.fail("not yet implemented — PLAN 04 task 2")


def test_golden() -> None:
    """P01-e2e-golden — outputs match frozen SHA256 hashes for regression.

    Expected behaviour (Plan 04 task 2):
        loads ``data_pre/zaha/tests/golden/DEBY_LOD2_4907179__goldens.json``,
        re-runs the pipeline, asserts sha256 of each produced NPY file
        matches the frozen value. Skipped until the golden file exists
        (generated after first green E2E run, per VALIDATION.md Wave 0 list).
    """
    pytest.fail("not yet implemented — PLAN 04 task 2")


def test_hard_failure_on_missing_sample() -> None:
    """P01-e2e-05 — orchestrator hard-fails if a readme sample is missing (D-22).

    Expected behaviour (Plan 04 task 2):
        when the readme manifest declares a sample that is not present on
        disk, the orchestrator exits with a non-zero status and an explicit
        error message. CONTEXT.md D-22 hard-failure policy: "any raw PCD
        that crashes the parser, any chunk that violates a sanity check,
        any file in the readme manifest that cannot be produced → Phase 1
        does not complete. No skip-on-error."
    """
    pytest.fail("not yet implemented — PLAN 04 task 2")
