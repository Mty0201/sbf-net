"""Tests for ``data_pre.zaha.utils.pcd_parser`` — DS-ZAHA-P1-01.

RED stubs until Plan 02 lands the streaming PCD parser. Each test body is a
single ``pytest.fail("not yet implemented — PLAN NN task K")`` so the Nyquist
feedback loop reports RED → GREEN per task from Wave 0 onward.

Target impl module: ``data_pre.zaha.utils.pcd_parser`` (created in Plan 02).
Behaviour references: ``01-RESEARCH.md §A.1`` (header format) and
``01-VALIDATION.md`` per-task map rows ``P01-parser-01..04``.
"""

from __future__ import annotations

import pytest


def test_header_roundtrip(synthetic_pcd_fixture) -> None:
    """P01-parser-01 — parses PCL v0.7 header, POINTS count matches FIELDS arity.

    Expected behaviour (Plan 02 task 1):
        parse synthetic_pcd_fixture's 11-line header, assert the ``POINTS``
        line value equals the number of data rows in the body.
    """
    pytest.fail("not yet implemented — PLAN 02 task 1")


def test_count_match(synthetic_pcd_fixture) -> None:
    """P01-parser-02 — streaming parse yields ``POINTS`` rows exactly.

    Expected behaviour (Plan 02 task 1):
        stream the synthetic PCD, assert ``sum(1 for _ in rows) == header_POINTS``.
    """
    pytest.fail("not yet implemented — PLAN 02 task 1")


def test_binary_rejected(synthetic_pcd_binary_fixture) -> None:
    """P01-parser-03 — ``DATA binary`` PCD is rejected with an ASCII error.

    Expected behaviour (Plan 02 task 1):
        parsing a ``DATA binary`` PCD raises ``AssertionError`` or ``ValueError``
        whose message contains the literal string ``"ASCII"``.
    """
    pytest.fail("not yet implemented — PLAN 02 task 1")


def test_rgb_dropped(synthetic_pcd_fixture) -> None:
    """P01-parser-04 — parser drops the ``rgb`` column (paper §3, no spectral).

    Expected behaviour (Plan 02 task 1):
        streaming parse returns rows with exactly columns ``['x', 'y', 'z', 'c']``
        (``c`` is the int classification); ``rgb`` must not appear.
    """
    pytest.fail("not yet implemented — PLAN 02 task 1")


def test_header_field_types(synthetic_pcd_fixture) -> None:
    """P01-parser-05 — PCL header FIELDS/SIZE/TYPE lines match ZAHA format.

    Expected behaviour (Plan 02 task 1):
        the parser verifies
        ``FIELDS x y z classification rgb``, ``SIZE 4 4 4 1 4``,
        ``TYPE F F F U U`` and rejects any file whose header deviates.
        Closes the DS-ZAHA-P1-01 acceptance gate "parser must handle the
        ASCII PCD header" from REQUIREMENTS.md.
    """
    pytest.fail("not yet implemented — PLAN 02 task 1")
