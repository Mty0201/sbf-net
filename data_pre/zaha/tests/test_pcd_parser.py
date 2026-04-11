"""Tests for ``data_pre.zaha.utils.pcd_parser`` — DS-ZAHA-P1-01.

Plan 02 Task 1 turned these GREEN. Each test exercises one behaviour from
01-02-PLAN.md ``<behavior>`` block.

Target impl module: ``data_pre.zaha.utils.pcd_parser`` (landed in Plan 02).
Behaviour references: ``01-RESEARCH.md §A.1`` (header format) and
``01-VALIDATION.md`` per-task map rows ``P01-parser-01..05``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_pre.zaha.utils.pcd_parser import (
    PcdFormatError,
    parse_pcd_header,
    stream_pcd,
)


def test_header_roundtrip(synthetic_pcd_fixture) -> None:
    """P01-parser-01 — parses PCL v0.7 header, POINTS count matches FIELDS arity.

    The synthetic fixture writes an 11-line ZAHA-shape header with 1000 data
    rows. ``parse_pcd_header`` must return that header dict and the exact
    number of header lines consumed (11).
    """
    header, n_header_lines = parse_pcd_header(synthetic_pcd_fixture)
    assert header["FIELDS"] == ["x", "y", "z", "classification", "rgb"]
    assert header["POINTS"] == ["1000"]
    assert header["data_format"] == "ascii"
    # Synthetic fixture emits exactly 11 header lines (VERSION..DATA inclusive).
    assert n_header_lines == 11
    # POINTS value must agree with WIDTH for a single-row cloud.
    assert header["WIDTH"] == ["1000"]


def test_count_match(synthetic_pcd_fixture) -> None:
    """P01-parser-02 — streaming parse yields ``POINTS`` rows exactly.

    Iterating ``stream_pcd`` with a small ``chunksize`` (200) produces 5
    chunks of 200 rows each, totalling 1000.
    """
    total = 0
    chunk_count = 0
    for chunk in stream_pcd(synthetic_pcd_fixture, chunksize=200):
        total += len(chunk)
        chunk_count += 1
    assert total == 1000
    # 1000 / 200 = 5 chunks.
    assert chunk_count == 5


def test_binary_rejected(synthetic_pcd_binary_fixture) -> None:
    """P01-parser-03 — ``DATA binary`` PCD is rejected with a clear ASCII error.

    The parser must refuse to consume ``DATA binary`` files and raise
    ``PcdFormatError`` whose message mentions the expected ASCII format.
    """
    with pytest.raises(PcdFormatError) as excinfo:
        # parse_pcd_header is the gate — stream_pcd delegates to it first.
        parse_pcd_header(synthetic_pcd_binary_fixture)
    # Error message must mention ASCII so callers know why they were rejected.
    assert "ASCII" in str(excinfo.value) or "ascii" in str(excinfo.value)

    # stream_pcd must also propagate the rejection.
    with pytest.raises(PcdFormatError):
        for _ in stream_pcd(synthetic_pcd_binary_fixture):
            pass


def test_rgb_dropped(synthetic_pcd_fixture) -> None:
    """P01-parser-04 — parser drops the ``rgb`` column (paper §3, no spectral).

    ``stream_pcd`` must emit DataFrames with columns exactly ``['x','y','z','c']``
    and dtypes ``float64/float64/float64/int32``. Reading the fixture that
    wrote ``rgb=13033652`` must not surface that column.
    """
    first = next(iter(stream_pcd(synthetic_pcd_fixture, chunksize=200)))
    assert list(first.columns) == ["x", "y", "z", "c"]
    assert first["x"].dtype == np.float64
    assert first["y"].dtype == np.float64
    assert first["z"].dtype == np.float64
    assert first["c"].dtype == np.int32
    # The fixture's rgb filler must NOT appear anywhere in the emitted frame.
    assert "rgb" not in first.columns
    # Classification range in the fixture is ``i % 17`` → [0, 16].
    assert first["c"].min() >= 0
    assert first["c"].max() <= 16


def test_header_field_types(synthetic_pcd_fixture, tmp_path: Path) -> None:
    """P01-parser-05 — PCL header FIELDS/SIZE/TYPE lines match ZAHA format.

    Positive path: the valid fixture must expose FIELDS, SIZE and TYPE lines
    that match the ZAHA release format (``FIELDS x y z classification rgb``,
    ``SIZE 4 4 4 1 4``, ``TYPE F F F U U``).

    Negative path: a malformed PCD whose TYPE column lies about the xyz
    encoding (``I I I U U`` instead of ``F F F U U``) must be rejected with
    ``PcdFormatError``. Closes the DS-ZAHA-P1-01 acceptance gate "parser must
    handle the ASCII PCD header" from REQUIREMENTS.md.
    """
    header, _ = parse_pcd_header(synthetic_pcd_fixture)
    assert header["FIELDS"] == ["x", "y", "z", "classification", "rgb"]
    assert header["SIZE"] == ["4", "4", "4", "1", "4"]
    assert header["TYPE"] == ["F", "F", "F", "U", "U"]
    # COUNT is one per field, all ones.
    assert header["COUNT"] == ["1", "1", "1", "1", "1"]
    assert header["HEIGHT"] == ["1"]

    # Negative path: build a PCD whose xyz TYPE is wrong.
    bad = tmp_path / "bad_type.pcd"
    bad.write_text(
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z classification rgb\n"
        "SIZE 4 4 4 1 4\n"
        "TYPE I I I U U\n"  # wrong — xyz must be F
        "COUNT 1 1 1 1 1\n"
        "WIDTH 1\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS 1\n"
        "DATA ascii\n"
        "0.0 0.0 0.0 1 0\n"
    )
    with pytest.raises(PcdFormatError):
        parse_pcd_header(bad)

    # A PCD with wrong FIELDS is also rejected.
    wrong_fields = tmp_path / "bad_fields.pcd"
    wrong_fields.write_text(
        "VERSION 0.7\n"
        "FIELDS x y z intensity rgb\n"  # wrong — must be 'classification'
        "SIZE 4 4 4 1 4\n"
        "TYPE F F F U U\n"
        "COUNT 1 1 1 1 1\n"
        "WIDTH 1\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS 1\n"
        "DATA ascii\n"
        "0.0 0.0 0.0 1 0\n"
    )
    with pytest.raises(PcdFormatError):
        parse_pcd_header(wrong_fields)
