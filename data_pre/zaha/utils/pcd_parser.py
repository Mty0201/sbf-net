"""Streaming PCL v0.7 ASCII PCD parser for the ZAHA offline pipeline.

Exports
-------
PcdFormatError(ValueError)
parse_pcd_header(path) -> (header_dict, n_header_lines)
stream_pcd(path, chunksize=2_000_000) -> Iterator[pd.DataFrame]

Design
------
* RESEARCH §A.1 locked the header shape: every ZAHA file has exactly 11 header
  lines, fields ``x y z classification rgb``, SIZE ``4 4 4 1 4``, TYPE
  ``F F F U U``, HEIGHT ``1``, DATA ``ascii``.
* RESEARCH §A.2 benchmarked pandas ``read_csv`` with ``engine='c'`` and
  ``chunksize=2_000_000`` at ~1 MB/ms streaming rate + ~250 MB peak RAM on
  the 136.8 M-point sample.
* The ``rgb`` column is dropped unconditionally via ``usecols=[0,1,2,3]``
  (paper §3 — no spectral information).

CRITICAL import-order note: this module imports pandas but NOT open3d.
Downstream callers that use the open3d library MUST bring it in BEFORE
importing this module, otherwise the ptv3 env raises the libstdc++.so.6
GLIBCXX_3.4.29 mismatch. See ``data_pre/zaha/docs/README.md`` and
RESEARCH §I.5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


class PcdFormatError(ValueError):
    """Raised when a PCD file cannot be parsed by the ZAHA pipeline."""


# Header keys that parse_pcd_header expects to see before the DATA line.
_HEADER_KEYS = {
    "VERSION",
    "FIELDS",
    "SIZE",
    "TYPE",
    "COUNT",
    "WIDTH",
    "HEIGHT",
    "VIEWPOINT",
    "POINTS",
}

# Required prefixes for the first four columns (xyz + classification).
# rgb (column 5) is dropped by stream_pcd so we do not care about its TYPE.
_REQUIRED_FIELDS = ["x", "y", "z", "classification"]
_REQUIRED_SIZE_PREFIX = ["4", "4", "4", "1"]
_REQUIRED_TYPE_PREFIX = ["F", "F", "F", "U"]


def parse_pcd_header(path: Path) -> tuple[dict, int]:
    """Parse the PCL v0.7 ASCII PCD header.

    Returns
    -------
    header : dict
        Keys are the header lines (``VERSION``, ``FIELDS``, ``SIZE``, ``TYPE``,
        ``COUNT``, ``WIDTH``, ``HEIGHT``, ``VIEWPOINT``, ``POINTS``) with
        list-of-str values, plus ``data_format`` = ``'ascii'``.
    n_header_lines : int
        Number of lines consumed from the start of the file (including blank
        lines and ``#`` comments) up to and including the ``DATA`` line.

    Raises
    ------
    PcdFormatError
        If any of:
            * no DATA line found
            * ``data_format`` != ``'ascii'`` (ZAHA files are all ASCII per RESEARCH §A.1)
            * ``FIELDS`` does not start with ``['x', 'y', 'z', 'classification']``
            * ``SIZE`` first four values != ``['4', '4', '4', '1']``
            * ``TYPE`` first four values != ``['F', 'F', 'F', 'U']``
            * any ``COUNT`` value != ``'1'``
            * ``HEIGHT`` != ``['1']``
            * ``POINTS`` missing
    """
    header: dict = {}
    n_header_lines = 0
    path = Path(path)
    with open(path, "r") as fh:
        for line in fh:
            n_header_lines += 1
            stripped = line.strip().lstrip("\ufeff")
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("DATA"):
                parts = stripped.split()
                if len(parts) != 2:
                    raise PcdFormatError(
                        f"{path}: malformed DATA line: {stripped!r}"
                    )
                header["data_format"] = parts[1]
                break
            key, *val = stripped.split()
            header[key] = val
        else:
            raise PcdFormatError(f"{path}: no DATA line found")

    if header.get("data_format") != "ascii":
        raise PcdFormatError(
            f"{path}: expected DATA ASCII, got {header.get('data_format')!r}"
        )

    fields = header.get("FIELDS", [])
    if fields[: len(_REQUIRED_FIELDS)] != _REQUIRED_FIELDS:
        raise PcdFormatError(
            f"{path}: unexpected FIELDS — expected prefix {_REQUIRED_FIELDS}, "
            f"got {fields}"
        )

    size = header.get("SIZE", [])
    if size[: len(_REQUIRED_SIZE_PREFIX)] != _REQUIRED_SIZE_PREFIX:
        raise PcdFormatError(
            f"{path}: unexpected SIZE — expected prefix "
            f"{_REQUIRED_SIZE_PREFIX}, got {size}"
        )

    type_ = header.get("TYPE", [])
    if type_[: len(_REQUIRED_TYPE_PREFIX)] != _REQUIRED_TYPE_PREFIX:
        raise PcdFormatError(
            f"{path}: unexpected TYPE — expected prefix "
            f"{_REQUIRED_TYPE_PREFIX}, got {type_}"
        )

    count = header.get("COUNT", [])
    if not count or any(c != "1" for c in count):
        raise PcdFormatError(
            f"{path}: COUNT must be all '1' values, got {count}"
        )

    height = header.get("HEIGHT", [])
    if height != ["1"]:
        raise PcdFormatError(f"{path}: HEIGHT != 1, got {height}")

    if "POINTS" not in header:
        raise PcdFormatError(f"{path}: missing POINTS header line")

    return header, n_header_lines


def stream_pcd(
    path: Path, chunksize: int = 2_000_000
) -> Iterator[pd.DataFrame]:
    """Stream a ZAHA ASCII PCD file in bounded-memory chunks.

    Parameters
    ----------
    path : Path
        Path to the .pcd file.
    chunksize : int
        Number of rows per yielded DataFrame. RESEARCH §A.2 benchmarks 2 M at
        ~250 MB peak RAM and chooses it as the default.

    Yields
    ------
    pd.DataFrame
        Columns exactly ``['x', 'y', 'z', 'c']``. Dtypes:
        ``x/y/z = float64``, ``c = int32``. The ``rgb`` column is dropped
        unconditionally (paper §3 — not spectral).

    Raises
    ------
    PcdFormatError
        Propagated from ``parse_pcd_header`` if the header is invalid or the
        file is not ASCII. Also raised if the total streamed row count does
        not match the header ``POINTS`` value (catches truncated files).
    """
    path = Path(path)
    header, n_skip = parse_pcd_header(path)
    expected_n = int(header["POINTS"][0])

    reader = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=n_skip,
        header=None,
        usecols=[0, 1, 2, 3],
        names=["x", "y", "z", "c"],
        dtype={
            "x": "float64",
            "y": "float64",
            "z": "float64",
            "c": "int32",
        },
        engine="c",
        chunksize=chunksize,
    )

    total = 0
    for chunk in reader:
        total += len(chunk)
        yield chunk

    if total != expected_n:
        raise PcdFormatError(
            f"{path}: expected {expected_n} points per header, streamed {total}"
        )
