"""Tests for the NPY output layout — DS-ZAHA-P1-06.

Turned GREEN by Plan 04 Task 1 — layout.py + manifest.py land the writer and
schema. Behaviour references: CONTEXT.md D-02 (segment.npy values in ``[0, 15]``
after remap), D-20 (manifest), D-21 (sanity gates), D-22 (hard-fail),
VALIDATION.md rows ``P01-layout-01..02``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _load_layout():
    return pytest.importorskip("data_pre.zaha.utils.layout")


def _fake_chunk(n: int = 100):
    rng = np.random.default_rng(0)
    coord = rng.standard_normal((n, 3)).astype(np.float32)
    segment = rng.integers(0, 16, size=n, dtype=np.int32)
    normal_raw = rng.standard_normal((n, 3)).astype(np.float32)
    norms = np.linalg.norm(normal_raw, axis=1, keepdims=True)
    normal = (normal_raw / norms).astype(np.float32)
    return coord, segment, normal


def test_file_structure(tmp_path) -> None:
    """P01-layout-01 — ``<root>/<split>/<sample>__c000/{coord,segment,normal}.npy``.

    After a write on a synthetic 100-point chunk, all three NPY files exist,
    reload with the expected ``(N,3)/(N,)/(N,3)`` shapes and
    ``float32/int32/float32`` dtypes.
    """
    layout = _load_layout()
    coord, segment, normal = _fake_chunk(100)
    out_dir = tmp_path / "DEBY_LOD2_TEST__c000"
    stats = layout.write_chunk_npys(out_dir, coord, segment, normal)
    for fn in ("coord.npy", "segment.npy", "normal.npy"):
        assert (out_dir / fn).exists(), f"{fn} missing from {out_dir}"
    c = np.load(out_dir / "coord.npy")
    s = np.load(out_dir / "segment.npy")
    nm = np.load(out_dir / "normal.npy")
    assert c.shape == (100, 3) and c.dtype == np.float32
    assert s.shape == (100,) and s.dtype == np.int32
    assert nm.shape == (100, 3) and nm.dtype == np.float32
    assert stats["point_count"] == 100
    # sha256 fields are opaque but must be strings of the expected hex length.
    for key in ("coord_sha256", "segment_sha256", "normal_sha256"):
        assert isinstance(stats[key], str)
        assert len(stats[key]) == 64


def test_segment_range(tmp_path) -> None:
    """P01-layout-02 — ``segment.npy`` values strictly in ``[0, 15]`` (D-02).

    The happy path (values ``[0, 7, 15, 3]``) writes cleanly; a bad value of
    ``16`` raises ``ValueError`` — D-22 hard-fail.
    """
    layout = _load_layout()
    coord, _, normal = _fake_chunk(4)
    good_seg = np.array([0, 7, 15, 3], dtype=np.int32)
    stats = layout.write_chunk_npys(tmp_path / "good__c000", coord, good_seg, normal)
    assert stats["segment_min"] == 0 and stats["segment_max"] == 15
    # Reload and double-check the invariant persists on disk.
    s = np.load(tmp_path / "good__c000" / "segment.npy")
    assert int(s.min()) >= 0 and int(s.max()) <= 15
    # Bad value 16 must hard-fail.
    bad_seg = np.array([0, 7, 16, 3], dtype=np.int32)
    with pytest.raises(ValueError, match=r"(?i)segment"):
        layout.write_chunk_npys(tmp_path / "bad__c000", coord, bad_seg, normal)
