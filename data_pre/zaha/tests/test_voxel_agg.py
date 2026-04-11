"""Tests for ``data_pre.zaha.utils.voxel_agg`` — DS-ZAHA-P1-02 + VOID drop order.

Plan 02 Task 2 turned these GREEN. Each test exercises one behaviour from the
``<behavior>`` block of 01-02-PLAN.md.

Target impl module: ``data_pre.zaha.utils.voxel_agg``.
Behaviour references:
    CONTEXT.md D-01  — VOID drop ordering
    CONTEXT.md D-02  — remap raw 1..16 → remapped 0..15
    CONTEXT.md D-16  — smallest-class-ID tie-break + stable sort determinism
    RESEARCH.md §B.1 — numpy sort aggregate kernel
    RESEARCH.md §B.2 — hash-partitioned external sort
    VALIDATION.md rows P01-voxel-01..04
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_pre.zaha.utils.voxel_agg import (
    GRID,
    VoxelAggregateResult,
    VoxelBatch,
    _compute_voxel_keys,
    pack_voxel_keys,
    stream_voxel_aggregate,
    voxel_aggregate_batch,
)


# ---------------------------------------------------------------------------
# Local helper: write a synthetic ASCII PCD matching the ZAHA header shape
# ---------------------------------------------------------------------------

def _write_minimal_pcd(
    path: Path, points: list[tuple[float, float, float, int]]
) -> Path:
    """Write a minimal ZAHA-shape ASCII PCD with the given (x, y, z, cls) tuples.

    Mirrors the header shape of ``conftest.py::_write_pcd`` but keeps the row
    set explicit so the test can reason about the exact voxelization and
    VOID-drop outcomes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z classification rgb\n"
        "SIZE 4 4 4 1 4\n"
        "TYPE F F F U U\n"
        "COUNT 1 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    with path.open("w") as fh:
        fh.write(header)
        for x, y, z, cls in points:
            fh.write(f"{x:.6f} {y:.6f} {z:.6f} {cls} 0\n")
    return path


# ---------------------------------------------------------------------------
# P01-voxel-01 — determinism (two runs are bitwise equal)
# ---------------------------------------------------------------------------

def test_determinism() -> None:
    """P01-voxel-01 — two aggregation runs on the same input are bitwise equal.

    Deterministic voxel aggregation is guaranteed by (a) stable sort on packed
    keys, (b) np.add.reduceat iterating sorted order, (c) np.argmax returning
    the first max index. Runs the in-RAM kernel twice on identical input and
    asserts all three output arrays are bitwise equal (``np.array_equal``,
    not approx).
    """
    rng = np.random.default_rng(0)
    xyz = rng.random((100, 3)).astype(np.float64)
    cls = rng.integers(0, 17, size=100).astype(np.int32)

    out1 = voxel_aggregate_batch(xyz, cls)
    out2 = voxel_aggregate_batch(xyz, cls)

    assert np.array_equal(out1.packed_keys, out2.packed_keys)
    assert np.array_equal(out1.centroid_xyz, out2.centroid_xyz)
    assert np.array_equal(out1.winner_class, out2.winner_class)
    assert np.array_equal(out1.counts, out2.counts)
    assert np.array_equal(out1.hist, out2.hist)

    # Run once more on a COPY of the inputs — must still bitwise-match.
    out3 = voxel_aggregate_batch(xyz.copy(), cls.copy())
    assert np.array_equal(out1.packed_keys, out3.packed_keys)
    assert np.array_equal(out1.centroid_xyz, out3.centroid_xyz)
    assert np.array_equal(out1.winner_class, out3.winner_class)


# ---------------------------------------------------------------------------
# P01-voxel-02 — tie-break: smallest raw class ID wins (D-16)
# ---------------------------------------------------------------------------

def test_tie_break() -> None:
    """P01-voxel-02 — majority-vote tie broken by smallest raw class ID (D-16).

    A voxel with one cls=3 point and one cls=5 point has a tie histogram
    ``{3: 1, 5: 1}``. ``np.argmax`` returns the first max → index 3 → winner 3.
    This check runs on raw-space ids before the D-02 remap.
    """
    # Two points at the same voxel (both in voxel (0,0,0) at GRID=0.02).
    xyz = np.array(
        [
            [0.005, 0.005, 0.005],
            [0.015, 0.015, 0.015],
        ],
        dtype=np.float64,
    )
    cls = np.array([3, 5], dtype=np.int32)
    result = voxel_aggregate_batch(xyz, cls)

    # One voxel only.
    assert len(result.packed_keys) == 1
    # Tie → smallest raw class wins.
    assert result.winner_class[0] == 3
    # Histogram row confirms the tie structure.
    assert int(result.hist[0, 3]) == 1
    assert int(result.hist[0, 5]) == 1

    # And the reverse order ALSO gives winner 3 (order-independent).
    cls_rev = np.array([5, 3], dtype=np.int32)
    result_rev = voxel_aggregate_batch(xyz, cls_rev)
    assert result_rev.winner_class[0] == 3


# ---------------------------------------------------------------------------
# P01-voxel-03 — hash partition routes equal keys to the same bin
# ---------------------------------------------------------------------------

def test_hash_partition() -> None:
    """P01-voxel-03 — two points at the same voxel land in the same hash bin.

    The external-sort partition step uses ``key % K`` so equal keys always
    route to the same bin. Verifies the equal-key → equal-bin guarantee the
    RESEARCH §B.2 external-sort plan relies on.
    """
    K = 16
    # Two points at the same voxel.
    xyz = np.array(
        [
            [0.001, 0.002, 0.003],
            [0.019, 0.018, 0.017],
        ],
        dtype=np.float64,
    )
    keys = _compute_voxel_keys(xyz)
    assert keys[0] == keys[1]
    bin_a = int(keys[0] % np.uint64(K))
    bin_b = int(keys[1] % np.uint64(K))
    assert bin_a == bin_b
    assert 0 <= bin_a < K

    # Sanity: a point in a distinct voxel lands in a well-defined bin.
    xyz2 = np.array([[10.005, 10.005, 10.005]], dtype=np.float64)
    k2 = _compute_voxel_keys(xyz2)
    bin_c = int(k2[0] % np.uint64(K))
    assert 0 <= bin_c < K


# ---------------------------------------------------------------------------
# P01-voxel-04 — VOID drop order (D-01 / D-02 / D-03)
# ---------------------------------------------------------------------------

def test_void_drop_order(tmp_path: Path) -> None:
    """P01-voxel-04 — VOID drop ordering matches CONTEXT D-01 / D-02 / D-03.

    Input cloud:
        voxel A (~origin)  — 2 points cls=0 (VOID), 1 point cls=1 (Wall)
                             → majority VOID → DROPPED (D-01)
        voxel B (~(1,1,1)) — 1 point cls=0 (VOID), 2 points cls=1 (Wall)
                             → majority Wall → KEPT + remapped to 0 (D-02)

    After ``stream_voxel_aggregate`` we expect:
        n_voxels_pre_void_drop  == 2
        n_voxels_post_void_drop == 1
        segment                 == [0]  (remapped Wall)
    """
    # Voxel A: three points in the same voxel at origin, majority VOID.
    # Voxel B: three points in a distinct voxel at (1,1,1), majority Wall.
    pcd_points = [
        # voxel A — cls=[0, 0, 1] → majority VOID
        (0.001, 0.002, 0.003, 0),
        (0.004, 0.005, 0.006, 0),
        (0.007, 0.008, 0.009, 1),
        # voxel B — cls=[0, 1, 1] → majority Wall
        (1.001, 1.002, 1.003, 0),
        (1.004, 1.005, 1.006, 1),
        (1.007, 1.008, 1.009, 1),
    ]
    pcd_path = _write_minimal_pcd(tmp_path / "void_drop.pcd", pcd_points)

    result = stream_voxel_aggregate(
        pcd_path=pcd_path,
        tmp_dir=tmp_path / "bins",
        K=4,
        chunksize=128,
    )

    assert isinstance(result, VoxelAggregateResult)
    assert result.n_raw_points == 6
    assert result.n_voxels_pre_void_drop == 2
    assert result.n_voxels_post_void_drop == 1
    assert result.segment.shape == (1,)
    assert result.segment.dtype == np.int32
    # Remapped Wall: raw class 1 → remapped 0.
    assert int(result.segment[0]) == 0

    # Centroid dtype is float32 after the post-process cast.
    assert result.centroid_xyz.shape == (1, 3)
    assert result.centroid_xyz.dtype == np.float32

    # Segment values are strictly in [0, 15] per D-02.
    assert int(result.segment.min()) >= 0
    assert int(result.segment.max()) <= 15

    # Raw histogram includes VOID; final histogram does not.
    raw = result.class_histogram_raw
    final = result.class_histogram_final
    # 3 VOID points + 3 Wall points in the raw cloud.
    assert raw[0] == 3
    assert raw[1] == 3
    # Final remapped space: only 1 voxel, remapped class 0 (Wall).
    assert final[0] == 1
    assert sum(final.values()) == 1

    # Temp bin directory is cleaned up on success.
    assert not (tmp_path / "bins").exists()

    # GRID constant is preserved.
    assert GRID == 0.02
