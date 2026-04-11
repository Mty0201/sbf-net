"""Tests for ``data_pre.zaha.utils.chunking`` — DS-ZAHA-P1-04.

GREEN tests matching the behaviour contract in 01-03-PLAN.md Task 2:
- test_deterministic: two invocations on the same input produce identical
  chunk_idx lists, identical bboxes, and row-major (x-outer, y-inner) order.
- test_budget: Z-band fallback produces chunks whose point counts are each
  below ``budget_per_chunk`` on a 4x4x4 m tile at 800k points of dense noise;
  the same inputs without Z-banding exceed budget (so the test asserts the
  fallback actually matters).
- test_overlap: adjacent tiles share >= overlap_xy metres in the shared axis.

Target impl module: ``data_pre.zaha.utils.chunking`` (created in Plan 03).
Behaviour references: CONTEXT.md D-06 (axis-aligned box grid, row-major
order, deterministic origin = bbox.min), D-07 (<= 0.6 M pts/chunk), D-08
(fixed XY tile size across dataset), D-10 (chunk directory naming),
VALIDATION.md rows ``P01-chunk-01..03``.
"""

from __future__ import annotations

import numpy as np
import pytest


def _load():
    """Guarded import — chunking.py is pure numpy, import should always succeed."""
    return pytest.importorskip("data_pre.zaha.utils.chunking")


def test_deterministic() -> None:
    """P01-chunk-01 — same inputs → same chunk sequence + row-major order."""
    chunking = _load()
    cfg = chunking.ChunkingConfig(tile_xy=4.0, overlap_xy=2.0)
    bbox_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    bbox_max = np.array([12.0, 12.0, 8.0], dtype=np.float32)
    c1 = chunking.compute_chunks(bbox_min, bbox_max, cfg)
    c2 = chunking.compute_chunks(bbox_min, bbox_max, cfg)

    # Frozen dataclasses compare by value.
    assert c1 == c2, "two calls on same input must produce identical chunks"

    # D-10 row-major x-outer, y-inner. With tile=4, overlap=2, stride=2 →
    # the edge enumerator emits [(0,4), (2,6), (4,8), (6,10), (6,12)] on a
    # 12-metre axis (5 tiles), so there are 5 × 5 = 25 chunks.
    assert c1[0].x_tile == 0 and c1[0].y_tile == 0
    assert c1[1].x_tile == 0 and c1[1].y_tile == 1  # y inner-loops first
    assert c1[2].x_tile == 0 and c1[2].y_tile == 2

    # After the y row for x=0 exhausts, x advances to 1 and y resets to 0.
    n_y = max(c.y_tile for c in c1 if c.x_tile == 0) + 1
    assert c1[n_y].x_tile == 1
    assert c1[n_y].y_tile == 0

    # chunk_idx is strictly 0..n-1
    assert [c.chunk_idx for c in c1] == list(range(len(c1)))


def test_budget() -> None:
    """P01-chunk-02 — ``z_mode='band:...'`` holds each chunk <= budget_per_chunk.

    Build a 4 x 4 x 4 m tile packed to 800k points. A single full-Z chunk
    would exceed the 600k budget; switching to ``z_mode='band:2.0'`` splits
    the tile into 2 Z-bands, each ~400k points, both below budget.
    """
    chunking = _load()

    rng = np.random.default_rng(1)
    xyz = rng.uniform(
        low=[0.0, 0.0, 0.0], high=[4.0, 4.0, 4.0], size=(800_000, 3)
    ).astype(np.float32)
    seg = np.zeros(len(xyz), dtype=np.int32)

    # (a) Full-Z on a single 4 m tile is one chunk whose point count exceeds
    # the 600k budget — this is the condition that forces Z-banding in Plan
    # 04's per-sample loop.
    cfg_full = chunking.ChunkingConfig(
        tile_xy=4.0, overlap_xy=2.0, budget_per_chunk=600_000
    )
    chunks_full = chunking.compute_chunks(
        np.array([0.0, 0.0, 0.0]), np.array([4.0, 4.0, 4.0]), cfg_full
    )
    assert len(chunks_full) == 1
    xyz_c, _ = chunking.iter_chunk_points(xyz, seg, chunks_full[0])
    assert len(xyz_c) == 800_000
    assert len(xyz_c) > cfg_full.budget_per_chunk, (
        "synthetic fixture must exceed budget to exercise Z-band fallback"
    )

    # (b) z_mode='band:2.0' splits the 4 m Z extent into 2 bands → each band
    # contains ~400k points which is below the 600k budget.
    cfg_band = chunking.ChunkingConfig(
        tile_xy=4.0,
        overlap_xy=2.0,
        z_mode="band:2.0",
        budget_per_chunk=600_000,
    )
    chunks_band = chunking.compute_chunks(
        np.array([0.0, 0.0, 0.0]), np.array([4.0, 4.0, 4.0]), cfg_band
    )
    assert len(chunks_band) == 2, (
        f"expected 2 Z-bands, got {len(chunks_band)}"
    )
    over_budget = []
    for c in chunks_band:
        xyz_c, _ = chunking.iter_chunk_points(xyz, seg, c)
        if len(xyz_c) > cfg_band.budget_per_chunk:
            over_budget.append((c.chunk_idx, len(xyz_c)))
    assert not over_budget, (
        f"Z-band chunks exceed budget: {over_budget}"
    )


def test_overlap() -> None:
    """P01-chunk-03 — adjacent tiles share >= overlap_xy metres in the shared axis."""
    chunking = _load()
    cfg = chunking.ChunkingConfig(tile_xy=4.0, overlap_xy=2.0)
    chunks = chunking.compute_chunks(
        np.array([0.0, 0.0, 0.0]),
        np.array([12.0, 4.0, 4.0]),
        cfg,
    )

    # With a 4 m XY cube and a 12 m x-extent we expect multiple x tiles,
    # each with y_tile=0 because the Y axis fits in a single 4 m tile.
    xs = [(c.bbox_min[0], c.bbox_max[0]) for c in chunks if c.y_tile == 0]
    assert len(xs) >= 2, f"expected multi-tile x axis, got {xs}"

    for (a0, a1), (b0, b1) in zip(xs, xs[1:]):
        overlap = a1 - b0
        assert overlap >= 2.0 - 1e-6, (
            f"adjacent x overlap {overlap:.6f} < 2.0 m (a=({a0},{a1}) b=({b0},{b1}))"
        )
