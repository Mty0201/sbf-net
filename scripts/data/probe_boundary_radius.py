"""Sampling probe: find the right r for BF_edge_chunk_npy boundary mask.

Runs the BFANet boundary-detection algorithm (radius search, mark a point if
it has any different-class neighbor inside r) on a random sample of chunks
at several candidate radii and reports the positive-ratio distribution.

Goal: pick r landing ratio in ~5-15%, matching BFANet's implicit design
window. Primary candidate is r = 0.06 m (BFANet absolute physical value);
sweep brackets it on both sides to verify monotonicity.

Usage:
    python scripts/data/probe_boundary_radius.py \\
        --root /home/mty0201/data/BF_edge_chunk_npy \\
        --split training \\
        --sample 10 \\
        --radii 0.03 0.06 0.09 0.12
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def detect_boundary(coord: np.ndarray, segment: np.ndarray, radius: float) -> np.ndarray:
    """BFANet-equivalent boundary detection via cKDTree radius search.

    Matches sem_margin.cu semantics: a point is a boundary point iff it has
    at least one neighbor within `radius` whose semantic label differs.
    """
    tree = cKDTree(coord)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if pairs.size == 0:
        return np.zeros(coord.shape[0], dtype=bool)
    diff = segment[pairs[:, 0]] != segment[pairs[:, 1]]
    boundary = np.zeros(coord.shape[0], dtype=bool)
    boundary[pairs[diff, 0]] = True
    boundary[pairs[diff, 1]] = True
    return boundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--sample", type=int, default=10, help="Number of chunks to sample")
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[0.03, 0.06, 0.09, 0.12],
        help="Candidate radii in meters",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split_dir = args.root / args.split
    all_chunks = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    rng = random.Random(args.seed)
    sample = rng.sample(all_chunks, min(args.sample, len(all_chunks)))

    print(f"# Probe: {len(sample)} chunks from {split_dir}")
    print(f"# Radii: {args.radii} meters")
    print(f"# Seed: {args.seed}\n")

    # results[radius][chunk_idx] = positive_ratio
    results: dict[float, list[float]] = {r: [] for r in args.radii}
    per_chunk_rows = []

    for i, chunk_id in enumerate(sample):
        chunk_dir = split_dir / chunk_id
        coord = np.load(chunk_dir / "coord.npy").astype(np.float64)
        segment = np.load(chunk_dir / "segment.npy").reshape(-1).astype(np.int32)
        n = coord.shape[0]

        row = {"chunk": chunk_id, "N": n}
        for r in args.radii:
            t0 = time.perf_counter()
            boundary = detect_boundary(coord, segment, r)
            dt = time.perf_counter() - t0
            ratio = float(boundary.mean())
            results[r].append(ratio)
            row[f"r{r}"] = ratio
            row[f"t{r}"] = dt
        per_chunk_rows.append(row)
        print(
            f"[{i+1}/{len(sample)}] {chunk_id}  N={n:>7d}  "
            + "  ".join(
                f"r={r}: {row[f'r{r}']*100:5.2f}% ({row[f't{r}']:4.1f}s)"
                for r in args.radii
            )
        )

    print("\n# Summary (positive ratio percentage)")
    print(f"{'radius (m)':>12} {'min':>8} {'mean':>8} {'median':>8} {'max':>8} {'std':>8}  in_window [5,15]%")
    for r in args.radii:
        vals = np.array(results[r]) * 100.0
        in_window = np.sum((vals >= 5.0) & (vals <= 15.0))
        print(
            f"{r:>12.3f} {vals.min():>8.2f} {vals.mean():>8.2f} {np.median(vals):>8.2f} "
            f"{vals.max():>8.2f} {vals.std():>8.2f}  {in_window}/{len(vals)}"
        )

    print("\n# Recommendation:")
    # Pick radius whose MEAN ratio is closest to the center of [5, 15] = 10,
    # but only among those with at least 60% of chunks inside the window.
    best = None
    for r in args.radii:
        vals = np.array(results[r]) * 100.0
        in_window_frac = np.mean((vals >= 5.0) & (vals <= 15.0))
        mean = vals.mean()
        score = abs(mean - 10.0)
        if in_window_frac >= 0.6 and (best is None or score < best[1]):
            best = (r, score, mean, in_window_frac)
    if best is None:
        print("  None of the tested radii land ≥60% of chunks in [5,15]%.")
        print("  Consider a wider sweep — check min/max columns above for direction.")
    else:
        print(f"  r = {best[0]} m (mean {best[2]:.2f}%, {best[3]*100:.0f}% of chunks in window)")


if __name__ == "__main__":
    main()
