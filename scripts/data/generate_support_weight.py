"""Generate per-chunk continuous support-weight fields for BF_edge_chunk_npy.

For each chunk, computes per-point distance to the nearest point of a
different semantic class, then maps that distance through a piecewise
decay to produce an s_weight field in [0, 1]:

    s_weight(d) = 1.0                                   if d <= core_radius
                  decay((d - core_radius) / width)      if core < d <= outer_radius
                  0.0                                   otherwise

where width = outer_radius - core_radius and the decay shape is
exponential `exp(-k * u)` with k = --decay-k, clamped to [0, 1].

This field is a pure per-point loss weight for the semantic CE: it does
NOT replace boundary_mask_r060 for the boundary branch. Core subset
(d <= core_radius) agrees set-wise with boundary_mask_r060 by
construction (every core point has a different-class neighbour within
the core radius, so its nearest-different-class distance is <= core).

Writes `s_weight_r{core_mm:03d}_r{outer_mm:03d}.npy` next to each chunk's
coord.npy, shape (N, 1) float32.

Usage:
    python scripts/data/generate_support_weight.py \\
        --root /home/mty0201/data/BF_edge_chunk_npy \\
        --core-radius 0.06 \\
        --outer-radius 0.12 \\
        --decay-k 3.0 \\
        --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def nearest_different_class_distance(
    coord: np.ndarray, segment: np.ndarray
) -> np.ndarray:
    """For each point p with class c, return the distance to the nearest
    point whose class is != c.

    Implementation: for each class c present in the chunk, build a
    cKDTree on the "other-class" points and query every point-of-this-
    class against it. O(C * N log N).

    Points whose class has no other-class neighbours anywhere in the
    chunk get distance +inf (should not happen on our building data).
    """
    n = coord.shape[0]
    dist = np.full(n, np.inf, dtype=np.float64)
    classes = np.unique(segment)
    for c in classes:
        mask_c = segment == c
        if not mask_c.any():
            continue
        other = coord[~mask_c]
        if other.shape[0] == 0:
            continue
        tree = cKDTree(other)
        d, _ = tree.query(coord[mask_c], k=1)
        dist[mask_c] = d
    return dist


def compute_s_weight(
    distance: np.ndarray,
    core_radius: float,
    outer_radius: float,
    decay_k: float,
) -> np.ndarray:
    """Piecewise exp-decay mapping from physical distance to [0, 1] weight."""
    width = outer_radius - core_radius
    assert width > 0, f"outer_radius must exceed core_radius, got {outer_radius} <= {core_radius}"
    s = np.zeros_like(distance, dtype=np.float32)
    core_mask = distance <= core_radius
    s[core_mask] = 1.0
    buffer_mask = (distance > core_radius) & (distance <= outer_radius)
    u = (distance[buffer_mask] - core_radius) / width
    s[buffer_mask] = np.exp(-decay_k * u).astype(np.float32)
    return s


def process_chunk(args):
    chunk_dir, core_r, outer_r, decay_k, out_name, overwrite = args
    out_path = chunk_dir / out_name
    if out_path.exists() and not overwrite:
        existing = np.load(out_path)
        return (
            chunk_dir.name,
            int(existing.shape[0]),
            float(existing.mean()),
            float((existing >= 1.0 - 1e-4).mean()),
            0.0,
            "skip",
        )

    t0 = time.perf_counter()
    coord = np.load(chunk_dir / "coord.npy").astype(np.float64)
    segment = np.load(chunk_dir / "segment.npy").reshape(-1).astype(np.int32)
    distance = nearest_different_class_distance(coord, segment)
    s_weight = compute_s_weight(distance, core_r, outer_r, decay_k)
    np.save(out_path, s_weight.reshape(-1, 1))
    dt = time.perf_counter() - t0
    return (
        chunk_dir.name,
        int(coord.shape[0]),
        float(s_weight.mean()),
        float((s_weight >= 1.0 - 1e-4).mean()),
        dt,
        "ok",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["training", "validation"])
    parser.add_argument("--core-radius", type=float, default=0.06)
    parser.add_argument("--outer-radius", type=float, default=0.12)
    parser.add_argument("--decay-k", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--chunks",
        nargs="+",
        default=None,
        help="Optional: restrict to a subset of chunk IDs (for pilot runs).",
    )
    args = parser.parse_args()

    core_mm = int(round(args.core_radius * 1000))
    outer_mm = int(round(args.outer_radius * 1000))
    out_name = f"s_weight_r{core_mm:03d}_r{outer_mm:03d}.npy"
    print(f"# core_radius={args.core_radius} m  outer_radius={args.outer_radius} m  "
          f"decay_k={args.decay_k}")
    print(f"# output file: {out_name}")
    print(f"# workers: {args.workers}  overwrite: {args.overwrite}")

    tasks = []
    for split in args.splits:
        split_dir = args.root / split
        chunk_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        if args.chunks is not None:
            chunk_dirs = [p for p in chunk_dirs if p.name in set(args.chunks)]
        for cd in chunk_dirs:
            tasks.append((cd, args.core_radius, args.outer_radius, args.decay_k, out_name, args.overwrite))
        print(f"  {split}: {len(chunk_dirs)} chunks")
    print(f"  total: {len(tasks)} chunks\n")

    if not tasks:
        print("# No chunks matched; exiting.")
        return

    t0 = time.perf_counter()
    results = []
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_chunk, tasks, chunksize=1)):
                results.append(res)
                chunk_id, n, mean_s, core_frac, dt, status = res
                print(
                    f"[{i+1}/{len(tasks)}] {chunk_id}  N={n:>7d}  "
                    f"mean_s={mean_s:6.4f}  core_frac={core_frac*100:5.2f}%  "
                    f"{dt:5.2f}s  {status}"
                )
    else:
        for i, task in enumerate(tasks):
            res = process_chunk(task)
            results.append(res)
            chunk_id, n, mean_s, core_frac, dt, status = res
            print(
                f"[{i+1}/{len(tasks)}] {chunk_id}  N={n:>7d}  "
                f"mean_s={mean_s:6.4f}  core_frac={core_frac*100:5.2f}%  "
                f"{dt:5.2f}s  {status}"
            )
    total_dt = time.perf_counter() - t0

    means = np.array([r[2] for r in results])
    core_fracs = np.array([r[3] for r in results]) * 100.0
    wrote = sum(1 for r in results if r[5] == "ok")
    skipped = sum(1 for r in results if r[5] == "skip")

    print(f"\n# Done in {total_dt:.1f}s  ({wrote} written, {skipped} skipped)")
    print(f"# mean s_weight — min {means.min():.4f}  mean {means.mean():.4f}  "
          f"median {np.median(means):.4f}  max {means.max():.4f}")
    print(f"# core_frac (s>=1) — min {core_fracs.min():.2f}%  mean {core_fracs.mean():.2f}%  "
          f"median {np.median(core_fracs):.2f}%  max {core_fracs.max():.2f}%")


if __name__ == "__main__":
    main()
