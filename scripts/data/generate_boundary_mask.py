"""Generate per-chunk boundary masks for BF_edge_chunk_npy at a fixed radius.

Runs the BFANet boundary-detection algorithm (radius search: a point is a
boundary point iff any neighbor within `radius` carries a different
semantic label) on every chunk of both training and validation splits.

Writes `boundary_mask_r{mm}.npy` next to each chunk's `coord.npy`, shape
(N, 1) uint8, matching segment.npy convention.

Usage:
    python scripts/data/generate_boundary_mask.py \\
        --root /home/mty0201/data/BF_edge_chunk_npy \\
        --radius 0.06 \\
        --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def detect_boundary(coord: np.ndarray, segment: np.ndarray, radius: float) -> np.ndarray:
    """BFANet-equivalent boundary detection via cKDTree radius search."""
    tree = cKDTree(coord)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    boundary = np.zeros(coord.shape[0], dtype=bool)
    if pairs.size:
        diff = segment[pairs[:, 0]] != segment[pairs[:, 1]]
        boundary[pairs[diff, 0]] = True
        boundary[pairs[diff, 1]] = True
    return boundary


def process_chunk(args):
    chunk_dir, radius, out_name, overwrite = args
    out_path = chunk_dir / out_name
    if out_path.exists() and not overwrite:
        n = int(np.load(out_path).shape[0])
        ratio = float(np.load(out_path).mean())
        return (chunk_dir.name, n, ratio, 0.0, "skip")

    t0 = time.perf_counter()
    coord = np.load(chunk_dir / "coord.npy").astype(np.float64)
    segment = np.load(chunk_dir / "segment.npy").reshape(-1).astype(np.int32)
    boundary = detect_boundary(coord, segment, radius)
    mask = boundary.astype(np.uint8).reshape(-1, 1)
    np.save(out_path, mask)
    dt = time.perf_counter() - t0
    return (chunk_dir.name, int(coord.shape[0]), float(boundary.mean()), dt, "ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["training", "validation"])
    parser.add_argument("--radius", type=float, default=0.06)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_name = f"boundary_mask_r{int(round(args.radius * 1000)):03d}.npy"
    print(f"# Radius: {args.radius} m  ->  output file: {out_name}")
    print(f"# Workers: {args.workers}  overwrite: {args.overwrite}")

    tasks = []
    per_split_counts: dict[str, int] = {}
    for split in args.splits:
        split_dir = args.root / split
        chunk_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
        per_split_counts[split] = len(chunk_dirs)
        for cd in chunk_dirs:
            tasks.append((cd, args.radius, out_name, args.overwrite))
        print(f"  {split}: {len(chunk_dirs)} chunks")
    print(f"  total: {len(tasks)} chunks\n")

    t0 = time.perf_counter()
    results = []
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(process_chunk, tasks, chunksize=1)):
                results.append(res)
                chunk_id, n, ratio, dt, status = res
                print(
                    f"[{i+1}/{len(tasks)}] {chunk_id}  N={n:>7d}  "
                    f"ratio={ratio*100:5.2f}%  {dt:4.1f}s  {status}"
                )
    else:
        for i, task in enumerate(tasks):
            res = process_chunk(task)
            results.append(res)
            chunk_id, n, ratio, dt, status = res
            print(
                f"[{i+1}/{len(tasks)}] {chunk_id}  N={n:>7d}  "
                f"ratio={ratio*100:5.2f}%  {dt:4.1f}s  {status}"
            )
    total_dt = time.perf_counter() - t0

    ratios = np.array([r[2] for r in results]) * 100.0
    wrote = sum(1 for r in results if r[4] == "ok")
    skipped = sum(1 for r in results if r[4] == "skip")

    print(f"\n# Done in {total_dt:.1f}s  ({wrote} written, {skipped} skipped)")
    print(f"# Positive ratio — min {ratios.min():.2f}%  mean {ratios.mean():.2f}%  "
          f"median {np.median(ratios):.2f}%  max {ratios.max():.2f}%  std {ratios.std():.2f}%")
    in_window = int(np.sum((ratios >= 5.0) & (ratios <= 15.0)))
    print(f"# In [5, 15]% window: {in_window}/{len(ratios)}")


if __name__ == "__main__":
    main()
