"""Verify alignment between runtime KNN edge detection and precomputed support geometry.

Compares two boundary definitions:
  A) Runtime KNN: for each voxel, check if its KNN neighbourhood spans multiple
     semantic classes (cross_ratio >= threshold) — this is what DLA-Net does at
     inference time.
  B) Precomputed support proxy: points where edge_support > some threshold
     (default 0.9), i.e. the "core" of the Gaussian-decay boundary zone produced
     by the data_pre pipeline.

The key question: do these two definitions agree? If not, the precomputed
(support+1)*valid BCE supervision in CR-F is teaching a different boundary than
what a runtime KNN detector would see.

Optimisations over the previous version:
  - Spatial chunking: splits the scene into overlapping cubes so each KNN call
    operates on a bounded subset. Only the non-overlap "core" of each chunk
    contributes to the final result, avoiding boundary artefacts.
  - All KNN and label comparison stays on GPU within each chunk.
  - Sweeps multiple support thresholds in a single pass.

Usage:
    conda run -n ptv3 python scripts/analysis/verify_knn_vs_support.py \
        [--edge-root PATH] [--chunk-size 10.0] [--overlap 0.5] \
        [--support-thresholds 0.5,0.7,0.9]
"""

from __future__ import annotations

import argparse
import gc as _gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_cluster import knn

# ---------------------------------------------------------------------------
# Config defaults (match training / data_pre pipeline)
# ---------------------------------------------------------------------------
GRID_SIZE = 0.06          # val-pipeline GridSample voxel size
KNN_K = 32                # Stage1Config.k
MIN_CROSS_RATIO = 0.15    # Stage1Config.min_cross_ratio
DEVICE = "cuda"

DEFAULT_SUPPORT_THRESHOLDS = [0.5, 0.7, 0.9]
DEFAULT_CHUNK_SIZE = 10.0   # metres — cube edge length
DEFAULT_OVERLAP = 0.5       # metres — overlap band on each side


# ---------------------------------------------------------------------------
# Grid sampling (reproduces val-pipeline deterministic voxelisation)
# ---------------------------------------------------------------------------

def grid_sample_val(coord, segment, edge):
    """Reproduce the val-pipeline GridSample (deterministic FNV-1a hash)."""
    scaled = coord / GRID_SIZE
    grid_coord = np.floor(scaled).astype(np.int64)
    grid_coord -= grid_coord.min(0)

    p = np.uint64(0x100000001B3)
    h = np.full(grid_coord.shape[0], 0x811C9DC5, dtype=np.uint64)
    for i in range(grid_coord.shape[1]):
        h ^= grid_coord[:, i].astype(np.uint64)
        h *= p

    idx_sort = np.argsort(h)
    key_sort = h[idx_sort]
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
    idx_select = np.cumsum(np.insert(count, 0, 0)[:-1])
    idx_unique = idx_sort[idx_select]

    return coord[idx_unique], segment[idx_unique], edge[idx_unique]


# ---------------------------------------------------------------------------
# Chunked GPU KNN edge detection
# ---------------------------------------------------------------------------

def _chunk_bounds(coord_min, coord_max, chunk_size, overlap):
    """Yield (lo, hi, core_lo, core_hi) for each chunk along each axis."""
    axes = []
    for d in range(3):
        lo = coord_min[d]
        hi = coord_max[d]
        edges = []
        while lo < hi:
            c_hi = min(lo + chunk_size, hi)
            # The padded range includes overlap on both sides
            pad_lo = lo - overlap
            pad_hi = c_hi + overlap
            edges.append((pad_lo, pad_hi, lo, c_hi))
            lo = c_hi
        axes.append(edges)
    # Cartesian product of axis chunks
    for ax in axes[0]:
        for ay in axes[1]:
            for az in axes[2]:
                pad_lo = np.array([ax[0], ay[0], az[0]], dtype=np.float32)
                pad_hi = np.array([ax[1], ay[1], az[1]], dtype=np.float32)
                core_lo = np.array([ax[2], ay[2], az[2]], dtype=np.float32)
                core_hi = np.array([ax[3], ay[3], az[3]], dtype=np.float32)
                yield pad_lo, pad_hi, core_lo, core_hi


def detect_knn_edges_chunked(coord, segment, k=KNN_K,
                             min_cross_ratio=MIN_CROSS_RATIO,
                             chunk_size=DEFAULT_CHUNK_SIZE,
                             overlap=DEFAULT_OVERLAP,
                             ignore_index=-1):
    """KNN semantic-heterogeneity edge detection with spatial chunking on GPU.

    Returns uint8 array of length N: 1 = edge point, 0 = interior.
    """
    n = coord.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.uint8)

    result = np.zeros(n, dtype=np.uint8)
    coord_min = coord.min(axis=0)
    coord_max = coord.max(axis=0)

    # Pre-upload segment to GPU once
    seg_gpu = torch.from_numpy(segment.astype(np.int64)).to(DEVICE)

    for pad_lo, pad_hi, core_lo, core_hi in _chunk_bounds(
            coord_min, coord_max, chunk_size, overlap):

        # Select points in padded region (for KNN context)
        pad_mask = np.all((coord >= pad_lo) & (coord < pad_hi), axis=1)
        pad_idx = np.where(pad_mask)[0]
        if len(pad_idx) == 0:
            continue

        # Select points in core region (for output)
        core_mask_local = np.all(
            (coord[pad_idx] >= core_lo) & (coord[pad_idx] < core_hi), axis=1
        )

        chunk_coord = torch.from_numpy(coord[pad_idx]).to(DEVICE)
        chunk_seg = seg_gpu[pad_idx]
        nc = chunk_coord.shape[0]

        if nc <= k + 1:
            # Too few points — can't do meaningful KNN, skip
            continue

        # KNN within chunk
        edge_index = knn(chunk_coord, chunk_coord, k=k + 1)
        neighbors = edge_index[1].reshape(nc, k + 1)

        # Label comparison on GPU
        point_idx_t = torch.arange(nc, device=DEVICE).unsqueeze(1)
        self_match = neighbors == point_idx_t

        neighbor_labels = chunk_seg[neighbors]
        neighbor_labels[self_match] = ignore_index

        self_labels = chunk_seg.unsqueeze(1)
        self_valid = chunk_seg != ignore_index
        neighbor_valid = neighbor_labels != ignore_index
        n_valid = neighbor_valid.sum(dim=1).clamp(min=1)

        diff = neighbor_valid & (neighbor_labels != self_labels)
        n_diff = diff.sum(dim=1)

        cross_ratio = n_diff.float() / n_valid.float()
        is_edge = (self_valid & (cross_ratio >= min_cross_ratio)).cpu().numpy()

        # Write only core points to result
        core_global_idx = pad_idx[core_mask_local]
        core_is_edge = is_edge[core_mask_local]
        result[core_global_idx] = core_is_edge.astype(np.uint8)

        del chunk_coord, chunk_seg, edge_index, neighbors, neighbor_labels
        torch.cuda.empty_cache()

    del seg_gpu
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Per-scene analysis
# ---------------------------------------------------------------------------

def analyze_scene(edge_dir: Path, support_thresholds: list[float],
                  chunk_size: float, overlap: float):
    """Run diagnostics on one scene across multiple support thresholds."""
    coord = np.load(edge_dir / "coord.npy").astype(np.float32)
    segment = np.load(edge_dir / "segment.npy").astype(np.int32)
    edge = np.load(edge_dir / "edge.npy").astype(np.float32)

    v_coord, v_seg, v_edge = grid_sample_val(coord, segment, edge)
    del coord, segment, edge
    _gc.collect()

    n_vox = v_coord.shape[0]
    v_support = v_edge[:, 3]   # Gaussian decay weight
    v_valid = v_edge[:, 4]     # binary validity

    # Runtime KNN edge detection (chunked)
    t0 = time.time()
    knn_edge = detect_knn_edges_chunked(
        v_coord, v_seg, chunk_size=chunk_size, overlap=overlap)
    knn_time = time.time() - t0
    knn_mask = knn_edge == 1

    # Sweep support thresholds
    threshold_results = {}
    for thr in support_thresholds:
        proxy_mask = v_support > thr
        both = int((knn_mask & proxy_mask).sum())
        knn_only = int((knn_mask & ~proxy_mask).sum())
        proxy_only = int((~knn_mask & proxy_mask).sum())
        union = both + knn_only + proxy_only

        iou = both / max(union, 1)
        recall_proxy = both / max(int(proxy_mask.sum()), 1)   # what fraction of proxy is in KNN
        recall_knn = both / max(int(knn_mask.sum()), 1)       # what fraction of KNN is in proxy

        threshold_results[thr] = dict(
            proxy_count=int(proxy_mask.sum()),
            proxy_ratio=float(proxy_mask.mean()),
            overlap=both,
            knn_only=knn_only,
            proxy_only=proxy_only,
            iou=iou,
            recall_proxy_in_knn=recall_proxy,
            recall_knn_in_proxy=recall_knn,
        )

    # Also compare with valid (the original target)
    valid_mask = v_valid > 0.5
    both_v = int((knn_mask & valid_mask).sum())
    knn_only_v = int((knn_mask & ~valid_mask).sum())
    valid_only_v = int((~knn_mask & valid_mask).sum())
    union_v = both_v + knn_only_v + valid_only_v
    valid_result = dict(
        valid_count=int(valid_mask.sum()),
        valid_ratio=float(valid_mask.mean()),
        overlap=both_v,
        knn_only=knn_only_v,
        valid_only=valid_only_v,
        iou=both_v / max(union_v, 1),
        recall_valid_in_knn=both_v / max(int(valid_mask.sum()), 1),
        recall_knn_in_valid=both_v / max(int(knn_mask.sum()), 1),
    )

    # Per-class breakdown (for the tightest threshold only)
    tightest = max(support_thresholds)
    proxy_tight = v_support > tightest
    class_names = [
        "balustrade", "balcony", "advboard", "wall",
        "eave", "column", "window", "clutter",
    ]
    per_class = {}
    for cls_id, cls_name in enumerate(class_names):
        cm = v_seg == cls_id
        nc = int(cm.sum())
        if nc == 0:
            continue
        ck = int((knn_mask & cm).sum())
        cp = int((proxy_tight & cm).sum())
        cb = int((knn_mask & proxy_tight & cm).sum())
        ci = cb / max(ck + cp - cb, 1)
        per_class[cls_name] = dict(
            n_points=nc, knn_edge=ck, proxy_edge=cp, overlap=cb, iou=ci,
        )

    del v_coord, v_seg, v_edge
    _gc.collect()

    return dict(
        scene=edge_dir.name,
        n_voxelized=n_vox,
        knn_count=int(knn_mask.sum()),
        knn_ratio=float(knn_mask.mean()),
        knn_time_s=knn_time,
        thresholds=threshold_results,
        valid=valid_result,
        per_class=per_class,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: list[dict], thresholds: list[float]):
    print("\n" + "=" * 90)
    print("KNN vs Precomputed Support Alignment Report (chunked GPU)")
    print(f"GridSample={GRID_SIZE}m  KNN_K={KNN_K}  min_cross_ratio={MIN_CROSS_RATIO}")
    print("=" * 90)

    for r in results:
        print(f"\n--- {r['scene']} ({r['n_voxelized']} vox, "
              f"KNN {r['knn_count']} pts = {r['knn_ratio']:.4f}, "
              f"{r['knn_time_s']:.1f}s) ---")

        # Valid comparison
        v = r["valid"]
        print(f"  vs valid: IoU={v['iou']:.4f}  "
              f"R(valid→KNN)={v['recall_valid_in_knn']:.4f}  "
              f"R(KNN→valid)={v['recall_knn_in_valid']:.4f}  "
              f"[valid={v['valid_count']}, overlap={v['overlap']}, "
              f"knn_only={v['knn_only']}, valid_only={v['valid_only']}]")

        # Support threshold comparisons
        for thr in thresholds:
            t = r["thresholds"][thr]
            print(f"  vs support>{thr}: IoU={t['iou']:.4f}  "
                  f"R(proxy→KNN)={t['recall_proxy_in_knn']:.4f}  "
                  f"R(KNN→proxy)={t['recall_knn_in_proxy']:.4f}  "
                  f"[proxy={t['proxy_count']}, overlap={t['overlap']}, "
                  f"knn_only={t['knn_only']}, proxy_only={t['proxy_only']}]")

        # Per-class
        if r["per_class"]:
            tightest = max(thresholds)
            print(f"  Per-class (support>{tightest}):")
            for cls, s in sorted(r["per_class"].items(), key=lambda x: x[1]["iou"]):
                print(f"    {cls:12s}: IoU={s['iou']:.4f}  "
                      f"knn={s['knn_edge']:5d}  proxy={s['proxy_edge']:5d}  "
                      f"overlap={s['overlap']:5d}")

    # Aggregate
    if len(results) > 1:
        print("\n" + "=" * 90)
        print("AGGREGATE")
        print("=" * 90)

        # Valid
        for key in ["iou", "recall_valid_in_knn", "recall_knn_in_valid"]:
            vals = [r["valid"][key] for r in results]
            print(f"  valid {key:25s}: mean={np.mean(vals):.4f} "
                  f"std={np.std(vals):.4f}")

        # Per threshold
        for thr in thresholds:
            print(f"  --- support>{thr} ---")
            for key in ["iou", "recall_proxy_in_knn", "recall_knn_in_proxy"]:
                vals = [r["thresholds"][thr][key] for r in results]
                print(f"    {key:25s}: mean={np.mean(vals):.4f} "
                      f"std={np.std(vals):.4f}")

        # Timing
        times = [r["knn_time_s"] for r in results]
        print(f"\n  KNN time: mean={np.mean(times):.1f}s "
              f"total={np.sum(times):.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--edge-root", type=str,
                    default="/mnt/e/WSL/data/BF_edge_chunk_npy",
                    help="Dataset root with coord/segment/edge.npy")
    pa.add_argument("--chunk-size", type=float, default=DEFAULT_CHUNK_SIZE,
                    help="Spatial chunk edge length in metres (default: 10.0)")
    pa.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP,
                    help="Overlap band in metres (default: 0.5)")
    pa.add_argument("--support-thresholds", type=str, default="0.5,0.7,0.9",
                    help="Comma-separated support thresholds (default: 0.5,0.7,0.9)")
    args = pa.parse_args()

    thresholds = sorted(float(x) for x in args.support_thresholds.split(","))

    val_dir = Path(args.edge_root) / "validation"
    if not val_dir.is_dir():
        print(f"ERROR: {val_dir} not found", file=sys.stderr)
        sys.exit(1)

    scene_dirs = sorted(d for d in val_dir.iterdir()
                        if d.is_dir() and (d / "edge.npy").is_file())
    print(f"Found {len(scene_dirs)} validation scenes  |  device={DEVICE}")
    print(f"Chunk size={args.chunk_size}m  overlap={args.overlap}m  "
          f"thresholds={thresholds}")

    results = []
    for i, sd in enumerate(scene_dirs):
        print(f"  [{i+1}/{len(scene_dirs)}] {sd.name} ...", end="", flush=True)
        r = analyze_scene(sd, thresholds, args.chunk_size, args.overlap)
        best_thr = max(thresholds)
        t = r["thresholds"][best_thr]
        print(f" IoU(s>{best_thr})={t['iou']:.4f}  "
              f"IoU(valid)={r['valid']['iou']:.4f}  "
              f"({r['knn_time_s']:.1f}s)")
        results.append(r)

    if results:
        print_report(results, thresholds)


if __name__ == "__main__":
    main()
