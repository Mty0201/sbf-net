"""Support shape probe: does the r=0.06 m band + 6-12/12-18 cm buffer shells
survive voxelisation under grid=0.06 vs grid=0.04?

Produces two artefacts per chunk:
  1. Numeric table (stdout + appended to SUMMARY.md) of band thickness and
     voxel counts at raw / grid=0.06 / grid=0.04, averaged over N Monte-Carlo
     runs to capture GridSample's per-epoch random-representative jitter.
  2. XYZ point clouds with a single integer label column
       0 = non-boundary, 1 = core (r<=0.06), 2 = buf 6-12cm, 3 = buf 12-18cm
     for CloudCompare scalar-field visualisation:
       {stem}_raw.xyz           full-resolution source
       {stem}_g006_rep.xyz      grid=0.06 voxel representatives (one MC run)
       {stem}_g004_rep.xyz      grid=0.04 voxel representatives (one MC run)

Thickness metric (per non-core voxel nothing; per core voxel only):
  for each core voxel v, find its nearest non-core voxel in L_inf distance
  measured in grid-steps. p50 = 1 means "one voxel step away" = 1-voxel-thick
  band. p50 >= 2 means the band is at least 2 voxels thick. This is the test
  for the "core collapsed to pulse" hypothesis.

Usage:
    python scripts/data/support_shape_probe.py \\
        --root /home/mty0201/data/BF_edge_chunk_npy \\
        --split training \\
        --chunks 020103 020501 020901 \\
        --grids 0.06 0.04 \\
        --mc 5 \\
        --out-dir .planning/phases/08-grid04-range-shift-validation/support_shape_probe
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# ---------- BFANet-equivalent boundary detection ----------

def detect_boundary(coord: np.ndarray, segment: np.ndarray, radius: float) -> np.ndarray:
    """A point is a boundary point iff it has at least one neighbour within
    `radius` whose semantic label differs. Matches BFANet's sem_margin.cu."""
    tree = cKDTree(coord)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    mask = np.zeros(coord.shape[0], dtype=bool)
    if pairs.size == 0:
        return mask
    diff = segment[pairs[:, 0]] != segment[pairs[:, 1]]
    mask[pairs[diff, 0]] = True
    mask[pairs[diff, 1]] = True
    return mask


# ---------- GridSample fnv train-mode (verbatim of pointcept/datasets/transform.py:845) ----------

def fnv_hash_vec(arr: np.ndarray) -> np.ndarray:
    """FNV64-1A over rows of an int array."""
    assert arr.ndim == 2
    arr = arr.copy().astype(np.uint64)
    hashed = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
        hashed = hashed * np.uint64(1099511628211)
        hashed = np.bitwise_xor(hashed, arr[:, j])
    return hashed


def grid_sample_train(
    coord: np.ndarray, grid_size: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx_unique, grid_coord_selected). Matches pointcept GridSample
    train-mode with hash_type='fnv'."""
    scaled = coord / grid_size
    grid_coord = np.floor(scaled).astype(np.int64)
    min_coord = grid_coord.min(0)
    grid_coord = grid_coord - min_coord
    key = fnv_hash_vec(grid_coord)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    idx_select = (
        np.cumsum(np.insert(count, 0, 0)[:-1])
        + rng.integers(0, count.max(), count.size) % count
    )
    idx_unique = idx_sort[idx_select]
    return idx_unique, grid_coord[idx_unique]


# ---------- Thickness: two complementary measures on a set of labelled voxels ----------

def shell_depth(positive_grid: np.ndarray, all_grid: np.ndarray) -> np.ndarray:
    """Return the per-voxel 'depth' of each positive voxel, defined as:

        depth(v) = L_inf distance (in grid steps) from v to the nearest
                   negative (non-positive) voxel.

    A positive voxel on the shell surface (has at least one negative
    neighbour within 1 L_inf step) gets depth=1. A voxel wrapped by another
    layer of positives gets depth=2. max(depth) = band thickness in voxels.
    Works by negating: for each positive voxel we look up distance to the
    nearest point in the NEGATIVE set.

    Parameters
    ----------
    positive_grid : (P, 3) int voxel coordinates of positives
    all_grid      : (A, 3) int voxel coordinates of all occupied voxels

    Returns
    -------
    depth : (P,) int array, or empty if positive_grid is empty.
    """
    if positive_grid.size == 0:
        return np.array([], dtype=np.int64)
    # Set-difference via structured-view unique trick: stack all_grid above
    # positive_grid, take unique rows with return_counts; rows with count==1
    # in the unique'd space that came from all_grid are negatives.
    all_rows = np.ascontiguousarray(all_grid, dtype=np.int64)
    pos_rows = np.ascontiguousarray(positive_grid, dtype=np.int64)
    dtype = np.dtype((np.void, all_rows.shape[1] * all_rows.dtype.itemsize))
    all_v = all_rows.view(dtype).ravel()
    pos_v = pos_rows.view(dtype).ravel()
    neg_mask_v = ~np.isin(all_v, pos_v)
    negative_grid = all_rows[neg_mask_v]
    if negative_grid.size == 0:
        return np.full(positive_grid.shape[0], 99, dtype=np.int64)
    tree = cKDTree(negative_grid.astype(np.float64))
    dists, _ = tree.query(positive_grid.astype(np.float64), k=1, p=np.inf)
    return dists.astype(np.int64)


# ---------- Per-chunk measurement ----------

def measure_chunk(
    chunk_dir: Path,
    grids: list[float],
    mc: int,
    rng: np.random.Generator,
) -> dict:
    coord = np.load(chunk_dir / "coord.npy").astype(np.float64)
    segment = np.load(chunk_dir / "segment.npy").reshape(-1)
    n_raw = coord.shape[0]

    # Raw masks — use raw cKDTree radius search for all three radii
    core = detect_boundary(coord, segment, radius=0.06)
    mid12 = detect_boundary(coord, segment, radius=0.12)
    mid18 = detect_boundary(coord, segment, radius=0.18)
    buf_12 = mid12 & ~core
    buf_18 = mid18 & ~mid12  # pure 12-18cm shell
    non = ~(core | buf_12 | buf_18)

    labels_raw = np.zeros(n_raw, dtype=np.uint8)
    labels_raw[core] = 1
    labels_raw[buf_12] = 2
    labels_raw[buf_18] = 3

    result = {
        "raw": {
            "n": n_raw,
            "n_core": int(core.sum()),
            "n_buf12": int(buf_12.sum()),
            "n_buf18": int(buf_18.sum()),
            "n_non": int(non.sum()),
            "ratio_core": float(core.mean()),
            "ratio_buf12": float(buf_12.mean()),
            "ratio_buf18": float(buf_18.mean()),
        },
        "labels_raw": labels_raw,
        "coord_raw": coord.astype(np.float32),
        "grids": {},
    }

    for g in grids:
        per_run = []
        rep_points_last = None
        rep_labels_last = None
        for mc_i in range(mc):
            idx, gcoord = grid_sample_train(coord, g, rng)
            # Per-voxel label = label of the chosen representative point.
            v_labels = labels_raw[idx]
            v_core = v_labels == 1
            v_buf12 = v_labels == 2
            v_buf18 = v_labels == 3

            # Thickness: L_inf grid-step distance from each core voxel to
            # nearest non-core voxel. A core voxel "is a pulse" if p50 == 1.
            core_g = gcoord[v_core]
            noncore_g = gcoord[~v_core]
            core_thick = linf_thickness(core_g, noncore_g)

            buf12_g = gcoord[v_buf12]
            non_buf12_g = gcoord[~v_buf12]
            buf12_thick = linf_thickness(buf12_g, non_buf12_g)

            buf18_g = gcoord[v_buf18]
            non_buf18_g = gcoord[~v_buf18]
            buf18_thick = linf_thickness(buf18_g, non_buf18_g)

            # Count of unique voxels that ever contained a core / buf point
            # AFTER the random-representative step — i.e. voxels we actually
            # learn from this epoch.
            per_run.append(
                {
                    "n_voxels": int(gcoord.shape[0]),
                    "n_core_vox": int(v_core.sum()),
                    "n_buf12_vox": int(v_buf12.sum()),
                    "n_buf18_vox": int(v_buf18.sum()),
                    "core_thick_p50": float(np.median(core_thick)) if core_thick.size else float("nan"),
                    "core_thick_p90": float(np.quantile(core_thick, 0.90)) if core_thick.size else float("nan"),
                    "core_thick_p99": float(np.quantile(core_thick, 0.99)) if core_thick.size else float("nan"),
                    "buf12_thick_p50": float(np.median(buf12_thick)) if buf12_thick.size else float("nan"),
                    "buf12_thick_p90": float(np.quantile(buf12_thick, 0.90)) if buf12_thick.size else float("nan"),
                    "buf18_thick_p50": float(np.median(buf18_thick)) if buf18_thick.size else float("nan"),
                    "buf18_thick_p90": float(np.quantile(buf18_thick, 0.90)) if buf18_thick.size else float("nan"),
                }
            )
            if mc_i == 0:
                rep_points_last = coord[idx].astype(np.float32)
                rep_labels_last = v_labels.copy()

        # Aggregate across MC runs
        keys_num = [k for k in per_run[0].keys()]
        agg = {}
        for k in keys_num:
            vals = np.array([r[k] for r in per_run], dtype=np.float64)
            agg[f"{k}_mean"] = float(np.nanmean(vals))
            agg[f"{k}_std"] = float(np.nanstd(vals))
        result["grids"][g] = {
            "per_run": per_run,
            "agg": agg,
            "rep_points": rep_points_last,
            "rep_labels": rep_labels_last,
        }
    return result


# ---------- XYZ export ----------

def write_xyz(path: Path, coord: np.ndarray, labels: np.ndarray) -> None:
    """Write x y z label (4 columns, space-separated). CloudCompare will load
    the 4th column as a scalar field."""
    assert coord.shape[0] == labels.shape[0]
    out = np.concatenate([coord.astype(np.float32), labels.reshape(-1, 1).astype(np.float32)], axis=1)
    np.savetxt(path, out, fmt="%.4f %.4f %.4f %d")


# ---------- CLI ----------

def fmt_row(chunk_id: str, grid: str, agg: dict) -> str:
    return (
        f"| {chunk_id} | {grid} | "
        f"{agg['n_voxels_mean']:.0f} | "
        f"{agg['n_core_vox_mean']:.0f} | "
        f"{agg['n_buf12_vox_mean']:.0f} | "
        f"{agg['n_buf18_vox_mean']:.0f} | "
        f"{agg['core_thick_p50_mean']:.2f} | "
        f"{agg['core_thick_p90_mean']:.2f} | "
        f"{agg['core_thick_p99_mean']:.2f} | "
        f"{agg['buf12_thick_p50_mean']:.2f} | "
        f"{agg['buf12_thick_p90_mean']:.2f} | "
        f"{agg['buf18_thick_p50_mean']:.2f} | "
        f"{agg['buf18_thick_p90_mean']:.2f} |"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--chunks", nargs="+", required=True)
    parser.add_argument("--grids", nargs="+", type=float, default=[0.06, 0.04])
    parser.add_argument("--mc", type=int, default=5)
    parser.add_argument("--seed", type=int, default=38873367)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(
        "| chunk | grid | n_vox | core_vox | buf12_vox | buf18_vox | "
        "core_p50 | core_p90 | core_p99 | buf12_p50 | buf12_p90 | buf18_p50 | buf18_p90 |"
    )
    print(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )

    all_rows = []
    raw_summary = []
    for chunk_id in args.chunks:
        chunk_dir = args.root / args.split / chunk_id
        res = measure_chunk(chunk_dir, args.grids, args.mc, rng)

        raw = res["raw"]
        raw_summary.append(
            f"{chunk_id}: raw N={raw['n']}  core={raw['n_core']} ({raw['ratio_core']*100:.2f}%)  "
            f"buf12={raw['n_buf12']} ({raw['ratio_buf12']*100:.2f}%)  "
            f"buf18={raw['n_buf18']} ({raw['ratio_buf18']*100:.2f}%)"
        )
        # Raw XYZ
        raw_path = args.out_dir / f"{chunk_id}_raw.xyz"
        write_xyz(raw_path, res["coord_raw"], res["labels_raw"])
        print(f"# wrote {raw_path}", flush=True)

        for g in args.grids:
            gtag = f"g{int(round(g*1000)):03d}"
            row = fmt_row(chunk_id, f"{g:.2f}", res["grids"][g]["agg"])
            print(row, flush=True)
            all_rows.append(row)

            g_path = args.out_dir / f"{chunk_id}_{gtag}_rep.xyz"
            write_xyz(
                g_path,
                res["grids"][g]["rep_points"],
                res["grids"][g]["rep_labels"],
            )
            print(f"# wrote {g_path}", flush=True)

    # Save machine-readable + markdown snippet
    md_path = args.out_dir / "probe_output.md"
    with md_path.open("w") as f:
        f.write("# Support shape probe — raw output\n\n")
        f.write("## Raw per-chunk ratios\n\n")
        for s in raw_summary:
            f.write(f"- {s}\n")
        f.write("\n## Per-(chunk, grid) downsampled stats (mean over MC runs)\n\n")
        f.write(
            "| chunk | grid | n_vox | core_vox | buf12_vox | buf18_vox | "
            "core_p50 | core_p90 | core_p99 | buf12_p50 | buf12_p90 | buf18_p50 | buf18_p90 |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in all_rows:
            f.write(r + "\n")
    print(f"# wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
