"""Per-building density measurement helper (RESEARCH §F.4).

Runs ``stream_voxel_aggregate`` on each ZAHA sample, then computes the
XY footprint and post-VOID-drop voxel density so the planner can pick a
chunking ``tile_xy`` on evidence instead of on the RESEARCH §F.4 worked
example.

For each sample the tool emits:
    * ``n_raw`` — streamed raw-point count from the PCD header
    * ``n_voxels_post_void_drop`` — centroid count after D-01 drop + D-02 remap
    * ``bbox_min`` / ``bbox_max`` — post-VOID-drop bbox
    * ``xy_footprint_m2`` — ``(xmax - xmin) * (ymax - ymin)``
    * ``density_pts_per_m2`` — voxel density over the XY footprint
    * ``suggested_tile_xy_m`` — ``sqrt(budget / density)``

A global recommendation keyed off the worst-case suggested tile is written
to the same JSON under ``recommendation``.

Import order note: this tool does NOT import open3d. It streams raw PCD via
``data_pre.zaha.utils.pcd_parser`` (pandas) and voxel-aggregates via
``data_pre.zaha.utils.voxel_agg`` (numpy). Running it on the ptv3 env is
safe regardless of the §I.5 GLIBCXX trap.

Usage
-----
    conda run -n ptv3 python -m data_pre.zaha.scripts.tools.measure_density \
        --input /home/mty0201/data/ZAHA_pcd \
        --output /tmp/zaha_density.json

    # Or restrict to a small subset of samples:
    conda run -n ptv3 python -m data_pre.zaha.scripts.tools.measure_density \
        --input /home/mty0201/data/ZAHA_pcd \
        --output /tmp/zaha_density_smoke.json \
        --samples DEBY_LOD2_4907179
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np  # noqa: F401  — kept for downstream type inspection

from data_pre.zaha.utils.voxel_agg import stream_voxel_aggregate


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Per-building density measurement for chunking tile selection",
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="ZAHA_pcd root (must contain training/ validation/ test/)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output JSON path for per-sample stats + global recommendation",
    )
    ap.add_argument(
        "--tmp",
        type=Path,
        default=Path("/tmp/zaha_density_tmp"),
        help="temp dir for the external-sort bins (wiped per sample)",
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=600_000,
        help="per-chunk point budget (D-07 default: 600_000)",
    )
    ap.add_argument(
        "--samples",
        type=str,
        default=None,
        help="comma-separated sample basenames to measure (default: all)",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="training,validation,test",
        help="comma-separated splits to walk (default: all three)",
    )
    return ap.parse_args()


def _collect_pcds(
    root: Path,
    splits: list[str],
    samples_filter: set[str] | None,
) -> list[tuple[str, Path]]:
    """Walk the ZAHA_pcd tree and return ``[(split, path), ...]`` sorted."""
    pcds: list[tuple[str, Path]] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for p in sorted(split_dir.glob("*.pcd")):
            if samples_filter is None or p.stem in samples_filter:
                pcds.append((split, p))
    return pcds


def main() -> None:
    args = parse_args()
    samples_filter = (
        set(args.samples.split(",")) if args.samples is not None else None
    )
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    pcds = _collect_pcds(args.input, splits, samples_filter)

    if not pcds:
        raise SystemExit(
            f"no .pcd files found under {args.input} for splits={splits} "
            f"samples_filter={samples_filter}"
        )

    out = {
        "budget_per_chunk": int(args.budget),
        "splits_walked": splits,
        "samples_filter": sorted(samples_filter) if samples_filter else None,
        "samples": [],
    }

    for i, (split, pcd_path) in enumerate(pcds):
        t0 = time.time()
        print(
            f"[{i + 1}/{len(pcds)}] {split}/{pcd_path.name}", flush=True
        )
        result = stream_voxel_aggregate(
            pcd_path, args.tmp / pcd_path.stem, K=16
        )
        xy_span = float(
            (result.bbox_max[0] - result.bbox_min[0])
            * (result.bbox_max[1] - result.bbox_min[1])
        )
        density = (
            result.n_voxels_post_void_drop / xy_span
            if xy_span > 0 else float("inf")
        )
        suggested_tile = (
            math.sqrt(args.budget / density)
            if density > 0 and math.isfinite(density)
            else float("inf")
        )

        entry = {
            "sample": pcd_path.stem,
            "split": split,
            "n_raw": int(result.n_raw_points),
            "n_voxels_pre_void_drop": int(result.n_voxels_pre_void_drop),
            "n_voxels_post_void_drop": int(result.n_voxels_post_void_drop),
            "bbox_min": [float(v) for v in result.bbox_min],
            "bbox_max": [float(v) for v in result.bbox_max],
            "z_extent_m": float(result.bbox_max[2] - result.bbox_min[2]),
            "xy_footprint_m2": xy_span,
            "density_pts_per_m2": float(density),
            "suggested_tile_xy_m": float(suggested_tile),
            "elapsed_s": float(time.time() - t0),
        }
        print(
            f"  n_raw={entry['n_raw']:,} n_vox={entry['n_voxels_post_void_drop']:,} "
            f"xy={entry['xy_footprint_m2']:.0f}m² "
            f"density={entry['density_pts_per_m2']:.0f}/m² "
            f"z_ext={entry['z_extent_m']:.1f}m "
            f"→ max_tile={entry['suggested_tile_xy_m']:.2f}m "
            f"(elapsed={entry['elapsed_s']:.1f}s)",
            flush=True,
        )
        out["samples"].append(entry)

    # Global recommendation: worst-case suggested tile across all samples.
    if out["samples"]:
        finite_tiles = [
            s["suggested_tile_xy_m"]
            for s in out["samples"]
            if math.isfinite(s["suggested_tile_xy_m"])
        ]
        if finite_tiles:
            worst = min(finite_tiles)
            worst_sample = min(
                out["samples"],
                key=lambda s: s["suggested_tile_xy_m"],
            )["sample"]
            # Round DOWN to 0.1 m so the recommendation is conservative.
            rec_tile = math.floor(worst * 10) / 10.0
            out["global_max_tile_xy_m"] = float(worst)
            out["worst_sample"] = worst_sample
            out["recommendation"] = {
                "tile_xy": float(rec_tile),
                "overlap_xy": 2.0,
                "z_mode": "full",
            }
            print(
                f"\nGlobal worst-case max tile: {worst:.2f} m "
                f"(sample={worst_sample}) → recommend {out['recommendation']}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
