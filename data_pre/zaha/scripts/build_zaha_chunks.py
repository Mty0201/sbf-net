"""ZAHA offline preprocessing pipeline — main entry point (Plan 04).

Runs the full per-sample pipeline end-to-end:

    parse PCD -> voxel aggregate -> SOR denoise -> chunk -> per-chunk normals
               -> write NPYs -> manifest.json -> D-21 sanity gates

Usage
-----
    conda run -n ptv3 python data_pre/zaha/scripts/build_zaha_chunks.py \\
        --input /home/mty0201/data/ZAHA_pcd \\
        --output /home/mty0201/data/ZAHA_chunked \\
        --workers 4

CRITICAL import-order (RESEARCH §I.5)
-------------------------------------
``import open3d`` MUST precede ``import numpy`` / ``import pandas`` /
``import scipy`` on the ptv3 conda env — otherwise the loader trips the
``libstdc++.so.6 GLIBCXX_3.4.29 not found`` trap. This module imports open3d
at the very top of the file, and each worker process re-imports open3d
first inside ``process_sample`` so fork+spawn semantics are both safe.

Parallelism model
-----------------
``ProcessPoolExecutor(max_workers=args.workers)`` over small samples, then
serial execution for the two known-large samples (``DEBY_LOD2_4906965``,
``DEBY_LOD2_4959458``) to keep WSL RAM bounded. Small samples benefit from
parallel throughput; the large ones would OOM if stacked in parallel.

Idempotence
-----------
Each sample writes a ``<output>/<split>/.state/<sample>.json`` sidecar after
a successful run. On re-entry the orchestrator reads that sidecar, compares
``source_pcd_sha256``, and skips the sample unless ``--force`` is passed.
"""
from __future__ import annotations

# open3d MUST be imported before numpy / pandas / scipy on the ptv3 env
# (RESEARCH §I.5). Do NOT reorder — imported for load-order side-effect.
import open3d as o3d  # noqa: F401

import argparse
import json
import logging
import multiprocessing as _mp
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from _bootstrap import ensure_zaha_root_on_path  # type: ignore[import-not-found]

ensure_zaha_root_on_path()

# Also put the repo root on sys.path so `data_pre.zaha.*` absolute imports
# resolve both when running via `python -m pytest` (repo-root cwd) and via
# direct invocation of this script from an arbitrary cwd.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402  — must follow open3d
import yaml  # noqa: E402

from data_pre.zaha.utils.common import sha256_file, setup_logger  # noqa: E402
from data_pre.zaha.utils.voxel_agg import stream_voxel_aggregate  # noqa: E402
from data_pre.zaha.utils.denoise import DenoiseConfig, denoise_cloud  # noqa: E402
from data_pre.zaha.utils.chunking import (  # noqa: E402
    ChunkingConfig,
    chunk_name,
    compute_chunks,
    compute_chunks_by_facade,
    iter_chunk_points,
)
from data_pre.zaha.utils.normals import NormalConfig, estimate_normals  # noqa: E402
from data_pre.zaha.utils.layout import write_chunk_npys  # noqa: E402
from data_pre.zaha.utils.manifest import (  # noqa: E402
    ChunkEntry,
    Manifest,
    PIPELINE_VERSION,
    SampleEntry,
    aggregate_dataset_stats,
    build_manifest_shell,
    run_sanity_checks,
    write_manifest,
)


#: Samples whose raw point count is large enough that running them in
#: parallel with two or more workers would exceed WSL's ~6 GB free RAM
#: budget. These are forced to serial execution after the small-sample
#: parallel pool drains. Size thresholds (Plan 01-04 Task 3 measurements,
#: 2026-04-11):
#:
#:   - DEBY_LOD2_4906965  — 4.4 GB PCD / ~86 M pts
#:   - DEBY_LOD2_4959457  — 3.9 GB PCD / ~77 M pts (added Task 3, Rule 2)
#:   - DEBY_LOD2_4959458  — 6.9 GB PCD / ~136 M pts (validation split)
LARGE_SAMPLES: frozenset[str] = frozenset(
    {
        "DEBY_LOD2_4906965",
        "DEBY_LOD2_4959457",
        "DEBY_LOD2_4959458",
    }
)


#: Target post-denoise points per chunk (Plan 01-04 Task 3 re-derivation,
#: 2026-04-12). The continuous adaptive formula picks
#: ``tile_xy = sqrt(TARGET_PTS_PER_CHUNK / density_pt_per_m2)`` per sample
#: and snaps the result to the nearest entry in ``TILE_SNAP_GRID``. 400k is
#: the target median; with the 2× planar-surface correction applied to the
#: raw bbox-projected density the realised max chunk typically lands in
#: 600k-800k and always stays below the 1M D-07 cap.
TARGET_PTS_PER_CHUNK: int = 400_000

#: Discrete snap grid for the continuous adaptive tile. Keeping the set
#: small makes manifest inspection and ablation easier. Overlap is always
#: tile/2 per D-06.
TILE_SNAP_GRID: tuple[float, ...] = (
    8.0,
    10.0,
    12.0,
    16.0,
    20.0,
    24.0,
    32.0,
)


#: Planar-surface correction factor on the raw density estimate.
#:
#: The raw density is ``post_denoise_count / (bbox_x * bbox_y)`` which
#: treats the whole-building point cloud as if it were a floor plan. ZAHA
#: buildings are thin-walled façade scans, so the *surface* area is roughly
#: ``perimeter * height + roof_area ≈ 2 × bbox_x * bbox_y`` on a typical
#: two-storey building — meaning the true point density per m² of
#: cuttable-surface is about half the raw number. Dividing the raw density
#: by this factor before the tile formula compensates for the bias, so the
#: computed ``tile_xy`` targets the real surface density rather than the
#: inflated projected one. Empirically validated on the attempt-5 cache
#: against the per-sample "reverse density" computation (Plan 01-04 Task 3
#: diagnostics, 2026-04-12).
PLANAR_SURFACE_FACTOR: float = 2.0


def compute_adaptive_tile_xy(
    density_pt_per_m2: float,
    target_pts: int = TARGET_PTS_PER_CHUNK,
) -> tuple[float, float, str, float]:
    """Compute ``(tile_xy, overlap_xy, z_mode, tile_xy_raw)`` from density.

    Parameters
    ----------
    density_pt_per_m2
        ``post_denoise_count / (bbox_x * bbox_y)``, after the planar-surface
        correction has already been applied by the caller. Higher density ⇒
        smaller tile.
    target_pts
        Target median points per chunk. Defaults to ``TARGET_PTS_PER_CHUNK``.

    Returns
    -------
    tuple
        ``(tile_xy_m, overlap_xy_m, z_mode_str, tile_xy_raw_m)``. The first
        three are the snapped config values; ``tile_xy_raw`` is the
        continuous pre-snap value, logged into the manifest so downstream
        inspection can verify the formula without re-running the pipeline.
    """
    if density_pt_per_m2 <= 0:
        raise ValueError(
            f"density must be > 0 for tile selection, got {density_pt_per_m2}"
        )
    tile_raw = float(np.sqrt(target_pts / density_pt_per_m2))
    tile_snap = min(TILE_SNAP_GRID, key=lambda t: abs(t - tile_raw))
    if tile_raw > TILE_SNAP_GRID[-1]:
        tile_snap = TILE_SNAP_GRID[-1]
    overlap = tile_snap / 2.0
    return float(tile_snap), float(overlap), f"band:{float(tile_snap)}", tile_raw


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ZAHA offline preprocessing pipeline (Phase 1)",
    )
    p.add_argument("--input", type=Path, required=True, help="ZAHA_pcd root")
    p.add_argument("--output", type=Path, required=True, help="ZAHA_chunked root")
    p.add_argument(
        "--samples",
        type=str,
        default=None,
        help="comma-separated sample basenames; default all",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--force",
        action="store_true",
        help="ignore .state/<sample>.json idempotence sidecars",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="parse + voxel aggregate only; do not write NPYs",
    )
    p.add_argument(
        "--tile-xy",
        type=float,
        default=None,
        help="XY tile size in metres; default = adaptive density bucket "
        "(D-08 supersession). When supplied, all three of --tile-xy / "
        "--overlap-xy / --z-mode must be supplied together to override the "
        "adaptive bucket selector with a fixed-mode CLI config.",
    )
    p.add_argument("--overlap-xy", type=float, default=None)
    p.add_argument("--z-mode", type=str, default=None)
    p.add_argument("--budget-per-chunk", type=int, default=1_000_000)
    p.add_argument("--normals-knn", type=int, default=30)
    p.add_argument(
        "--denoise-notes",
        type=Path,
        default=None,
        help="denoising_notes.md to parse Final Decision YAML block from",
    )
    p.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="lofg3_to_lofg2.yaml path (recorded into manifest.normal_estimation)",
    )
    p.add_argument("--tmp", type=Path, default=Path("/tmp/zaha_build_tmp"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def parse_denoise_final_decision(
    notes_path: Path,
) -> tuple[list[DenoiseConfig], float]:
    """Parse the ``## Final Decision`` YAML block from ``denoising_notes.md``.

    Supported schema:

        method: sor                       # single string
        sor:
          nb_neighbors: 30
          std_ratio: 2.0
        max_drop_frac: 0.10

    Or a sequential pipeline form:

        method: [sor, radius]
        sor:
          nb_neighbors: 30
          std_ratio: 2.0
        radius:
          nb_points: 8
          radius: 0.05
        max_drop_frac: 0.10

    Returns
    -------
    (cfgs, aggregate_drop_cap)
        ``cfgs`` is a list of ``DenoiseConfig`` (one per sequential method);
        each per-step ``max_drop_frac`` is set to 1.0 and the aggregate cap
        is applied in orchestration per D-13.
    """
    if not notes_path.exists():
        raise FileNotFoundError(
            f"{notes_path} not found — Plan 03 Task 1 did not run or was "
            "not committed"
        )
    text = notes_path.read_text()
    start = text.find("## Final Decision")
    if start < 0:
        raise ValueError(
            f'{notes_path}: no "## Final Decision" heading'
        )
    tail = text[start:]
    y0 = tail.find("```yaml")
    if y0 < 0:
        y0 = tail.find("```YAML")
    if y0 < 0:
        raise ValueError(
            f"{notes_path}: no ```yaml fenced block under Final Decision"
        )
    y0 = tail.find("\n", y0) + 1
    y1 = tail.find("```", y0)
    if y1 < 0:
        raise ValueError(
            f"{notes_path}: unterminated ```yaml block under Final Decision"
        )
    block = tail[y0:y1]
    d = yaml.safe_load(block)
    if not isinstance(d, dict):
        raise ValueError(
            f"{notes_path}: Final Decision YAML must be a mapping, got {type(d).__name__}"
        )
    method = d.get("method")
    if isinstance(method, str):
        methods = [method]
    elif isinstance(method, list):
        methods = list(method)
    else:
        raise ValueError(
            f"{notes_path}: method must be string or list, got {method!r}"
        )
    cap = float(d.get("max_drop_frac", 0.10))
    cfgs: list[DenoiseConfig] = []
    for m in methods:
        params = dict(d.get(m, {}))
        # Per-step cap is relaxed; aggregate cap is enforced in orchestration.
        cfgs.append(DenoiseConfig(method=m, params=params, max_drop_frac=1.0))
    return cfgs, cap


# ---------------------------------------------------------------------------
# Sample discovery + idempotence sidecar
# ---------------------------------------------------------------------------


def discover_samples(
    input_root: Path,
    requested: Optional[str],
) -> list[tuple[str, Path]]:
    """Return a list of ``(split, pcd_path)`` tuples sorted by basename."""
    splits = ["training", "validation", "test"]
    found: list[tuple[str, Path]] = []
    requested_set: set[str] | None = None
    if requested:
        requested_set = {s.strip() for s in requested.split(",") if s.strip()}
    for split in splits:
        split_dir = input_root / split
        if not split_dir.is_dir():
            continue
        for p in sorted(split_dir.glob("*.pcd")):
            if requested_set is None or p.stem in requested_set:
                found.append((split, p))
    if requested_set is not None:
        missing = requested_set - {p.stem for _, p in found}
        if missing:
            raise FileNotFoundError(
                f"samples not found under {input_root}: {sorted(missing)}"
            )
    return found


def state_path(output_root: Path, split: str, sample: str) -> Path:
    """Return the ``.state/<sample>.json`` sidecar path for idempotence."""
    return output_root / split / ".state" / f"{sample}.json"


def check_idempotence(
    output_root: Path,
    split: str,
    sample: str,
    source_sha: str,
    force: bool,
) -> bool:
    """Return True if this sample can be skipped (sidecar matches current PCD)."""
    if force:
        return False
    sp = state_path(output_root, split, sample)
    if not sp.exists():
        return False
    try:
        state = json.loads(sp.read_text())
        return (
            state.get("status") == "done"
            and state.get("source_pcd_sha256") == source_sha
        )
    except (OSError, json.JSONDecodeError):
        return False


def write_state(
    output_root: Path,
    split: str,
    sample: str,
    payload: dict,
) -> None:
    """Write the ``.state/<sample>.json`` sidecar atomically."""
    sp = state_path(output_root, split, sample)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Worker entry point — per-sample pipeline
# ---------------------------------------------------------------------------


def process_sample(
    split: str,
    pcd_path: Path,
    output_root: Path,
    tmp_root: Path,
    denoise_cfgs: list[DenoiseConfig],
    aggregate_drop_cap: float,
    chunking_cfg_override: ChunkingConfig | None,
    budget_per_chunk: int,
    normal_cfg: NormalConfig,
    force: bool,
) -> dict:
    """Per-sample worker — runs parse -> voxel agg -> denoise -> chunk -> write.

    ``chunking_cfg_override`` is the CLI-supplied fixed-mode config (used when
    the operator passes ``--tile-xy`` etc. on the command line). When it is
    ``None`` the worker computes the per-sample point density from the
    voxel-agg + denoise output, applies the ``PLANAR_SURFACE_FACTOR``
    correction, runs ``compute_adaptive_tile_xy`` targeting
    ``TARGET_PTS_PER_CHUNK`` points per chunk, and snaps the result to
    ``TILE_SNAP_GRID`` — D-08 supersession v2 (Plan 01-04 Task 3, 2026-04-12
    re-derivation after the first adaptive_density bucket attempt produced
    over-fragmented chunks with medians of 20-30 k points).

    Returns a plain dict (not a dataclass) so the result can be pickled back
    through the ProcessPoolExecutor without importing manifest.py in the
    child-to-parent transport layer.
    """
    # Each worker process must re-import open3d first (RESEARCH §I.5).
    import open3d as _o3d_local  # noqa: F401

    logger = setup_logger(f"zaha.{pcd_path.stem}")
    t0 = time.time()
    sample = pcd_path.stem
    source_sha = sha256_file(pcd_path, max_bytes=8192)

    if check_idempotence(output_root, split, sample, source_sha, force):
        logger.info(f"{sample}: idempotence hit, skipping")
        return {"sample": sample, "split": split, "status": "skipped"}

    logger.info(f"{sample}: starting pipeline")
    write_state(
        output_root,
        split,
        sample,
        {
            "sample": sample,
            "split": split,
            "status": "partial",
            "source_pcd_sha256": source_sha,
            "t0": t0,
        },
    )

    # Stage 1+2: parse + voxel aggregate -----------------------------------
    tmp_sample = tmp_root / sample
    agg = stream_voxel_aggregate(pcd_path, tmp_sample, K=16)
    logger.info(
        f"{sample}: parsed {agg.n_raw_points:,} pts -> "
        f"{agg.n_voxels_post_void_drop:,} voxels post-VOID-drop"
    )

    # Stage 3: denoise (pre-chunking per D-13) -----------------------------
    xyz = agg.centroid_xyz
    seg = agg.segment
    n_in_total = len(xyz)
    for cfg in denoise_cfgs:
        r = denoise_cloud(xyz, seg, cfg)
        xyz = r.xyz
        seg = r.segment
        logger.info(
            f"{sample}: denoise[{cfg.method}] "
            f"{r.n_in:,} -> {r.n_out:,} ({r.drop_frac * 100:.2f}% drop)"
        )
    aggregate_drop = (
        (n_in_total - len(xyz)) / max(n_in_total, 1) if n_in_total > 0 else 0.0
    )
    if aggregate_drop > aggregate_drop_cap:
        raise RuntimeError(
            f"{sample}: aggregate denoise drop "
            f"{aggregate_drop * 100:.2f}% > cap "
            f"{aggregate_drop_cap * 100:.2f}%"
        )
    post_denoise_count = len(xyz)

    # Stage 4: chunk -------------------------------------------------------
    if post_denoise_count < 1:
        raise RuntimeError(
            f"{sample}: zero points after denoise — cannot chunk"
        )
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)

    # Facade-aware chunking (default) or legacy grid (CLI override).
    if chunking_cfg_override is not None:
        # Legacy axis-aligned grid path (--tile-xy CLI override).
        chunks = compute_chunks(bbox_min, bbox_max, chunking_cfg_override)
        tile_bucket_record = {
            "mode": "fixed_cli",
            "tile_xy": float(chunking_cfg_override.tile_xy),
            "overlap_xy": float(chunking_cfg_override.overlap_xy),
            "z_mode": chunking_cfg_override.z_mode,
        }
        logger.info(
            f"{sample}: legacy grid {len(chunks)} chunks "
            f"(CLI override tile={chunking_cfg_override.tile_xy}m)"
        )
        # Build index arrays from ChunkSpec for unified loop below.
        chunk_index_arrays: list[np.ndarray] = []
        for cspec in chunks:
            c_xyz_tmp, _ = iter_chunk_points(xyz, seg, cspec)
            if len(c_xyz_tmp) < 1:
                continue
            mn = cspec.bbox_min
            mx = cspec.bbox_max
            mask = (
                (xyz[:, 0] >= mn[0]) & (xyz[:, 0] <= mx[0])
                & (xyz[:, 1] >= mn[1]) & (xyz[:, 1] <= mx[1])
                & (xyz[:, 2] >= mn[2]) & (xyz[:, 2] <= mx[2])
            )
            chunk_index_arrays.append(np.where(mask)[0])
    else:
        # Facade-aware occupancy chunking.
        chunk_index_arrays = compute_chunks_by_facade(
            xyz, seg, budget=budget_per_chunk, min_pts=10_000,
        )
        tile_bucket_record = {
            "mode": "facade_occupancy",
            "n_components_raw": len(chunk_index_arrays),
            "budget_per_chunk": budget_per_chunk,
            "min_pts": 10_000,
            "occupancy_cell_m": 1.0,
        }
        logger.info(
            f"{sample}: facade chunking -> {len(chunk_index_arrays)} chunks"
        )

    # Stage 5+6: per-chunk normals + write ---------------------------------
    chunk_entries: list[dict] = []
    for ci, idx_arr in enumerate(chunk_index_arrays):
        c_xyz = xyz[idx_arr].astype(np.float32, copy=False)
        c_seg = seg[idx_arr].astype(np.int32, copy=False)
        if len(c_xyz) < normal_cfg.knn:
            logger.warning(
                f"{sample} c{ci:04d}: "
                f"{len(c_xyz)} pts < knn={normal_cfg.knn}, dropping"
            )
            continue
        # Normals (D-19 per-chunk).
        c_normals = estimate_normals(c_xyz, normal_cfg)
        # Write NPYs (D-22 hard-fail on any invariant violation).
        dir_name = chunk_name(sample, ci)
        out_dir = output_root / split / dir_name
        stats = write_chunk_npys(out_dir, c_xyz, c_seg, c_normals)
        # Per-chunk class histogram in remapped 0..15 space.
        bc = np.bincount(c_seg, minlength=16)
        ch_hist = {str(int(k)): int(v) for k, v in enumerate(bc[:16])}
        c_bbox_min = c_xyz.min(axis=0).tolist()
        c_bbox_max = c_xyz.max(axis=0).tolist()
        chunk_entries.append(
            {
                "chunk_idx": ci,
                "dir_name": dir_name,
                "x_tile": 0,
                "y_tile": 0,
                "bbox_min": [float(v) for v in c_bbox_min],
                "bbox_max": [float(v) for v in c_bbox_max],
                "point_count": int(stats["point_count"]),
                "class_histogram": ch_hist,
                "coord_sha256": stats["coord_sha256"],
                "segment_sha256": stats["segment_sha256"],
                "normal_sha256": stats["normal_sha256"],
            }
        )
    if not chunk_entries:
        raise RuntimeError(
            f"{sample}: no chunks written — all tiles empty or below knn floor"
        )
    logger.info(f"{sample}: wrote {len(chunk_entries)} chunks")

    # Stage 7: state + done ------------------------------------------------
    elapsed = time.time() - t0
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = peak_rss_kb / 1024.0  # Linux reports kB
    write_state(
        output_root,
        split,
        sample,
        {
            "sample": sample,
            "split": split,
            "status": "done",
            "source_pcd_sha256": source_sha,
            "n_chunks": len(chunk_entries),
            "elapsed_s": elapsed,
            "peak_rss_mb": peak_rss_mb,
        },
    )

    return {
        "sample": sample,
        "split": split,
        "source_pcd": str(pcd_path),
        "source_pcd_sha256": source_sha,
        "raw_point_count": int(agg.n_raw_points),
        "post_downsample_voxel_count": int(agg.n_voxels_pre_void_drop),
        "post_void_drop_voxel_count": int(agg.n_voxels_post_void_drop),
        "post_denoise_point_count": int(post_denoise_count),
        "bbox_min": [float(v) for v in bbox_min],
        "bbox_max": [float(v) for v in bbox_max],
        "chunks": chunk_entries,
        "tile_bucket": tile_bucket_record,
        "elapsed_s": float(elapsed),
        "peak_rss_mb": float(peak_rss_mb),
        "class_histogram_raw": {
            str(k): int(v) for k, v in agg.class_histogram_raw.items()
        },
        "class_histogram_final": {
            str(k): int(v) for k, v in agg.class_histogram_final.items()
        },
        "status": "done",
    }


# ---------------------------------------------------------------------------
# Module-level worker wrapper (picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _worker_entry(task: dict) -> dict:
    """Picklable wrapper — ProcessPoolExecutor requires a top-level callable."""
    return process_sample(
        split=task["split"],
        pcd_path=Path(task["pcd_path"]),
        output_root=Path(task["output_root"]),
        tmp_root=Path(task["tmp_root"]),
        denoise_cfgs=task["denoise_cfgs"],
        aggregate_drop_cap=task["aggregate_drop_cap"],
        chunking_cfg_override=task["chunking_cfg_override"],
        budget_per_chunk=task["budget_per_chunk"],
        normal_cfg=task["normal_cfg"],
        force=task["force"],
    )


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("zaha.build", args.output / "build.log")
    logger.info(f"ZAHA build starting — pipeline v{PIPELINE_VERSION}")

    phase_dir = (
        _REPO_ROOT
        / ".planning"
        / "workstreams"
        / "dataset-handover-s3dis-chas"
        / "phases"
        / "01-zaha-offline-preprocessing-pipeline"
    )
    denoise_notes = args.denoise_notes or (phase_dir / "denoising_notes.md")
    yaml_path = args.yaml or (phase_dir / "lofg3_to_lofg2.yaml")

    denoise_cfgs, drop_cap = parse_denoise_final_decision(denoise_notes)
    logger.info(
        f"denoise: {[c.method for c in denoise_cfgs]} "
        f"aggregate_drop_cap={drop_cap}"
    )

    cli_tile_flags = (args.tile_xy, args.overlap_xy, args.z_mode)
    cli_tile_supplied = [v is not None for v in cli_tile_flags]
    if any(cli_tile_supplied) and not all(cli_tile_supplied):
        logger.error(
            "CLI override requires all three of --tile-xy / --overlap-xy / "
            "--z-mode together; got "
            f"tile_xy={args.tile_xy} overlap_xy={args.overlap_xy} "
            f"z_mode={args.z_mode}"
        )
        return 1
    if all(cli_tile_supplied):
        chunking_cfg_override: ChunkingConfig | None = ChunkingConfig(
            tile_xy=args.tile_xy,
            overlap_xy=args.overlap_xy,
            z_mode=args.z_mode,
            budget_per_chunk=args.budget_per_chunk,
        )
        logger.info(
            f"chunking: CLI override tile={args.tile_xy} "
            f"overlap={args.overlap_xy} z_mode={args.z_mode}"
        )
    else:
        chunking_cfg_override = None
        logger.info(
            f"chunking: adaptive continuous (D-08 supersession v2): "
            f"tile_xy = round_to_grid(sqrt({TARGET_PTS_PER_CHUNK} / "
            f"(raw_density / {PLANAR_SURFACE_FACTOR})), grid={TILE_SNAP_GRID})"
        )
    normal_cfg = NormalConfig(knn=args.normals_knn, orient=False, fast=False)

    samples = discover_samples(args.input, args.samples)
    logger.info(f"found {len(samples)} samples to process")
    if not samples:
        logger.error("no samples — aborting")
        return 1

    # Schedule: small samples in parallel first, then large samples serial.
    small = [(s, p) for s, p in samples if p.stem not in LARGE_SAMPLES]
    large = [(s, p) for s, p in samples if p.stem in LARGE_SAMPLES]
    logger.info(
        f"schedule: {len(small)} small parallel, {len(large)} large serial"
    )

    sample_payloads: list[dict] = []
    workers = max(1, int(args.workers))

    def _task(split: str, pcd_path: Path) -> dict:
        return {
            "split": split,
            "pcd_path": str(pcd_path),
            "output_root": str(args.output),
            "tmp_root": str(args.tmp),
            "denoise_cfgs": denoise_cfgs,
            "aggregate_drop_cap": drop_cap,
            "chunking_cfg_override": chunking_cfg_override,
            "budget_per_chunk": int(args.budget_per_chunk),
            "normal_cfg": normal_cfg,
            "force": bool(args.force),
        }

    if workers > 1 and len(small) > 1:
        # Force 'spawn' to avoid open3d/OMP futex deadlock under fork.
        # Parent imports open3d at module top (GLIBCXX load-order trap),
        # which initialises an OMP thread pool; fork inherits those locks
        # in a broken state and workers deadlock on first open3d call.
        # (Plan 01-04 Task 3 Rule-3 fix, 2026-04-12.)
        _ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=_ctx) as pool:
            futures = {
                pool.submit(_worker_entry, _task(split, p)): (split, p)
                for split, p in small
            }
            for f in as_completed(futures):
                split, p = futures[f]
                try:
                    sample_payloads.append(f.result())
                except Exception as exc:
                    logger.error(f"{p.stem} failed: {exc}")
                    raise
    else:
        for split, p in small:
            sample_payloads.append(_worker_entry(_task(split, p)))

    for split, p in large:
        logger.info(f"{p.stem}: large sample, serial")
        sample_payloads.append(_worker_entry(_task(split, p)))

    # Build manifest -------------------------------------------------------
    ran_samples = [d for d in sample_payloads if d.get("status") == "done"]
    sample_entries = [
        SampleEntry(
            sample=d["sample"],
            split=d["split"],
            source_pcd=d["source_pcd"],
            source_pcd_sha256=d["source_pcd_sha256"],
            raw_point_count=int(d["raw_point_count"]),
            post_downsample_voxel_count=int(d["post_downsample_voxel_count"]),
            post_void_drop_voxel_count=int(d["post_void_drop_voxel_count"]),
            post_denoise_point_count=int(d["post_denoise_point_count"]),
            bbox_min=list(d["bbox_min"]),
            bbox_max=list(d["bbox_max"]),
            chunks=[ChunkEntry(**c) for c in d["chunks"]],
            elapsed_s=float(d["elapsed_s"]),
            peak_rss_mb=float(d["peak_rss_mb"]),
            class_histogram_raw=dict(d["class_histogram_raw"]),
            class_histogram_final=dict(d["class_histogram_final"]),
        )
        for d in ran_samples
    ]
    manifest = build_manifest_shell(
        denoising={
            "methods": [c.method for c in denoise_cfgs],
            "params": {c.method: c.params for c in denoise_cfgs},
            "aggregate_drop_cap": drop_cap,
            "decision_source": str(denoise_notes),
        },
        normal_estimation={
            "method": "adaptive_radius_pca",
            "params": {
                "knn": normal_cfg.knn,
                "orient": normal_cfg.orient,
                "fast": normal_cfg.fast,
            },
            "decision_source": str(phase_dir / "normals_notes.md"),
        },
        chunking=(
            {
                "mode": "fixed_cli",
                "tile_xy": chunking_cfg_override.tile_xy,
                "overlap_xy": chunking_cfg_override.overlap_xy,
                "stride_xy": chunking_cfg_override.stride_xy,
                "z_mode": chunking_cfg_override.z_mode,
                "budget_per_chunk": chunking_cfg_override.budget_per_chunk,
            }
            if chunking_cfg_override is not None
            else {
                "mode": "adaptive_continuous",
                "target_pts_per_chunk": int(TARGET_PTS_PER_CHUNK),
                "planar_surface_factor": float(PLANAR_SURFACE_FACTOR),
                "snap_grid_m": [float(t) for t in TILE_SNAP_GRID],
                "budget_per_chunk": int(args.budget_per_chunk),
                "per_sample": {
                    d["sample"]: d["tile_bucket"]
                    for d in sample_payloads
                    if d.get("status") == "done" and "tile_bucket" in d
                },
            }
        ),
    )
    manifest.samples = sample_entries
    manifest.dataset_stats = aggregate_dataset_stats(sample_entries)
    # Record the lofg3_to_lofg2.yaml path as provenance even though Phase 1
    # does not apply the remap itself (deferred to Phase 1b / config time).
    manifest.normal_estimation["lofg3_to_lofg2_yaml"] = str(yaml_path)
    write_manifest(args.output, manifest)

    # Sanity checks (D-21) -------------------------------------------------
    expected = {p.stem for _, p in samples}
    errors = run_sanity_checks(manifest, expected)
    hard_fails = [e for e in errors if "HARD FAIL" in e]
    warnings = [e for e in errors if "HARD FAIL" not in e]
    for w in warnings:
        logger.warning(w)
    if hard_fails:
        for e in hard_fails:
            logger.error(e)
        logger.error(f"{len(hard_fails)} sanity failures — exit 2")
        return 2
    logger.info(
        f"build complete: {len(sample_entries)} samples, "
        f"{sum(len(s.chunks) for s in sample_entries)} chunks"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
