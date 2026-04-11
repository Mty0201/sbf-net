"""Deterministic grid=0.02 voxel aggregator for the ZAHA offline pipeline.

Exports
-------
GRID : float = 0.02                  (module constant — NOT a parameter)
VoxelBatch (dataclass)               per-bin aggregate result
VoxelAggregateResult (dataclass)     whole-file aggregate result
pack_voxel_keys(ix, iy, iz)          pack int64 voxel coords to uint64 key
voxel_aggregate_batch(xyz, cls)      deterministic in-RAM aggregate
stream_voxel_aggregate(path, ...)    hash-partitioned external sort aggregate

Design (RESEARCH §B.1 / §B.2)
-----------------------------
CONTEXT D-14 supersession: the plain Python-dict aggregate in the original
D-14 wording does not fit WSL's 4 GB free RAM for the 136.8 M-point sample
(would need 17-34 GB of dict state). This module implements the hash-
partitioned external sort of RESEARCH §B.2 instead. Peak RAM ~1.5 GB.

Pass 1 (partition):
    stream_pcd → per-chunk numpy voxel keys → partition into K disk bins via
    ``key % K``. Equal keys always land in the same bin by construction.
Pass 2 (per-bin aggregate):
    load each bin into RAM, run voxel_aggregate_batch on it (stable sort +
    np.unique + reduceat), concatenate results across bins.
Pass 3 (post-process):
    VOID drop (D-01) + remap 1..16 → 0..15 (D-02), float32 cast, bbox.

Determinism (D-16)
------------------
* np.argsort(kind='stable') — deterministic order within equal keys.
* np.add.reduceat over the sorted array — summation order is the same on
  every run, so float64 sums are bitwise identical across runs.
* np.argmax returns first-max → smallest class ID wins on ties.
* Partition bin index is a pure function of the packed key (``key % K``).

Import order note: this module imports numpy + the sibling pcd_parser
module. It does NOT import the open3d library. Callers that use open3d must
bring it in BEFORE importing this module (see RESEARCH §I.5).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_pre.zaha.utils.pcd_parser import stream_pcd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID: float = 0.02  # locked — NOT a parameter
_INV: float = 1.0 / GRID
_SHIFT: int = 1 << 20   # ±1 M voxels per axis = ±20 km at grid=0.02
_NUM_RAW_CLASSES: int = 17  # classes 0..16 in the raw schema


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VoxelBatch:
    """Per-bin / per-batch aggregate result."""

    packed_keys: np.ndarray       # uint64  (M,)
    centroid_xyz: np.ndarray      # float64 (M, 3)
    winner_class: np.ndarray      # int32   (M,)  — raw class in [0, 16]
    counts: np.ndarray            # int64   (M,)
    hist: np.ndarray              # int32   (M, 17)


@dataclass
class VoxelAggregateResult:
    """Whole-file aggregate result after VOID drop + remap."""

    centroid_xyz: np.ndarray             # float32 (N, 3) — post drop/remap
    segment: np.ndarray                  # int32   (N,)   — strictly in [0, 15]
    bbox_min: np.ndarray                 # float32 (3,)
    bbox_max: np.ndarray                 # float32 (3,)
    n_raw_points: int                    # sum over streamed rows
    n_voxels_pre_void_drop: int          # voxel count before D-01 drop
    n_voxels_post_void_drop: int         # voxel count after D-01 drop
    class_histogram_raw: dict            # raw 0..16 (includes VOID)
    class_histogram_final: dict          # remapped 0..15 (no VOID)


# ---------------------------------------------------------------------------
# Key packing
# ---------------------------------------------------------------------------

def pack_voxel_keys(
    ix: np.ndarray, iy: np.ndarray, iz: np.ndarray
) -> np.ndarray:
    """Pack (ix, iy, iz) int64 voxel coordinates into a single uint64 key.

    Shifts each axis by ``+_SHIFT = +2**20`` so negative voxel coords work;
    uses 21 bits per axis = ±1 M voxels per direction = ±20 km at grid=0.02.
    """
    x = (ix.astype(np.int64) + _SHIFT).astype(np.uint64)
    y = (iy.astype(np.int64) + _SHIFT).astype(np.uint64)
    z = (iz.astype(np.int64) + _SHIFT).astype(np.uint64)
    return (x << 42) | (y << 21) | z


def _compute_voxel_keys(xyz: np.ndarray) -> np.ndarray:
    """Compute packed voxel keys for an (N, 3) float xyz array."""
    ix = np.floor(xyz[:, 0] * _INV).astype(np.int64)
    iy = np.floor(xyz[:, 1] * _INV).astype(np.int64)
    iz = np.floor(xyz[:, 2] * _INV).astype(np.int64)
    return pack_voxel_keys(ix, iy, iz)


# ---------------------------------------------------------------------------
# In-RAM batch aggregate (for small clouds and per-bin aggregation)
# ---------------------------------------------------------------------------

def voxel_aggregate_batch(xyz: np.ndarray, cls: np.ndarray) -> VoxelBatch:
    """Deterministic single-batch voxel aggregate.

    Implementation: stable sort on packed keys → np.unique → np.add.reduceat
    for centroid sums → np.add.at histogram → np.argmax winner. Tie-break is
    smallest raw class ID (np.argmax returns the first max index → D-16).

    Peak memory: ~3x input. Safe for clouds up to ~10 M points on WSL.
    Used as the inner kernel of stream_voxel_aggregate per partition bin.
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3, "xyz must be (N, 3)"
    assert cls.ndim == 1 and len(cls) == len(xyz), "cls must be (N,) matching xyz"

    if len(xyz) == 0:
        return VoxelBatch(
            np.zeros(0, dtype=np.uint64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, _NUM_RAW_CLASSES), dtype=np.int32),
        )

    xyz_f64 = np.ascontiguousarray(xyz, dtype=np.float64)
    cls_i32 = np.ascontiguousarray(cls, dtype=np.int32)
    keys = _compute_voxel_keys(xyz_f64)

    # STABLE sort — determinism (D-16). Equal keys keep the input order,
    # so np.add.reduceat sums float64 coordinates in a deterministic order
    # across runs → bitwise-identical output for the same input.
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    xyz_s = xyz_f64[order]
    cls_s = cls_i32[order]

    uniq, starts, counts = np.unique(
        keys_s, return_index=True, return_counts=True
    )
    sx = np.add.reduceat(xyz_s[:, 0], starts)
    sy = np.add.reduceat(xyz_s[:, 1], starts)
    sz = np.add.reduceat(xyz_s[:, 2], starts)
    centroid = np.stack([sx, sy, sz], axis=1) / counts[:, None]  # float64

    # Per-voxel class histogram via fancy-index add.
    voxel_idx = np.repeat(np.arange(len(uniq), dtype=np.int32), counts)
    hist = np.zeros((len(uniq), _NUM_RAW_CLASSES), dtype=np.int32)
    # Clamp any out-of-range class values defensively (ZAHA schema is 0..16).
    cls_clamped = np.clip(cls_s, 0, _NUM_RAW_CLASSES - 1)
    np.add.at(hist, (voxel_idx, cls_clamped), 1)
    # np.argmax returns the FIRST max → smallest class ID wins on ties (D-16).
    winner = np.argmax(hist, axis=1).astype(np.int32)

    return VoxelBatch(
        packed_keys=uniq,
        centroid_xyz=centroid,
        winner_class=winner,
        counts=counts.astype(np.int64),
        hist=hist,
    )


# ---------------------------------------------------------------------------
# Hash-partitioned external sort (streaming whole-file aggregate)
# ---------------------------------------------------------------------------

def _record_dtype() -> np.dtype:
    """Structured dtype used for the per-bin temp file records.

    40 bytes per record:
        uint64 key + (f64 x, f64 y, f64 z) + int32 cls + 4 bytes pad
    """
    return np.dtype(
        [
            ("key", np.uint64),
            ("x", np.float64),
            ("y", np.float64),
            ("z", np.float64),
            ("cls", np.int32),
            ("_pad", np.int32),
        ]
    )


def stream_voxel_aggregate(
    pcd_path: Path,
    tmp_dir: Path,
    K: int = 16,
    chunksize: int = 2_000_000,
) -> VoxelAggregateResult:
    """Stream + hash-partition external-sort voxel aggregator.

    Pipeline
    --------
    Pass 1 (partition):
        for chunk in stream_pcd(pcd_path):
            compute voxel keys, hash-partition to K disk bins via ``key % K``.
    Pass 2 (per-bin aggregate):
        for bin in bins:
            load bin into RAM, voxel_aggregate_batch, append to concat list.
    Pass 3 (post-process):
        VOID drop (D-01) on winner == 0, remap winner - 1 (D-02), float32 cast.

    Parameters
    ----------
    pcd_path : Path
        ASCII ZAHA .pcd file.
    tmp_dir : Path
        Temp directory for bin files; will be cleaned up on both success and
        failure paths.
    K : int
        Number of hash-partition bins (default 16, per RESEARCH §B.2).
    chunksize : int
        Pandas chunk size passed to stream_pcd.

    Returns
    -------
    VoxelAggregateResult

    Notes
    -----
    * Peak RAM (K=16, 136.8 M input points): ~1.3-1.5 GB per RESEARCH §B.2.
    * Temp disk (K=16, 136.8 M input points): ~5.5 GB.
    * Deterministic: same PCD + same commit → bitwise-identical output.
    * Idempotent: if tmp_dir already exists it is wiped and recreated.
    """
    pcd_path = Path(pcd_path)
    tmp_dir = Path(tmp_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dtype = _record_dtype()
    writers = [
        open(tmp_dir / f"bin_{b:02d}.bin", "wb") for b in range(K)
    ]

    n_raw = 0
    raw_hist = np.zeros(_NUM_RAW_CLASSES, dtype=np.int64)

    try:
        # Pass 1: partition ------------------------------------------------
        try:
            for chunk in stream_pcd(pcd_path, chunksize=chunksize):
                xyz = chunk[["x", "y", "z"]].values  # float64
                cls = chunk["c"].values              # int32
                n_raw += len(xyz)

                # Raw (per-point, not per-voxel) class histogram.
                cls_clamped = np.clip(cls, 0, _NUM_RAW_CLASSES - 1)
                bc = np.bincount(cls_clamped, minlength=_NUM_RAW_CLASSES)
                raw_hist += bc[:_NUM_RAW_CLASSES].astype(np.int64)

                keys = _compute_voxel_keys(xyz)
                bins = (keys % np.uint64(K)).astype(np.int64)

                for b in range(K):
                    mask = bins == b
                    if not mask.any():
                        continue
                    rec = np.empty(int(mask.sum()), dtype=dtype)
                    rec["key"] = keys[mask]
                    rec["x"] = xyz[mask, 0]
                    rec["y"] = xyz[mask, 1]
                    rec["z"] = xyz[mask, 2]
                    rec["cls"] = cls[mask]
                    rec.tofile(writers[b])
        finally:
            for w in writers:
                w.close()

        # Pass 2: per-bin aggregate ---------------------------------------
        all_keys = []
        all_centroid = []
        all_winner = []
        total_voxels_pre = 0

        for b in range(K):
            bin_path = tmp_dir / f"bin_{b:02d}.bin"
            if not bin_path.exists() or bin_path.stat().st_size == 0:
                continue
            rec = np.fromfile(bin_path, dtype=dtype)
            xyz_b = np.stack([rec["x"], rec["y"], rec["z"]], axis=1)
            cls_b = rec["cls"]
            batch = voxel_aggregate_batch(xyz_b, cls_b)
            all_keys.append(batch.packed_keys)
            all_centroid.append(batch.centroid_xyz)
            all_winner.append(batch.winner_class)
            total_voxels_pre += len(batch.packed_keys)
    finally:
        # Always clean up temp bins (success AND failure paths).
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if total_voxels_pre == 0:
        raise ValueError(
            f"{pcd_path}: voxel aggregate produced 0 voxels"
        )

    keys_cat = np.concatenate(all_keys)
    centroid_cat = np.concatenate(all_centroid, axis=0)
    winner_cat = np.concatenate(all_winner)
    del keys_cat  # unused after concat; kept above only for clarity

    # Pass 3a: VOID drop (D-01) — drop voxels whose winner is raw class 0.
    keep = winner_cat != 0
    n_post = int(keep.sum())
    centroid_kept = centroid_cat[keep]
    winner_kept = winner_cat[keep]

    # Pass 3b: remap raw 1..16 → remapped 0..15 (D-02).
    segment = (winner_kept - 1).astype(np.int32)
    if n_post > 0:
        assert int(segment.min()) >= 0 and int(segment.max()) <= 15, (
            f"{pcd_path}: segment out of range [0,15] — "
            f"got [{int(segment.min())},{int(segment.max())}]"
        )

    # Pass 3c: float32 cast + bbox (D-16).
    centroid_f32 = centroid_kept.astype(np.float32)
    if n_post > 0:
        bbox_min = centroid_f32.min(axis=0)
        bbox_max = centroid_f32.max(axis=0)
    else:
        bbox_min = np.zeros(3, dtype=np.float32)
        bbox_max = np.zeros(3, dtype=np.float32)

    # Final histogram (remapped space).
    if n_post > 0:
        final_bc = np.bincount(segment, minlength=16)
    else:
        final_bc = np.zeros(16, dtype=np.int64)
    final_hist = {int(k): int(v) for k, v in enumerate(final_bc[:16])}
    raw_hist_dict = {int(k): int(v) for k, v in enumerate(raw_hist)}

    return VoxelAggregateResult(
        centroid_xyz=centroid_f32,
        segment=segment,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        n_raw_points=n_raw,
        n_voxels_pre_void_drop=total_voxels_pre,
        n_voxels_post_void_drop=n_post,
        class_histogram_raw=raw_hist_dict,
        class_histogram_final=final_hist,
    )
