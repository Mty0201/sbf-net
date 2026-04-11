"""Per-chunk NPY writer with strict DS-ZAHA-P1-06 enforcement.

Writes the three NPY files the ZAHA dataset contract promises:

    <output>/<split>/<sample>__c<idx:03d>/
        coord.npy     # float32 (N, 3)    — post-denoise XYZ centroids
        segment.npy   # int32   (N,)      — remapped LoFG3 class in [0, 15]
        normal.npy    # float32 (N, 3)    — unit-length PCA normals

This module is pure numpy — no open3d, no pandas, no scipy. Safe to call from
any worker process regardless of earlier imports.

D-22 hard-failure policy: every dtype/shape/finiteness/range violation raises
``ValueError`` so the orchestrator exits non-zero with partial output
preserved for forensics. No silent repair, no coercion.

Invariants enforced
-------------------
* ``coord`` : ``(N, 3)`` ``float32``, finite. ``N >= 1`` — empty chunks must
  be dropped by the caller (D-03).
* ``segment`` : ``(N,)`` ``int32`` with every value in the closed interval
  ``[0, 15]`` (D-02 post-remap + D-01 VOID drop).
* ``normal`` : same shape/dtype as ``coord``, finite, unit-length with
  ``0.99 <= ||n|| <= 1.01`` (D-18 bar, matches ``normals.NormalConfig``
  default ``unit_length_threshold=0.99``).

The per-file SHA256 hashes returned by ``write_chunk_npys`` are the raw bytes
of the numpy arrays ``tobytes()`` (contiguous, little-endian, no NPY header).
Orchestrator code records them into ``manifest.json`` for the determinism
audit hook.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


#: Filenames emitted into every ``<sample>__c<idx>`` directory.
CHUNK_FILES: tuple[str, ...] = ("coord.npy", "segment.npy", "normal.npy")

#: Segment value bounds after D-01 VOID drop and D-02 remap.
_SEGMENT_MIN: int = 0
_SEGMENT_MAX: int = 15

#: Unit-length tolerance for normal vectors (D-18 bar).
_NORM_LO: float = 0.99
_NORM_HI: float = 1.01


def _sha256_array(arr: np.ndarray) -> str:
    """SHA256 of ``arr.tobytes()`` — used for the manifest audit hash.

    ``arr`` must be C-contiguous (numpy arrays produced by this module always
    are — either freshly cast float32/int32 from the orchestrator or numpy's
    own contiguous block). We explicitly call ``np.ascontiguousarray`` as a
    belt-and-suspenders measure so any future caller who passes a view / slice
    still gets a deterministic hash.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def write_chunk_npys(
    out_dir: Path,
    coord: np.ndarray,
    segment: np.ndarray,
    normal: np.ndarray,
) -> dict[str, Any]:
    """Write ``coord/segment/normal`` NPYs to ``out_dir`` with strict checks.

    Parameters
    ----------
    out_dir : Path
        Target directory. Will be created if missing.
    coord : np.ndarray
        ``(N, 3)`` ``float32`` point coordinates.
    segment : np.ndarray
        ``(N,)`` ``int32`` class labels in ``[0, 15]``.
    normal : np.ndarray
        ``(N, 3)`` ``float32`` unit-length normals.

    Returns
    -------
    dict
        ``{'point_count', 'coord_sha256', 'segment_sha256', 'normal_sha256',
           'segment_min', 'segment_max'}`` — consumed by the orchestrator when
        building ``manifest.json`` per D-20 / RESEARCH §H.2.

    Raises
    ------
    ValueError
        On any dtype / shape / finiteness / range / unit-length violation.
        D-22 hard-failure policy — the orchestrator must propagate this
        upward and exit non-zero with partial output preserved.
    """
    out_dir = Path(out_dir)

    # Shape + dtype enforcement --------------------------------------------
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(
            f"coord shape {tuple(coord.shape)}, expected (N, 3)"
        )
    if coord.dtype != np.float32:
        raise ValueError(
            f"coord dtype {coord.dtype}, expected float32"
        )
    n = int(coord.shape[0])
    if n < 1:
        raise ValueError(
            f"empty chunk at {out_dir} — caller must drop before write"
        )
    if segment.shape != (n,):
        raise ValueError(
            f"segment shape {tuple(segment.shape)}, expected ({n},)"
        )
    if segment.dtype != np.int32:
        raise ValueError(
            f"segment dtype {segment.dtype}, expected int32"
        )
    if normal.shape != coord.shape:
        raise ValueError(
            f"normal shape {tuple(normal.shape)}, expected "
            f"{tuple(coord.shape)}"
        )
    if normal.dtype != np.float32:
        raise ValueError(
            f"normal dtype {normal.dtype}, expected float32"
        )

    # Finiteness -----------------------------------------------------------
    if not np.isfinite(coord).all():
        raise ValueError(
            f"coord has non-finite values at {out_dir}"
        )
    if not np.isfinite(normal).all():
        raise ValueError(
            f"normal has non-finite values at {out_dir}"
        )

    # Segment range (D-02 / D-03) -----------------------------------------
    seg_min = int(segment.min())
    seg_max = int(segment.max())
    if seg_min < _SEGMENT_MIN or seg_max > _SEGMENT_MAX:
        raise ValueError(
            f"segment out of [{_SEGMENT_MIN}, {_SEGMENT_MAX}] at {out_dir}: "
            f"[{seg_min}, {seg_max}]"
        )

    # Normal unit-length (D-18) -------------------------------------------
    norms = np.linalg.norm(normal, axis=1)
    nmin = float(norms.min())
    nmax = float(norms.max())
    if nmin < _NORM_LO or nmax > _NORM_HI:
        raise ValueError(
            f"normal norms out of [{_NORM_LO}, {_NORM_HI}] at {out_dir}: "
            f"[{nmin:.4f}, {nmax:.4f}]"
        )

    # Write ----------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "coord.npy", coord, allow_pickle=False)
    np.save(out_dir / "segment.npy", segment, allow_pickle=False)
    np.save(out_dir / "normal.npy", normal, allow_pickle=False)

    return {
        "point_count": n,
        "coord_sha256": _sha256_array(coord),
        "segment_sha256": _sha256_array(segment),
        "normal_sha256": _sha256_array(normal),
        "segment_min": seg_min,
        "segment_max": seg_max,
    }
