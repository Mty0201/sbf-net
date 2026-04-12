"""Chunking for ZAHA — facade-aware occupancy-based splitting.

Two chunking strategies are provided:

1. ``compute_chunks()`` — the legacy axis-aligned grid (D-06/D-08/D-10).
   Kept for CLI ``--tile-xy`` override and test compatibility.
2. ``compute_chunks_by_facade()`` — occupancy-based connected-component
   splitting driven by facade-class points only. This is the default
   path in ``build_zaha_chunks.py`` from Plan 01-04 Task 3 v3 onwards.

The facade chunker projects facade-class points (Wall, Window, Door,
Balcony, Molding, Deco, Column, Arch, Blinds — remapped IDs {0..7, 12})
onto a coarse XY occupancy grid, runs connected-component labelling, and
recursively bisects oversized components along their principal axis until
every component falls within the point budget.  Non-facade points (floor,
terrain, roof, interior, …) are assigned to the nearest facade component
by XY proximity so nothing is lost.

Import order (RESEARCH §I.5):
    this module is open3d-free and pandas-free. It imports only ``numpy``
    plus stdlib ``math`` / ``dataclasses`` / ``typing``, so it can be
    imported at any point in the pipeline without triggering the GLIBCXX
    ptv3 trap.

Exports
-------
ChunkingConfig         — frozen dataclass with tile_xy/overlap_xy/z_mode/budget
ChunkSpec              — frozen dataclass describing a single chunk box
compute_chunks()       — legacy deterministic row-major tile enumeration
compute_chunks_by_facade() — facade-aware occupancy chunking (default)
chunk_name()           — ``{basename}__c{idx:04d}`` naming per RESEARCH §F.6
iter_chunk_points()    — extract the (xyz, segment) subset falling in a chunk
MAX_CHUNK_INDEX        — 9999 (bumped by widening MAX_CHUNK_DIGITS if needed)
FACADE_CLASS_IDS       — remapped LoFG3 IDs used for footprint extraction
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Naming constants (RESEARCH §F.2 / F.6)
# ---------------------------------------------------------------------------

MAX_CHUNK_DIGITS: int = 4
MAX_CHUNK_INDEX: int = 10 ** MAX_CHUNK_DIGITS - 1  # 9999

# Remapped LoFG3 class IDs that define the building facade.
# Wall(0), Window(1), Door(2), Balcony(3), Molding(4),
# Deco(5), Column(6), Arch(7), Blinds(12).
# These classes drive the XY footprint extraction in the facade chunker.
# Floor/Terrain/Roof/Interior are excluded so the footprint reflects
# the vertical-surface outline rather than the ground plane.
FACADE_CLASS_IDS: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 12})

# Default occupancy grid cell size for facade footprint extraction.
OCCUPANCY_GRID_M: float = 1.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking configuration.

    Parameters
    ----------
    tile_xy : float
        XY tile edge length in metres. Default 4.0 matches the RESEARCH §F.4
        worked example starting budget; the planner refines this after the
        density measurement pass in ``measure_density.py``.
    overlap_xy : float
        XY overlap between adjacent tiles in metres. Minimum 2.0 per D-06.
        Must be strictly less than ``tile_xy`` (stride = tile - overlap > 0).
    z_mode : str
        ``'full'`` (default) → single Z tile spanning the full bbox Z extent.
        ``'band:{depth}'`` (e.g. ``'band:6.0'``) → slice into ``depth``-metre
        Z bands with no Z-overlap. Fallback for buildings where a
        ``tile_xy × tile_xy × Z_full`` column exceeds ``budget_per_chunk``.
    budget_per_chunk : int
        Maximum points per chunk post-denoise (D-07). The chunker does NOT
        enforce this — Plan 04 detects over-budget tiles and falls back to
        ``z_mode='band:{depth}'``. Default 1_000_000 (D-07 supersession
        2026-04-12, Plan 01-04 Task 3): the original 600k cap was set against
        the grid=0.02 + fixed-4m-tile regime; under grid=0.04 + adaptive
        continuous sizing with TARGET_PTS=400k, chunks land in the 200k-700k
        range and the 1M cap is the true "fail only if something is wrong"
        bound rather than a routine squeeze point.
    """

    tile_xy: float = 4.0
    overlap_xy: float = 2.0
    z_mode: str = "full"
    budget_per_chunk: int = 1_000_000

    def __post_init__(self) -> None:
        if self.tile_xy <= 0:
            raise ValueError(f"tile_xy must be > 0, got {self.tile_xy}")
        if self.overlap_xy < 0 or self.overlap_xy >= self.tile_xy:
            raise ValueError(
                f"overlap_xy must be in [0, tile_xy), got {self.overlap_xy}"
            )
        if self.z_mode != "full" and not self.z_mode.startswith("band:"):
            raise ValueError(
                f'z_mode must be "full" or "band:{{depth}}", got {self.z_mode!r}'
            )
        if self.z_mode.startswith("band:"):
            try:
                depth = float(self.z_mode.split(":", 1)[1])
            except ValueError as exc:  # pragma: no cover — malformed input
                raise ValueError(
                    f"z_mode band depth must be a float, got {self.z_mode!r}"
                ) from exc
            if depth <= 0:
                raise ValueError(
                    f"z_mode band depth must be > 0, got {depth}"
                )
        if self.budget_per_chunk <= 0:
            raise ValueError(
                f"budget_per_chunk must be > 0, got {self.budget_per_chunk}"
            )

    @property
    def stride_xy(self) -> float:
        """Centre-to-centre XY stride between adjacent tiles."""
        return self.tile_xy - self.overlap_xy


@dataclass(frozen=True)
class ChunkSpec:
    """A single tile in the deterministic box-grid chunker.

    Attributes
    ----------
    chunk_idx : int
        Linear row-major index. ``chunks[c].chunk_idx == c`` after
        ``compute_chunks``.
    x_tile, y_tile : int
        Logical tile coordinates in the (x_outer, y_inner) grid. In
        ``z_mode='full'`` there is one ChunkSpec per (x_tile, y_tile); in
        ``z_mode='band:{depth}'`` each (x_tile, y_tile) expands to one
        ChunkSpec per Z band.
    bbox_min, bbox_max : tuple[float, float, float]
        Inclusive min / inclusive max corner of the tile in world
        coordinates. ``iter_chunk_points`` uses a closed interval on both
        sides so last-tile points at ``bbox_max`` are never dropped.
    """

    chunk_idx: int
    x_tile: int
    y_tile: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]


# ---------------------------------------------------------------------------
# Naming (RESEARCH §F.6)
# ---------------------------------------------------------------------------

def chunk_name(sample_basename: str, chunk_idx: int) -> str:
    """Build the ``{basename}__c{idx:04d}`` chunk directory name.

    The 4-digit zero-pad is locked per D-10 and §F.6 — bumped from 3→4 in
    Plan 01-04 Task 3 (2026-04-12) after ``--z-mode band:6.0`` multiplied
    chunk counts beyond the old 999 ceiling on large footprint samples.
    Widening further requires bumping ``MAX_CHUNK_DIGITS`` together with
    any downstream manifest parsers.
    """
    if chunk_idx < 0 or chunk_idx > MAX_CHUNK_INDEX:
        raise ValueError(
            f"chunk_idx {chunk_idx} out of range [0, {MAX_CHUNK_INDEX}] — "
            f"bump MAX_CHUNK_DIGITS if ZAHA grows beyond 10000 chunks/sample"
        )
    return f"{sample_basename}__c{chunk_idx:0{MAX_CHUNK_DIGITS}d}"


# ---------------------------------------------------------------------------
# Tile edge enumeration (one axis)
# ---------------------------------------------------------------------------

def _tile_edges(
    axis_min: float, axis_max: float, tile: float, overlap: float
) -> list[tuple[float, float]]:
    """Enumerate ``(start, end)`` edges along one axis.

    Semantics
    ---------
    * ``axis_max - axis_min <= tile``: single tile spanning ``[min, max]``.
      Overlap is irrelevant in this degenerate case.
    * ``axis_max - axis_min >  tile``: step by ``stride = tile - overlap``
      starting at ``axis_min``. The last tile is clipped so its ``end``
      equals ``axis_max``; its ``start`` is pulled back by up to ``tile`` so
      the last tile still has full width when possible.

    Returns
    -------
    list[tuple[float, float]]
        Strictly ordered by ``start``, each pair satisfies ``start < end``.
    """
    stride = tile - overlap
    if stride <= 0:
        raise ValueError(
            f"non-positive stride={stride} (tile={tile}, overlap={overlap})"
        )
    span = axis_max - axis_min
    if span <= tile:
        return [(float(axis_min), float(axis_max))]

    # n_tiles such that (n_tiles - 1) * stride + tile >= span
    n_tiles = int(math.ceil((span - tile) / stride)) + 1
    edges: list[tuple[float, float]] = []
    for i in range(n_tiles):
        start = axis_min + i * stride
        end = start + tile
        if i == n_tiles - 1:
            # Clip last tile so its end is exactly axis_max; keep full width
            # by pulling start back if possible.
            end = axis_max
            start = min(start, axis_max - tile)
        edges.append((float(start), float(end)))
    return edges


# ---------------------------------------------------------------------------
# Row-major (x-outer, y-inner) tile enumeration (D-10)
# ---------------------------------------------------------------------------

def compute_chunks(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    cfg: ChunkingConfig,
) -> list[ChunkSpec]:
    """Deterministic row-major (x-outer, y-inner) tile enumeration.

    Parameters
    ----------
    bbox_min, bbox_max : array-like of shape (3,)
        Post-denoise cloud's bbox (D-08). Cast internally to float64 for
        numerical stability of the stride calculation; ChunkSpec stores the
        result as Python floats.
    cfg : ChunkingConfig

    Returns
    -------
    list[ChunkSpec]
        Length ``n_x * n_y`` for ``z_mode='full'`` or ``n_x * n_y * n_z``
        for ``z_mode='band:{depth}'``.

        Guarantees:
          * deterministic order (same inputs → same output on every run);
          * ``chunks[c].chunk_idx == c`` for all ``c``;
          * row-major x-outer, y-inner iteration: first N_y chunks span the
            first x row, next N_y chunks span the second x row, etc.;
          * adjacent (x or y) tiles share at least ``cfg.overlap_xy`` metres
            in the respective axis (or the whole bbox if only one tile fits);
          * the last tile on each axis has its high edge clipped to
            ``bbox_max``.
    """
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    if bbox_min.shape != (3,) or bbox_max.shape != (3,):
        raise ValueError(
            f"bbox must be (3,)/(3,), got {bbox_min.shape}/{bbox_max.shape}"
        )
    if not np.all(bbox_max > bbox_min):
        raise ValueError(
            f"degenerate bbox: min={bbox_min.tolist()} max={bbox_max.tolist()}"
        )

    x_edges = _tile_edges(
        float(bbox_min[0]), float(bbox_max[0]), cfg.tile_xy, cfg.overlap_xy
    )
    y_edges = _tile_edges(
        float(bbox_min[1]), float(bbox_max[1]), cfg.tile_xy, cfg.overlap_xy
    )

    if cfg.z_mode == "full":
        z_edges = [(float(bbox_min[2]), float(bbox_max[2]))]
    else:
        depth = float(cfg.z_mode.split(":", 1)[1])
        # Z-band has no overlap; each band fully tiles the Z extent.
        z_edges = _tile_edges(
            float(bbox_min[2]), float(bbox_max[2]), depth, 0.0
        )

    n_y = len(y_edges)
    n_z = len(z_edges)

    chunks: list[ChunkSpec] = []
    linear_idx = 0
    # D-10: x-outer, y-inner.
    for ix, (x0, x1) in enumerate(x_edges):
        for iy, (y0, y1) in enumerate(y_edges):
            for iz, (z0, z1) in enumerate(z_edges):
                chunks.append(
                    ChunkSpec(
                        chunk_idx=linear_idx,
                        x_tile=ix,
                        y_tile=iy,
                        bbox_min=(x0, y0, z0),
                        bbox_max=(x1, y1, z1),
                    )
                )
                linear_idx += 1

    # Silence lint for unused helpers in case of a single-z single-tile case.
    _ = n_y, n_z
    return chunks


# ---------------------------------------------------------------------------
# Chunk point extraction
# ---------------------------------------------------------------------------

def iter_chunk_points(
    xyz: np.ndarray, segment: np.ndarray, chunk: ChunkSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xyz, segment)`` subset whose coords fall inside ``chunk``.

    Uses closed intervals on all three axes (``bbox_min <= p <= bbox_max``)
    so points on the clipped last-tile edge are not lost. Because overlap
    regions are not deduped (D-09), adjacent chunks will legitimately share
    the same points.

    Parameters
    ----------
    xyz : np.ndarray
        ``(N, 3)`` float array.
    segment : np.ndarray
        ``(N,)`` int array.
    chunk : ChunkSpec

    Returns
    -------
    xyz_sub : np.ndarray
        ``(M, 3)`` float32 view/copy of the points inside ``chunk``.
    seg_sub : np.ndarray
        ``(M,)`` int32 view/copy of the matching labels.
    """
    xyz = np.asarray(xyz)
    segment = np.asarray(segment)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")
    if segment.ndim != 1 or len(segment) != len(xyz):
        raise ValueError(
            f"segment must be (N,) matching xyz, got {segment.shape}"
        )

    mn = chunk.bbox_min
    mx = chunk.bbox_max
    mask = (
        (xyz[:, 0] >= mn[0]) & (xyz[:, 0] <= mx[0])
        & (xyz[:, 1] >= mn[1]) & (xyz[:, 1] <= mx[1])
        & (xyz[:, 2] >= mn[2]) & (xyz[:, 2] <= mx[2])
    )
    return (
        xyz[mask].astype(np.float32, copy=False),
        segment[mask].astype(np.int32, copy=False),
    )


# ---------------------------------------------------------------------------
# Facade-aware occupancy-based chunking
# ---------------------------------------------------------------------------


def _xy_to_grid(
    xy: np.ndarray, origin: np.ndarray, cell: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert XY coords to integer grid indices."""
    gx = ((xy[:, 0] - origin[0]) / cell).astype(np.int32)
    gy = ((xy[:, 1] - origin[1]) / cell).astype(np.int32)
    return gx, gy


def _bisect_indices(
    xy: np.ndarray, indices: np.ndarray, budget: int, min_pts: int,
) -> list[np.ndarray]:
    """Recursively bisect a point set along its longest XY axis until
    each piece has <= ``budget`` points. Pieces smaller than ``min_pts``
    are kept as-is (never split further)."""
    if len(indices) <= budget:
        return [indices]
    pts = xy[indices]
    spans = pts.max(axis=0) - pts.min(axis=0)
    axis = int(np.argmax(spans))
    med = float(np.median(pts[:, axis]))
    left_mask = pts[:, axis] <= med
    left_idx = indices[left_mask]
    right_idx = indices[~left_mask]
    if len(left_idx) < min_pts or len(right_idx) < min_pts:
        return [indices]
    return _bisect_indices(xy, left_idx, budget, min_pts) + \
           _bisect_indices(xy, right_idx, budget, min_pts)


def compute_chunks_by_facade(
    xyz: np.ndarray,
    segment: np.ndarray,
    budget: int = 1_000_000,
    min_pts: int = 10_000,
    cell: float = OCCUPANCY_GRID_M,
    facade_ids: frozenset[int] = FACADE_CLASS_IDS,
) -> list[np.ndarray]:
    """Split a point cloud into chunks guided by facade-class XY footprint.

    Algorithm
    ---------
    1. Extract facade-class points and project onto a coarse XY grid.
    2. Build a binary occupancy image and label connected components.
    3. Assign ALL points (including non-facade) to the nearest component
       by XY cell proximity.
    4. Recursively bisect oversized components along their longest XY axis
       until every piece has <= ``budget`` points.
    5. Drop pieces with < ``min_pts`` points.

    Parameters
    ----------
    xyz : (N, 3) float32
    segment : (N,) int32 — remapped LoFG3 class IDs
    budget : max points per chunk
    min_pts : minimum points to keep a chunk
    cell : occupancy grid resolution in metres
    facade_ids : set of remapped class IDs considered facade

    Returns
    -------
    list[np.ndarray]
        Each entry is an index array into xyz/segment for one chunk.
        Chunks are sorted by the X coordinate of their centroid.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    segment = np.asarray(segment, dtype=np.int32)
    N = len(xyz)
    xy = xyz[:, :2]

    # --- Step 1: facade mask + occupancy grid ---
    facade_mask = np.isin(segment, list(facade_ids))
    origin = xy.min(axis=0)

    # Grid dimensions
    extent = xy.max(axis=0) - origin
    nx = max(1, int(np.ceil(extent[0] / cell)) + 1)
    ny = max(1, int(np.ceil(extent[1] / cell)) + 1)

    # Facade occupancy image
    occ = np.zeros((nx, ny), dtype=np.int32)
    if facade_mask.any():
        fxy = xy[facade_mask]
        fgx, fgy = _xy_to_grid(fxy, origin, cell)
        fgx = np.clip(fgx, 0, nx - 1)
        fgy = np.clip(fgy, 0, ny - 1)
        occ[fgx, fgy] = 1

    # --- Step 2: connected components ---
    labeled, n_components = ndimage.label(occ)

    if n_components == 0:
        # No facade points at all — single chunk with everything.
        return [np.arange(N)]

    # --- Step 3: assign ALL points to nearest component ---
    all_gx, all_gy = _xy_to_grid(xy, origin, cell)
    all_gx = np.clip(all_gx, 0, nx - 1)
    all_gy = np.clip(all_gy, 0, ny - 1)

    point_labels = labeled[all_gx, all_gy]  # 0 = unassigned (no facade cell)

    # Points landing on unoccupied cells: assign to nearest occupied cell.
    unassigned = point_labels == 0
    if unassigned.any():
        # Distance transform gives the distance to nearest occupied cell;
        # we need the label of that nearest cell.
        # Use scipy distance_transform_edt with return_indices.
        _, nearest_idx = ndimage.distance_transform_edt(
            labeled == 0, return_distances=True, return_indices=True,
        )
        near_gx = nearest_idx[0][all_gx[unassigned], all_gy[unassigned]]
        near_gy = nearest_idx[1][all_gx[unassigned], all_gy[unassigned]]
        point_labels[unassigned] = labeled[near_gx, near_gy]

    # Any still at 0 (whole image is empty — shouldn't happen) gets label 1.
    point_labels[point_labels == 0] = 1

    # --- Step 4: collect per-component indices, bisect if over budget ---
    result: list[np.ndarray] = []
    for comp_id in range(1, n_components + 1):
        comp_indices = np.where(point_labels == comp_id)[0]
        if len(comp_indices) < min_pts:
            continue
        pieces = _bisect_indices(xy, comp_indices, budget, min_pts)
        for piece in pieces:
            if len(piece) >= min_pts:
                result.append(piece)

    # --- Step 5: sort by centroid X for deterministic ordering ---
    result.sort(key=lambda idx: float(xy[idx, 0].mean()))

    return result
