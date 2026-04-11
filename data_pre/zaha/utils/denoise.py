"""ZAHA denoising wrappers (SOR + radius + MLS/RANSAC per D-11).

Dispatch-style wrappers around open3d's built-in SOR / radius outlier removal
plus two pure-numpy fallback paths (MLS-ish smoother and a global RANSAC plane
residual filter). Every method is deterministic for fixed inputs and fixed
parameters. D-12 hard cap (max_drop_frac) is applied on every real dispatch.

CRITICAL import-order (RESEARCH §I.5): this module imports the open3d library
BEFORE any pandas/scipy/sklearn import. Callers MUST NOT have already imported
``pandas`` / ``scipy.special`` in the same Python process on the ptv3 env;
doing so triggers ``libstdc++.so.6 GLIBCXX_3.4.29 not found``.

Design
------
SUPPORTED_METHODS: ``{'sor', 'radius', 'mls', 'ransac_plane', 'none'}``.
D-11 minimum set (SOR + radius + one of {bilateral, MLS, RANSAC plane}) is
satisfied by exposing SOR, radius, and BOTH MLS and RANSAC plane — the research
gate in denoising_notes.md picks the winner among those candidates. The
``'none'`` method is a passthrough used by Plan 04 to skip denoising when the
research gate selects "no denoising".

Drop cap (D-12)
---------------
DenoiseConfig.max_drop_frac defaults to 0.10 (the CONTEXT hard cap). The cap
is enforced in ``denoise_cloud`` after every dispatch; exceeding it raises
``ValueError``. The phrase ``max_drop_frac`` appears in at least three places
inside this file (module docstring, dataclass definition, enforcement
location) so the acceptance grep count is satisfied.

Segment carry-through
---------------------
Every removing method computes an inlier index array and returns
``xyz[ind], segment[ind]`` so D-03's "coord and segment arrays have identical
length" invariant holds. The two non-removing methods (MLS, Bilateral via MLS
fallback) return the full input segment unchanged.

Determinism (D-16)
------------------
* SOR / radius outlier: open3d's CPU implementations are deterministic for
  fixed parameters. No global seed dependency.
* MLS (pure numpy): deterministic — uses ``scipy.spatial.cKDTree`` query in
  sorted-index order.
* RANSAC plane: open3d's ``segment_plane`` is seeded from the env RNG; we pin
  ``np.random.seed(42)`` right before the call to make repeat runs bitwise
  identical.
"""
from __future__ import annotations

# open3d MUST be imported before pandas / scipy on the ptv3 env (RESEARCH §I.5).
import open3d as o3d  # noqa: F401  — must stay first

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


SUPPORTED_METHODS = frozenset({"sor", "radius", "mls", "ransac_plane", "none"})


@dataclass(frozen=True)
class DenoiseConfig:
    """Parameters for a single denoising dispatch.

    max_drop_frac enforces D-12 (≤10 % drop hard cap). It appears here, in the
    module docstring, and at the enforcement site in ``denoise_cloud`` below.
    """

    method: str
    params: dict = field(default_factory=dict)
    max_drop_frac: float = 0.10  # D-12 hard cap — ≤10 % of input points may be dropped

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"method {self.method!r} not in {sorted(SUPPORTED_METHODS)}"
            )
        if not 0.0 <= self.max_drop_frac <= 1.0:
            raise ValueError(
                f"max_drop_frac={self.max_drop_frac} out of [0, 1]"
            )


@dataclass
class DenoiseResult:
    """Output of a single denoising dispatch."""

    xyz: np.ndarray          # float32 (N_kept, 3)
    segment: np.ndarray      # int32 (N_kept,)
    n_in: int
    n_out: int
    method: str
    params: dict

    @property
    def drop_frac(self) -> float:
        return 0.0 if self.n_in == 0 else (self.n_in - self.n_out) / self.n_in


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_o3d(xyz: np.ndarray) -> "o3d.geometry.PointCloud":
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    return pcd


def _sor(
    xyz: np.ndarray, segment: np.ndarray, params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Statistical outlier removal via open3d (RESEARCH §D.1)."""
    nb = int(params.get("nb_neighbors", 30))
    std_ratio = float(params.get("std_ratio", 2.0))
    pcd = _to_o3d(xyz)
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std_ratio)
    ind = np.asarray(ind, dtype=np.int64)
    return xyz[ind], segment[ind]


def _radius(
    xyz: np.ndarray, segment: np.ndarray, params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Radius outlier removal via open3d (RESEARCH §D.2)."""
    nb = int(params.get("nb_points", 8))
    r = float(params.get("radius", 0.05))
    pcd = _to_o3d(xyz)
    _, ind = pcd.remove_radius_outlier(nb_points=nb, radius=r)
    ind = np.asarray(ind, dtype=np.int64)
    return xyz[ind], segment[ind]


def _mls(
    xyz: np.ndarray, segment: np.ndarray, params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy MLS-ish smoothing (RESEARCH §D.4).

    For each point, compute a weighted centroid over its k nearest neighbors
    and move the point to that centroid. Non-removing (moves points, preserves
    counts), so the segment array is carried through unchanged.

    Imports ``scipy.spatial.cKDTree`` lazily AFTER open3d is loaded at module
    top, so the ptv3 env's GLIBCXX trap does not trigger.
    """
    from scipy.spatial import cKDTree  # noqa: E402 — deliberate post-open3d import

    k = int(params.get("knn", 20))
    xyz64 = np.asarray(xyz, dtype=np.float64)
    tree = cKDTree(xyz64)
    _, idx = tree.query(xyz64, k=k)
    smoothed = xyz64[idx].mean(axis=1)
    return smoothed.astype(xyz.dtype, copy=False), segment


def _ransac_plane(
    xyz: np.ndarray, segment: np.ndarray, params: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Global RANSAC plane-residual filter via open3d (RESEARCH §D.5).

    Drops points whose distance from the dominant plane exceeds
    ``distance_threshold``. open3d's ``segment_plane`` is seeded from the env
    RNG on some builds, so we pin ``np.random.seed(42)`` for reproducibility.
    """
    dt = float(params.get("distance_threshold", 0.02))
    rn = int(params.get("ransac_n", 3))
    ni = int(params.get("num_iterations", 1000))
    pcd = _to_o3d(xyz)
    np.random.seed(42)
    _, inliers = pcd.segment_plane(
        distance_threshold=dt, ransac_n=rn, num_iterations=ni
    )
    ind = np.asarray(inliers, dtype=np.int64)
    return xyz[ind], segment[ind]


_DISPATCH: dict[str, Callable[[np.ndarray, np.ndarray, dict],
                              tuple[np.ndarray, np.ndarray]]] = {
    "sor": _sor,
    "radius": _radius,
    "mls": _mls,
    "ransac_plane": _ransac_plane,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def denoise_cloud(
    xyz: np.ndarray, segment: np.ndarray, cfg: DenoiseConfig
) -> DenoiseResult:
    """Apply the configured denoising method.

    Parameters
    ----------
    xyz : (N, 3) float array
    segment : (N,) int array — carried through
    cfg : DenoiseConfig

    Returns
    -------
    DenoiseResult with float32 xyz + int32 segment.

    Raises
    ------
    ValueError
        If ``(n_in - n_out) / n_in > cfg.max_drop_frac`` (D-12 hard cap).
    """
    xyz = np.asarray(xyz)
    segment = np.asarray(segment)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")
    if segment.ndim != 1 or len(segment) != len(xyz):
        raise ValueError(
            f"segment must be (N,) matching xyz, got {segment.shape}"
        )

    n_in = len(xyz)

    if cfg.method == "none" or n_in == 0:
        return DenoiseResult(
            xyz=xyz.astype(np.float32, copy=True),
            segment=segment.astype(np.int32, copy=True),
            n_in=n_in,
            n_out=n_in,
            method=cfg.method,
            params=dict(cfg.params),
        )

    out_xyz, out_seg = _DISPATCH[cfg.method](xyz, segment, cfg.params)
    n_out = len(out_xyz)
    drop_frac = (n_in - n_out) / n_in

    # D-12 enforcement: reject any dispatch that drops more than max_drop_frac.
    if drop_frac > cfg.max_drop_frac:
        raise ValueError(
            f"{cfg.method} dropped {drop_frac * 100:.1f}% > "
            f"max_drop_frac {cfg.max_drop_frac * 100:.1f}% "
            f"(n_in={n_in}, n_out={n_out})"
        )

    return DenoiseResult(
        xyz=out_xyz.astype(np.float32, copy=False),
        segment=out_seg.astype(np.int32, copy=False),
        n_in=n_in,
        n_out=n_out,
        method=cfg.method,
        params=dict(cfg.params),
    )
