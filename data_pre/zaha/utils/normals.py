"""Adaptive-radius PCA normals for ZAHA chunks (DS-ZAHA-P1-05).

CRITICAL import-order (RESEARCH §I.5):
    this module imports the open3d library BEFORE numpy. In the ptv3 conda
    env, ``import open3d`` must run before anything that pulls in
    ``pandas`` / ``scipy.special`` / ``sklearn`` — otherwise the loader hits
    ``libstdc++.so.6 GLIBCXX_3.4.29 not found``. Callers importing this
    module from a process that has already imported pandas or scipy will
    trip the same trap. See ``data_pre/zaha/docs/README.md`` for details.

Design
------
* D-17 default: ``open3d.geometry.KDTreeSearchParamKNN(knn=30)``. Open3d's
  C++ kernel does PCA of the 30-nearest-neighbour covariance matrix and
  returns the smallest-eigenvalue eigenvector (Hoppe-92). This is
  mathematically equivalent to "adaptive radius holding exactly k=30
  neighbours".
* D-18 bar: unit-length float32 ``(N, 3)`` + no NaN. Any non-finite output
  or any normal whose magnitude is below ``NormalConfig.unit_length_threshold``
  after a belt-and-suspenders renormalisation raises ``ValueError`` — the
  caller (Plan 04 per-sample orchestration) must drop the chunk or widen
  ``knn``, rather than silently emitting NaN.
* D-18 / §E.5: orientation is NOT propagated. The downstream BFDataset
  treats normals as an unsigned feature vector; propagating sign via
  ``orient_normals_consistent_tangent_plane`` would add ~20 s per 500k
  chunk × many chunks with no training benefit.
* D-19: normals are computed per chunk (not per whole building). This
  module does not know about chunks — it takes an ``(N, 3)`` array and
  returns ``(N, 3)`` normals. The per-chunk invocation is Plan 04's job.

Exports
-------
NormalConfig        — knn, orient, fast, unit_length_threshold
estimate_normals    — the single entry point
"""
from __future__ import annotations

# open3d MUST be imported before numpy on the ptv3 env — do NOT reorder.
import open3d as o3d  # noqa: F401  — must stay first

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalConfig:
    """Adaptive-radius PCA normal estimation parameters.

    Parameters
    ----------
    knn : int
        Number of nearest neighbours for PCA (D-17 default 30). Must be
        >= 3 for the covariance matrix to have a non-degenerate normal
        direction.
    orient : bool
        Whether to propagate sign via
        ``pcd.orient_normals_consistent_tangent_plane(k=knn)``. Defaults to
        ``False`` per §E.5 — downstream BFDataset does not depend on sign.
    fast : bool
        Whether to use open3d's fast (approximate) normal computation.
        Defaults to ``False`` so the full eigendecomposition is used.
    unit_length_threshold : float
        Minimum ``||n||`` a normal may have before triggering a renormalise
        + raise cycle. Default 0.99 — open3d returns unit vectors by
        construction, so this is a belt-and-suspenders check that catches
        non-finite entries and true collinear degeneracies.
    """

    knn: int = 30
    orient: bool = False
    fast: bool = False
    unit_length_threshold: float = 0.99

    def __post_init__(self) -> None:
        if self.knn < 3:
            raise ValueError(f"knn must be >= 3, got {self.knn}")
        if not 0.0 < self.unit_length_threshold <= 1.0:
            raise ValueError(
                f"unit_length_threshold out of (0, 1]: {self.unit_length_threshold}"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_normals(
    xyz: np.ndarray, cfg: NormalConfig | None = None
) -> np.ndarray:
    """Adaptive-radius PCA normals via open3d ``KDTreeSearchParamKNN``.

    Parameters
    ----------
    xyz : np.ndarray
        ``(N, 3)`` array of points. ``float32`` or ``float64`` accepted;
        internally cast to ``float64`` for open3d's point cloud API.
    cfg : NormalConfig, optional
        Defaults to ``NormalConfig()`` (knn=30, orient=False, fast=False).

    Returns
    -------
    normals : np.ndarray
        ``(N, 3)`` float32, unit length, finite. Every row satisfies
        ``np.linalg.norm(normals[i]) >= cfg.unit_length_threshold`` after a
        final renormalisation pass.

    Raises
    ------
    ValueError
        * ``xyz`` shape is not ``(N, 3)``.
        * ``N < cfg.knn`` — open3d's KNN PCA needs at least ``knn`` points.
          Caller must split the chunk or reduce ``knn``.
        * open3d returned non-finite normals.
        * any row's ``||n||`` remains below ``unit_length_threshold`` after
          renormalisation (true collinear / isolated-point degeneracy).
    """
    if cfg is None:
        cfg = NormalConfig()

    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")

    n = len(xyz)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if n < cfg.knn:
        raise ValueError(
            f"chunk has {n} points < knn={cfg.knn}; "
            "split chunk or reduce knn"
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=cfg.knn),
        fast_normal_computation=cfg.fast,
    )
    if cfg.orient:
        pcd.orient_normals_consistent_tangent_plane(k=cfg.knn)

    normals = np.asarray(pcd.normals, dtype=np.float32)
    if normals.shape != (n, 3):
        raise ValueError(
            f"open3d returned shape {normals.shape}, expected {(n, 3)}"
        )

    if not np.isfinite(normals).all():
        raise ValueError(
            f"{int(np.sum(~np.isfinite(normals)))} non-finite normals — "
            "degenerate neighborhood or open3d internal failure"
        )

    norms = np.linalg.norm(normals, axis=1)
    bad_mask = norms < cfg.unit_length_threshold
    if int(bad_mask.sum()) > 0:
        # Belt-and-suspenders renormalise. open3d is supposed to return
        # unit vectors — but on a degenerate neighbourhood the kernel can
        # emit ``[0, 0, 1]`` with norm 1 (handled by the >=0.99 check) or
        # an under-length vector near the numerical floor (handled here).
        safe = np.where(norms > 1e-8, norms, 1.0)
        normals = (normals / safe[:, None]).astype(np.float32)
        norms2 = np.linalg.norm(normals, axis=1)
        bad_mask2 = norms2 < cfg.unit_length_threshold
        if bad_mask2.any():
            raise ValueError(
                f"{int(bad_mask2.sum())} normals have ||n|| < "
                f"{cfg.unit_length_threshold} even after renormalisation — "
                "true degeneracy (collinear / isolated). Caller must drop "
                "the chunk or widen knn."
            )

    return normals
