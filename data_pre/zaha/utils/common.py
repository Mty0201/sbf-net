"""Shared helpers for the ZAHA offline preprocessing pipeline.

Exports
-------
normalize_rows(arr, eps=1e-8) -> np.ndarray
    Row-wise unit-length normalize for (N, 3) arrays. Zero rows stay zero (no NaN).
sha256_file(path, max_bytes=1024) -> str
    Content hash of the first ``max_bytes`` of ``path``. Used by
    per_sample_state.json for partial-failure re-entry detection.
setup_logger(name, log_path=None) -> logging.Logger
    Idempotent stdout + optional file logger with a single format.

Import order note: this module imports numpy but NOT open3d. It is safe to
import before or after open3d.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np


def normalize_rows(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise unit-length normalize of an (N, 3) array.

    Rows whose norm is below ``eps`` are kept at zero (divisor clamped to 1.0)
    so the output has no NaN/Inf regardless of input degeneracy.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr.reshape(-1, 3).astype(arr.dtype, copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.where(norms < eps, 1.0, norms)
    return arr / safe_norms


def sha256_file(path: Path, max_bytes: int = 1024) -> str:
    """Content hash of the first ``max_bytes`` of ``path``.

    Used by per_sample_state.json to detect whether a raw PCD changed since
    the last (partial) pipeline run. Hashing the whole 6.9 GB ASCII file is
    unnecessary — the header + leading rows are enough to detect reruns with
    a modified input.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(max_bytes))
    return h.hexdigest()


def setup_logger(name: str, log_path: Path | None = None) -> logging.Logger:
    """Stdout (+ optional file) logger with a single ISO-8601 format.

    Idempotent: repeated calls with the same ``name`` do not stack handlers,
    so the pipeline orchestrator can call this per-sample without leaking
    log lines.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
