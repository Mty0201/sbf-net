from __future__ import annotations

from pathlib import Path

import numpy as np


EPS = 1e-8


def normalize_vector(vec: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Normalize one vector. Returns zeros when the norm is too small."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros(3, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def normalize_rows(arr: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Normalize vectors row-wise."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(-1, 3).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (arr / norms).astype(np.float32)


def save_xyz(path: Path, data: np.ndarray, fmt: list[str]) -> None:
    """Save whitespace-delimited xyz/txt style output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), data, fmt=fmt, delimiter=" ")


def save_npz(path: Path, payload: dict) -> None:
    """Export compressed npz payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
