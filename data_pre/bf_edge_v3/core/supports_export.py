"""
Support element I/O and visualization export.

Extracted from supports_core.py during Phase 2 refactor.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from utils.common import (
    save_npz,
    save_xyz,
)


def export_npz(path: Path, payload: dict) -> None:
    """Export npz."""
    save_npz(path, payload)


def sample_segment_geometry(start: np.ndarray, end: np.ndarray, step: float) -> np.ndarray:
    """Sample one segment densely enough for visualization."""
    length = float(np.linalg.norm(end - start))
    num = max(int(np.ceil(length / max(step, 1e-4))) + 1, 2)
    alpha = np.linspace(0.0, 1.0, num=num, dtype=np.float32)
    return ((1.0 - alpha[:, None]) * start[None, :] + alpha[:, None] * end[None, :]).astype(np.float32)


def export_support_geometry_xyz(supports_payload: dict, output_dir: Path, sample_step: float = 0.02) -> None:
    """Export support geometry sampling for CloudCompare-style inspection."""
    rows = []
    for support_id in range(int(supports_payload["support_id"].shape[0])):
        support_type = int(supports_payload["support_type"][support_id])
        offset = int(supports_payload["segment_offset"][support_id])
        length = int(supports_payload["segment_length"][support_id])

        seed = (support_id * 1103515245 + 12345) & 0x7FFFFFFF
        color = np.array(
            [64 + (seed & 127), 64 + ((seed >> 8) & 127), 64 + ((seed >> 16) & 127)],
            dtype=np.float32,
        )

        for seg_idx in range(offset, offset + length):
            points = sample_segment_geometry(
                start=supports_payload["segment_start"][seg_idx],
                end=supports_payload["segment_end"][seg_idx],
                step=sample_step,
            )
            support_col = np.full((points.shape[0], 1), float(support_id), dtype=np.float32)
            type_col = np.full((points.shape[0], 1), float(support_type), dtype=np.float32)
            rows.append(
                np.concatenate(
                    [points, np.tile(color[None, :], (points.shape[0], 1)), support_col, type_col],
                    axis=1,
                )
            )

    if rows:
        data = np.concatenate(rows, axis=0).astype(np.float32)
    else:
        data = np.empty((0, 8), dtype=np.float32)

    save_xyz(
        output_dir / "support_geometry.xyz",
        data,
        ["%.6f", "%.6f", "%.6f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f"],
    )


def export_trigger_group_classes_xyz(debug_payload: dict, output_dir: Path) -> None:
    """Export trigger candidate subgroup classes for manual inspection."""
    data = debug_payload["trigger_group_visualization"]
    save_xyz(
        output_dir / "trigger_group_classes.xyz",
        data,
        ["%.6f", "%.6f", "%.6f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f"],
    )
