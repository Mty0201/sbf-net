"""Shared pytest fixtures for the ZAHA offline preprocessing test suite.

Fixtures:
- ``synthetic_pcd_fixture(tmp_path)`` — writes a 1000-point ASCII PCD with a
  valid ZAHA-style PCL v0.7 header and classifications cycling ``[0..16]``.
  Does not depend on ``/home/mty0201/data/``.
- ``real_zaha_root`` — returns ``Path("/home/mty0201/data/ZAHA_pcd")`` guarded
  by ``pytest.skip`` if the real dataset is absent (smoke-only tests).
- ``phase_dir`` — returns the workstream phase directory relative to ``cwd``
  for loading ``lofg3_to_lofg2.yaml`` and friends.

The PCD header is byte-identical to the ZAHA release format verified in
``01-RESEARCH.md §A.1`` (three sample files cross-checked).
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Synthetic 1000-point PCD writer
# ---------------------------------------------------------------------------

def _write_pcd(
    path: Path,
    n_points: int = 1000,
    data_mode: str = "ascii",
    rgb_filler: int = 13033652,
) -> Path:
    """Write a minimal ZAHA-shape PCD file.

    Points lie on a 10x10x10 grid spanning ``[0, 10)`` on each axis; classifications
    cycle through ``[0, 16]`` so the test suite sees every class at least 58 times
    (1000 / 17 = 58.8). ``rgb`` is set to a constant filler to prove the rgb-drop
    path in the parser removes it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z classification rgb\n"
        "SIZE 4 4 4 1 4\n"
        "TYPE F F F U U\n"
        "COUNT 1 1 1 1 1\n"
        f"WIDTH {n_points}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n_points}\n"
        f"DATA {data_mode}\n"
    )

    lines = [header]
    for i in range(n_points):
        x = float(i % 10)
        y = float((i // 10) % 10)
        z = float((i // 100) % 10)
        cls = i % 17  # cycles 0..16, hits VOID (0) and OuterCeilingSurface (14)
        lines.append(f"{x:.4f} {y:.4f} {z:.4f} {cls} {rgb_filler}\n")

    with path.open("w") as fh:
        fh.writelines(lines)
    return path


@pytest.fixture
def synthetic_pcd_fixture(tmp_path: Path) -> Path:
    """1000-point synthetic ASCII PCD with valid ZAHA v0.7 header."""
    return _write_pcd(tmp_path / "synthetic.pcd", n_points=1000)


@pytest.fixture
def synthetic_pcd_binary_fixture(tmp_path: Path) -> Path:
    """1-line synthetic PCD with ``DATA binary`` to exercise the rejection path."""
    path = tmp_path / "synthetic_binary.pcd"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z classification rgb\n"
        "SIZE 4 4 4 1 4\n"
        "TYPE F F F U U\n"
        "COUNT 1 1 1 1 1\n"
        "WIDTH 1\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS 1\n"
        "DATA binary\n"
    )
    path.write_text(header + "\x00\x00\x00\x00" * 4)
    return path


# ---------------------------------------------------------------------------
# Real-data guard
# ---------------------------------------------------------------------------

@pytest.fixture
def real_zaha_root() -> Path:
    """Path to the real ZAHA PCD dataset, skip if absent."""
    root = Path("/home/mty0201/data/ZAHA_pcd")
    if not root.exists():
        pytest.skip(f"real ZAHA root not present at {root}")
    return root


# ---------------------------------------------------------------------------
# Workstream phase directory
# ---------------------------------------------------------------------------

@pytest.fixture
def phase_dir() -> Path:
    """Workstream Phase 1 directory (relative to cwd)."""
    return Path(
        ".planning/workstreams/dataset-handover-s3dis-chas/phases/"
        "01-zaha-offline-preprocessing-pipeline"
    )
