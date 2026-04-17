"""Export semantic GT as colored XYZ point clouds for validation sets.

For each sample under <data_root>/<split>/, load coord.npy (Nx3) and
segment.npy (N,), map every class label to a fixed color, and write
<out_root>/<split>/<sample>.xyz  with lines "x y z R G B".

A legend.txt at <out_root>/legend.txt records the label -> (class_name, RGB)
mapping actually used for that dataset.

Usage:
    python scripts/viz/gt_color_export.py --dataset all --limit 1
    python scripts/viz/gt_color_export.py --dataset zaha
    python scripts/viz/gt_color_export.py --dataset bf --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "bf": {
        "data_root": "/home/mty0201/data/BF_edge_chunk_npy",
        "out_root": "/home/mty0201/data/BF_edge_chunk_npy_gt_viz",
        "split": "validation",
        "classes": [
            "balustrade",
            "balcony",
            "advboard",
            "wall",
            "eave",
            "column",
            "window",
            "clutter",
        ],
    },
    "s3dis": {
        "data_root": "/home/mty0201/data/s3dis",
        "out_root": "/home/mty0201/data/s3dis_gt_viz",
        "split": "validation",
        "classes": [
            "ceiling",
            "floor",
            "wall",
            "beam",
            "column",
            "window",
            "door",
            "table",
            "chair",
            "sofa",
            "bookcase",
            "board",
            "clutter",
        ],
    },
    "zaha": {
        "data_root": "/home/mty0201/data/ZAHA_chunked",
        "out_root": "/home/mty0201/data/ZAHA_chunked_gt_viz",
        "split": "validation",
        "classes": [
            "floor",
            "decoration",
            "structural",
            "opening",
            "other_el",
        ],
    },
}

# Shared distinct-color palette (Kelly's 22 + Glasbey extras), up to 32 classes.
# Index i always maps to the same color across datasets so class 0 of BF and
# class 0 of S3DIS would share a hue if they happened to align — here they do
# not, but the stability keeps legends reproducible.
PALETTE = np.array(
    [
        (230,  25,  75),  # 0  red
        ( 60, 180,  75),  # 1  green
        (255, 225,  25),  # 2  yellow
        (  0, 130, 200),  # 3  blue
        (245, 130,  48),  # 4  orange
        (145,  30, 180),  # 5  purple
        ( 70, 240, 240),  # 6  cyan
        (240,  50, 230),  # 7  magenta
        (210, 245,  60),  # 8  lime
        (250, 190, 212),  # 9  pink
        (  0, 128, 128),  # 10 teal
        (220, 190, 255),  # 11 lavender
        (170, 110,  40),  # 12 brown
        (255, 250, 200),  # 13 beige
        (128,   0,   0),  # 14 maroon
        (170, 255, 195),  # 15 mint
        (128, 128,   0),  # 16 olive
        (255, 215, 180),  # 17 apricot
        (  0,   0, 128),  # 18 navy
        (128, 128, 128),  # 19 grey
        (255, 255, 255),  # 20 white
        ( 64,  64,  64),  # 21 darkgrey
        (139,  69,  19),  # 22 saddlebrown
        ( 46, 139,  87),  # 23 seagreen
        (255, 105, 180),  # 24 hotpink
        ( 75,   0, 130),  # 25 indigo
        (255, 140,   0),  # 26 darkorange
        (  0, 191, 255),  # 27 deepskyblue
        (154, 205,  50),  # 28 yellowgreen
        (199,  21, 133),  # 29 mediumvioletred
        (184, 134,  11),  # 30 darkgoldenrod
        ( 47,  79,  79),  # 31 darkslategrey
    ],
    dtype=np.uint8,
)

IGNORE_COLOR = np.array([0, 0, 0], dtype=np.uint8)


def build_color_lut(num_classes: int) -> np.ndarray:
    """Return LUT of shape (num_classes, 3) uint8; extends palette by cycling."""
    if num_classes <= len(PALETTE):
        return PALETTE[:num_classes].copy()
    reps = (num_classes + len(PALETTE) - 1) // len(PALETTE)
    return np.tile(PALETTE, (reps, 1))[:num_classes]


def write_legend(out_root: Path, dataset_name: str, class_names: list[str], lut: np.ndarray) -> None:
    legend_path = out_root / "legend.txt"
    lines = [f"# Dataset: {dataset_name}", "# label  class_name  R  G  B"]
    for i, name in enumerate(class_names):
        r, g, b = lut[i]
        lines.append(f"{i:>3}  {name:<12s}  {r:>3d}  {g:>3d}  {b:>3d}")
    r, g, b = IGNORE_COLOR
    lines.append(f"{-1:>3}  {'ignore':<12s}  {r:>3d}  {g:>3d}  {b:>3d}")
    legend_path.parent.mkdir(parents=True, exist_ok=True)
    legend_path.write_text("\n".join(lines) + "\n")
    print(f"  legend -> {legend_path}")


def iter_sample_dirs(split_dir: Path) -> Iterable[Path]:
    if not split_dir.is_dir():
        return []
    return sorted(p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith("."))


def process_sample(
    sample_dir: Path,
    out_file: Path,
    lut: np.ndarray,
    num_classes: int,
    overwrite: bool,
) -> tuple[bool, str]:
    coord_p = sample_dir / "coord.npy"
    segment_p = sample_dir / "segment.npy"
    if not coord_p.exists() or not segment_p.exists():
        return False, f"missing coord/segment in {sample_dir.name}"

    if out_file.exists() and not overwrite:
        return False, f"exists (use --overwrite): {out_file.name}"

    coord = np.load(coord_p)
    segment = np.load(segment_p)
    if segment.ndim == 2 and segment.shape[1] == 1:
        segment = segment[:, 0]
    segment = segment.astype(np.int64)

    if coord.shape[0] != segment.shape[0]:
        return False, (
            f"shape mismatch in {sample_dir.name}: "
            f"coord={coord.shape}, segment={segment.shape}"
        )

    # Labels outside [0, num_classes) map to ignore (black).
    valid = (segment >= 0) & (segment < num_classes)
    rgb = np.empty((segment.shape[0], 3), dtype=np.uint8)
    rgb[valid] = lut[segment[valid]]
    rgb[~valid] = IGNORE_COLOR

    out_file.parent.mkdir(parents=True, exist_ok=True)
    arr = np.concatenate([coord.astype(np.float32), rgb.astype(np.int32)], axis=1)
    np.savetxt(
        out_file,
        arr,
        fmt=["%.4f", "%.4f", "%.4f", "%d", "%d", "%d"],
    )
    return True, f"{out_file.name} ({segment.shape[0]} pts)"


def process_dataset(name: str, spec: dict, limit: int | None, overwrite: bool) -> None:
    data_root = Path(spec["data_root"])
    out_root = Path(spec["out_root"])
    split = spec["split"]
    class_names = spec["classes"]
    num_classes = len(class_names)
    lut = build_color_lut(num_classes)

    split_dir = data_root / split
    out_split = out_root / split

    print(f"\n=== {name.upper()} ===")
    print(f"  data_root: {data_root}")
    print(f"  out_root:  {out_root}")
    print(f"  classes:   {num_classes}")

    if not split_dir.is_dir():
        print(f"  SKIP: split dir not found: {split_dir}")
        return

    write_legend(out_root, name, class_names, lut)

    samples = list(iter_sample_dirs(split_dir))
    if limit is not None:
        samples = samples[:limit]
    print(f"  samples:   {len(samples)}")

    n_ok = 0
    n_skip = 0
    for sample in samples:
        out_file = out_split / f"{sample.name}.xyz"
        ok, msg = process_sample(sample, out_file, lut, num_classes, overwrite)
        if ok:
            n_ok += 1
            print(f"  [ok]   {msg}")
        else:
            n_skip += 1
            print(f"  [skip] {msg}")

    print(f"  DONE {name}: wrote {n_ok}, skipped {n_skip}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        choices=["bf", "s3dis", "zaha", "all"],
        default="all",
    )
    parser.add_argument("--data-root", default=None, help="Override default data root")
    parser.add_argument("--out-root", default=None, help="Override default out root")
    parser.add_argument("--split", default=None, help="Override split (default: validation)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N samples per dataset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .xyz files")
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in targets:
        spec = dict(DATASETS[name])  # shallow copy
        if args.data_root and len(targets) == 1:
            spec["data_root"] = args.data_root
        if args.out_root and len(targets) == 1:
            spec["out_root"] = args.out_root
        if args.split:
            spec["split"] = args.split
        process_dataset(name, spec, args.limit, args.overwrite)

    return 0


if __name__ == "__main__":
    sys.exit(main())
