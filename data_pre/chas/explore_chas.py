"""
CHAS dataset exploration: voxel downsample + extract unique GT colors.

Usage:
    python data_pre/chas/explore_chas.py
"""
import os
import numpy as np
import open3d as o3d
from pathlib import Path
from collections import Counter

CHAS_ROOT = Path("/mnt/e/WSL/data/CHAS")
OUT_DIR = CHAS_ROOT / "downsampled_0.1m"

# All 5 scenes and their GT file names
SCENES = {
    "baile":    {"raw": "baile.ply",      "gt": "gt_baile.ply"},
    "boa_morte":{"raw": "boa_morte.ply",  "gt": "boa_morte.ply"},
    "curia":    {"raw": "curia.ply",      "gt": "curia.ply"},
    "engenho":  {"raw": "engenho.ply",    "gt": "gt_engenho.ply"},
    "quilombo": {"raw": "quilombo.ply",   "gt": "gt_quilombo.ply"},
}

VOXEL_SIZE = 0.1  # meters


def downsample_and_analyze(name: str, info: dict):
    """Downsample raw + GT, extract unique GT colors."""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    out_scene = OUT_DIR / name
    out_scene.mkdir(parents=True, exist_ok=True)

    # --- GT first (smaller, has labels) ---
    gt_path = CHAS_ROOT / "gt" / info["gt"]
    print(f"  Loading GT: {gt_path} ...")
    gt_pcd = o3d.io.read_point_cloud(str(gt_path))
    n_gt_orig = len(gt_pcd.points)
    print(f"  GT original points: {n_gt_orig:,}")

    gt_down = gt_pcd.voxel_down_sample(VOXEL_SIZE)
    n_gt_down = len(gt_down.points)
    print(f"  GT downsampled points: {n_gt_down:,} ({n_gt_down/n_gt_orig*100:.1f}%)")

    # Save downsampled GT
    o3d.io.write_point_cloud(str(out_scene / f"gt_{name}.ply"), gt_down)

    # Extract unique colors
    if gt_down.has_colors():
        colors_uint8 = (np.asarray(gt_down.colors) * 255).round().astype(np.uint8)
        unique_colors = np.unique(colors_uint8, axis=0)
        print(f"  Unique GT colors ({len(unique_colors)}):")
        # Count per color
        color_tuples = [tuple(c) for c in colors_uint8]
        counter = Counter(color_tuples)
        for color, count in counter.most_common():
            pct = count / n_gt_down * 100
            print(f"    RGB({color[0]:3d}, {color[1]:3d}, {color[2]:3d}): {count:>10,} pts ({pct:5.1f}%)")
    else:
        print("  WARNING: GT has no colors!")

    # Save GT arrays
    np.save(str(out_scene / "gt_coord.npy"), np.asarray(gt_down.points).astype(np.float32))
    if gt_down.has_colors():
        np.save(str(out_scene / "gt_color.npy"), colors_uint8)

    # --- Raw ---
    raw_path = CHAS_ROOT / "raw" / info["raw"]
    print(f"\n  Loading Raw: {raw_path} ...")
    raw_pcd = o3d.io.read_point_cloud(str(raw_path))
    n_raw_orig = len(raw_pcd.points)
    print(f"  Raw original points: {n_raw_orig:,}")

    raw_down = raw_pcd.voxel_down_sample(VOXEL_SIZE)
    n_raw_down = len(raw_down.points)
    print(f"  Raw downsampled points: {n_raw_down:,} ({n_raw_down/n_raw_orig*100:.1f}%)")

    # Save downsampled raw
    o3d.io.write_point_cloud(str(out_scene / f"raw_{name}.ply"), raw_down)

    # Save raw arrays
    np.save(str(out_scene / "raw_coord.npy"), np.asarray(raw_down.points).astype(np.float32))
    if raw_down.has_colors():
        raw_colors_uint8 = (np.asarray(raw_down.colors) * 255).round().astype(np.uint8)
        np.save(str(out_scene / "raw_color.npy"), raw_colors_uint8)

    print(f"  Saved to: {out_scene}")
    return {
        "name": name,
        "raw_orig": n_raw_orig,
        "raw_down": n_raw_down,
        "gt_orig": n_gt_orig,
        "gt_down": n_gt_down,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for name, info in SCENES.items():
        try:
            r = downsample_and_analyze(name, info)
            results.append(r)
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['name']:12s}: raw {r['raw_orig']:>12,} -> {r['raw_down']:>10,}  |  gt {r['gt_orig']:>12,} -> {r['gt_down']:>10,}")


if __name__ == "__main__":
    main()
