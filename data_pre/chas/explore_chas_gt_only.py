"""
CHAS dataset GT exploration: voxel downsample GT files only, extract dominant colors.
Processes one scene at a time to avoid OOM.

Usage:
    python data_pre/chas/explore_chas_gt_only.py
"""
import numpy as np
import open3d as o3d
from pathlib import Path
from collections import Counter
import gc

CHAS_ROOT = Path("/mnt/e/WSL/data/CHAS")
OUT_DIR = CHAS_ROOT / "downsampled_0.1m"

GT_FILES = {
    "baile":     "gt_baile.ply",
    "boa_morte": "boa_morte.ply",
    "curia":     "curia.ply",
    "engenho":   "gt_engenho.ply",
    "quilombo":  "gt_quilombo.ply",
}

VOXEL_SIZE = 0.1


def process_gt(name: str, gt_file: str):
    print(f"\n{'='*60}")
    print(f"GT: {name}")
    print(f"{'='*60}")

    gt_path = CHAS_ROOT / "gt" / gt_file
    out_scene = OUT_DIR / name
    out_scene.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    if (out_scene / "gt_color.npy").exists():
        print(f"  Already processed, loading cached npy...")
        colors_uint8 = np.load(str(out_scene / "gt_color.npy"))
        coord = np.load(str(out_scene / "gt_coord.npy"))
        n_down = len(coord)
    else:
        print(f"  Loading: {gt_path} ...")
        pcd = o3d.io.read_point_cloud(str(gt_path))
        n_orig = len(pcd.points)
        print(f"  Original: {n_orig:,}")

        pcd_down = pcd.voxel_down_sample(VOXEL_SIZE)
        n_down = len(pcd_down.points)
        print(f"  Downsampled: {n_down:,} ({n_down/n_orig*100:.1f}%)")

        coord = np.asarray(pcd_down.points).astype(np.float32)
        colors_uint8 = (np.asarray(pcd_down.colors) * 255).round().astype(np.uint8)

        np.save(str(out_scene / "gt_coord.npy"), coord)
        np.save(str(out_scene / "gt_color.npy"), colors_uint8)
        o3d.io.write_point_cloud(str(out_scene / f"gt_{name}.ply"), pcd_down)

        del pcd, pcd_down
        gc.collect()

    # Analyze dominant colors (those with > 0.5% of points)
    color_tuples = [tuple(c) for c in colors_uint8]
    counter = Counter(color_tuples)
    print(f"\n  Total downsampled points: {n_down:,}")
    print(f"  Total unique colors: {len(counter)}")

    threshold = n_down * 0.005  # 0.5%
    dominant = [(c, cnt) for c, cnt in counter.most_common() if cnt >= threshold]
    noise_count = sum(cnt for c, cnt in counter.items() if cnt < threshold)

    print(f"\n  Dominant colors (>0.5% of points): {len(dominant)}")
    for color, count in dominant:
        pct = count / n_down * 100
        print(f"    RGB({color[0]:3d}, {color[1]:3d}, {color[2]:3d}): {count:>10,} pts ({pct:5.1f}%)")
    print(f"    [noise / boundary mix]:           {noise_count:>10,} pts ({noise_count/n_down*100:5.1f}%)")

    return {
        "name": name,
        "n_down": n_down,
        "n_unique": len(counter),
        "dominant": dominant,
        "noise_count": noise_count,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for name, gt_file in GT_FILES.items():
        try:
            r = process_gt(name, gt_file)
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Cross-scene color analysis
    print(f"\n{'='*60}")
    print("CROSS-SCENE DOMINANT COLOR ANALYSIS")
    print(f"{'='*60}")

    all_dominant_colors = set()
    for r in all_results:
        for color, _ in r["dominant"]:
            all_dominant_colors.add(color)

    print(f"\nAll dominant colors across scenes: {len(all_dominant_colors)}")
    # For each color, show which scenes it appears in
    for color in sorted(all_dominant_colors):
        scenes = []
        for r in all_results:
            for c, cnt in r["dominant"]:
                if c == color:
                    scenes.append(f"{r['name']}({cnt/r['n_down']*100:.1f}%)")
        print(f"  RGB({color[0]:3d}, {color[1]:3d}, {color[2]:3d}): {', '.join(scenes)}")


if __name__ == "__main__":
    main()
