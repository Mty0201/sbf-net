"""
NET-01 Diagnosis: Stratified density-bucketed analysis of pipeline stages.

Analyzes the boundary supervision pipeline per density bucket (dense/mid/sparse)
to identify where sparse-region data quality degrades relative to dense regions.

Diagnosis stages:
  - Stage 2 (clustering): survival rate by density bucket
  - Stage 4 (pointwise edge): valid yield and weight by density bucket
  - Cross-stage gap attribution

Usage:
    python diagnose_net01.py \
        --scenes path/to/scene1 path/to/scene2 \
        --output path/to/diagnosis_output.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# Inline path setup (tools/ is one level below scripts/)
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from utils.stage_io import load_boundary_centers, load_local_clusters


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NET-01 diagnosis: density-bucketed pipeline analysis"
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        required=True,
        help="One or more scene directories to diagnose",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write diagnosis_output.json",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Density computation
# ---------------------------------------------------------------------------


def compute_scene_knn_distances(coord: np.ndarray, k: int = 10) -> np.ndarray:
    """Compute mean kNN distance for every scene point.

    Queries k+1 neighbors (includes self), discards self, takes mean over k.
    Returns shape (N,) of mean kNN distances.
    """
    n = coord.shape[0]
    query_k = min(k + 1, n)
    tree = cKDTree(coord)
    dist, _ = tree.query(coord, k=query_k, workers=-1)
    if query_k == 1:
        dist = dist.reshape(-1, 1)
    if query_k <= 1:
        return np.zeros(n, dtype=np.float32)
    # Exclude self (column 0), take mean over remaining k columns
    knn_dist = dist[:, 1:].astype(np.float32)
    return knn_dist.mean(axis=1).astype(np.float32)


def assign_density_buckets(
    densities: np.ndarray, p25: float, p75: float
) -> np.ndarray:
    """Assign bucket labels: 0=dense, 1=mid, 2=sparse."""
    buckets = np.ones(densities.shape[0], dtype=np.int32)  # default=mid
    buckets[densities <= p25] = 0  # dense (low knn dist = high density)
    buckets[densities >= p75] = 2  # sparse (high knn dist = low density)
    return buckets


BUCKET_NAMES = {0: "dense", 1: "mid", 2: "sparse"}


# ---------------------------------------------------------------------------
# Stage 2 analysis: cluster survival
# ---------------------------------------------------------------------------


def analyze_stage2(
    boundary_centers: dict,
    local_clusters: dict,
    center_buckets: np.ndarray,
) -> dict:
    """Compute cluster survival rate per density bucket."""
    num_centers = boundary_centers["center_coord"].shape[0]
    survived_set = set(local_clusters["center_index"].tolist())

    result = {}
    for bucket_id, bucket_name in BUCKET_NAMES.items():
        in_bucket = np.where(center_buckets == bucket_id)[0]
        total = int(in_bucket.shape[0])
        survived = int(sum(1 for idx in in_bucket if int(idx) in survived_set))
        rate = float(survived / max(total, 1))
        noise = total - survived
        noise_rate = float(noise / max(total, 1))

        # Mean cluster size for surviving centers in this bucket
        survived_indices = np.array(
            [idx for idx in in_bucket if int(idx) in survived_set], dtype=np.int32
        )
        mean_cluster_size = 0.0
        if survived_indices.size > 0:
            # Map each survived center to its cluster_id, then look up cluster_size
            ci = local_clusters["center_index"]
            cid = local_clusters["cluster_id"]
            csz = local_clusters["cluster_size"]
            # Build center_index -> cluster_id map
            center_to_cluster = {}
            for i in range(ci.shape[0]):
                center_to_cluster[int(ci[i])] = int(cid[i])
            sizes = []
            for idx in survived_indices:
                cluster = center_to_cluster.get(int(idx))
                if cluster is not None and cluster < csz.shape[0]:
                    sizes.append(int(csz[cluster]))
            if sizes:
                mean_cluster_size = float(np.mean(sizes))

        result[bucket_name] = {
            "total": total,
            "survived": survived,
            "rate": round(rate, 6),
            "noise_rate": round(noise_rate, 6),
            "mean_cluster_size": round(mean_cluster_size, 2),
        }
    return result


# ---------------------------------------------------------------------------
# Stage 4 analysis: valid yield and weight
# ---------------------------------------------------------------------------


def analyze_stage4(
    edge: np.ndarray,
    scene_point_buckets: np.ndarray,
) -> dict:
    """Compute valid yield and weight stats per density bucket for scene points.

    edge.npy columns (6-col converted format):
        [dir_x, dir_y, dir_z, dist, weight(support), valid]
    """
    weight_col = edge[:, 4]
    valid_col = edge[:, 5]

    result = {}
    for bucket_id, bucket_name in BUCKET_NAMES.items():
        in_bucket = np.where(scene_point_buckets == bucket_id)[0]
        total = int(in_bucket.shape[0])

        bucket_valid = valid_col[in_bucket]
        bucket_weight = weight_col[in_bucket]

        valid_mask = bucket_valid > 0.5
        num_valid = int(np.count_nonzero(valid_mask))
        valid_rate = float(num_valid / max(total, 1))

        mean_weight = 0.0
        if num_valid > 0:
            mean_weight = float(np.mean(bucket_weight[valid_mask]))

        high_weight_mask = valid_mask & (bucket_weight >= 0.5)
        high_weight_count = int(np.count_nonzero(high_weight_mask))
        high_weight_rate = float(high_weight_count / max(total, 1))

        result[bucket_name] = {
            "total": total,
            "valid": num_valid,
            "rate": round(valid_rate, 6),
            "mean_weight": round(mean_weight, 6),
            "high_weight_count": high_weight_count,
            "high_weight_rate": round(high_weight_rate, 6),
        }
    return result


# ---------------------------------------------------------------------------
# Gap attribution
# ---------------------------------------------------------------------------


def compute_gaps(stage2: dict, stage4: dict) -> dict:
    """Cross-stage gap attribution: dense - sparse."""
    stage2_gap = stage2["dense"]["rate"] - stage2["sparse"]["rate"]
    stage4_gap = stage4["dense"]["rate"] - stage4["sparse"]["rate"]
    stage4_weight_gap = (
        stage4["dense"]["high_weight_rate"] - stage4["sparse"]["high_weight_rate"]
    )
    return {
        "stage2_survival_gap": round(stage2_gap, 6),
        "stage4_valid_gap": round(stage4_gap, 6),
        "stage4_weight_gap": round(stage4_weight_gap, 6),
    }


# ---------------------------------------------------------------------------
# Main diagnosis flow
# ---------------------------------------------------------------------------


def diagnose_scene(scene_dir: Path) -> dict:
    """Run full density-bucketed diagnosis for one scene."""
    scene_name = scene_dir.name

    # Load scene points
    coord = np.load(scene_dir / "coord.npy").astype(np.float32)
    num_points = coord.shape[0]

    # Load pipeline artifacts
    boundary_centers = load_boundary_centers(scene_dir)
    local_clusters = load_local_clusters(scene_dir)
    edge = np.load(scene_dir / "edge.npy").astype(np.float32)

    print(f"\n{'='*70}")
    print(f"  Scene: {scene_name}")
    print(f"  Points: {num_points}")
    print(f"  Boundary centers: {boundary_centers['center_coord'].shape[0]}")
    print(f"  Clusters: {local_clusters['cluster_size'].shape[0]}")
    print(f"  Edge shape: {edge.shape}")
    print(f"{'='*70}")

    # 1. Compute scene-wide kNN density
    print(f"\n  Computing kNN density (k=10) for {num_points} points...")
    mean_knn_dist = compute_scene_knn_distances(coord, k=10)

    # 2. Density buckets for boundary centers
    source_point_index = boundary_centers["source_point_index"]
    center_densities = mean_knn_dist[source_point_index]
    p25 = float(np.percentile(center_densities, 25))
    p75 = float(np.percentile(center_densities, 75))
    center_buckets = assign_density_buckets(center_densities, p25, p75)

    # 3. Density buckets for scene points (using same thresholds from centers)
    scene_point_buckets = assign_density_buckets(mean_knn_dist, p25, p75)

    print(f"  Density thresholds: P25={p25:.6f}, P75={p75:.6f}")
    for bucket_id, name in BUCKET_NAMES.items():
        n_centers = int(np.count_nonzero(center_buckets == bucket_id))
        n_points = int(np.count_nonzero(scene_point_buckets == bucket_id))
        print(f"    {name}: {n_centers} centers, {n_points} scene points")

    # 4. Stage 2 analysis
    stage2 = analyze_stage2(boundary_centers, local_clusters, center_buckets)
    print(f"\n  Stage 2 - Cluster survival:")
    for name in ["dense", "mid", "sparse"]:
        s = stage2[name]
        print(
            f"    {name}: {s['survived']}/{s['total']} survived "
            f"(rate={s['rate']:.4f}, noise={s['noise_rate']:.4f}, "
            f"mean_cluster_size={s['mean_cluster_size']:.1f})"
        )

    # 5. Stage 4 analysis
    stage4 = analyze_stage4(edge, scene_point_buckets)
    print(f"\n  Stage 4 - Valid yield:")
    for name in ["dense", "mid", "sparse"]:
        s = stage4[name]
        print(
            f"    {name}: {s['valid']}/{s['total']} valid "
            f"(rate={s['rate']:.4f}, mean_weight={s['mean_weight']:.4f}, "
            f"high_weight_rate={s['high_weight_rate']:.4f})"
        )

    # 6. Gap attribution
    gaps = compute_gaps(stage2, stage4)
    print(f"\n  Gap attribution (dense - sparse):")
    print(f"    Stage 2 survival gap: {gaps['stage2_survival_gap']:+.4f}")
    print(f"    Stage 4 valid gap:    {gaps['stage4_valid_gap']:+.4f}")
    print(f"    Stage 4 weight gap:   {gaps['stage4_weight_gap']:+.4f}")

    return {
        "density_thresholds": {"P25": round(p25, 6), "P75": round(p75, 6)},
        "stage2_cluster_survival": stage2,
        "stage4_valid_yield": stage4,
        "gap_attribution": gaps,
    }


def main() -> None:
    args = parse_args()
    scenes = [Path(s) for s in args.scenes]
    output_path = Path(args.output)

    results = {"scenes": {}}

    for scene_dir in scenes:
        if not scene_dir.is_dir():
            print(f"WARNING: {scene_dir} is not a directory, skipping")
            continue
        scene_id = scene_dir.name
        results["scenes"][scene_id] = diagnose_scene(scene_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n{'='*70}")
    print(f"  Diagnosis output written to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
