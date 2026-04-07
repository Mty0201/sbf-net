"""Unit tests for the density-adaptive noise rescue function (ALG-02).

Uses synthetic numpy data with controlled random seed -- no sample scene
required.  Tests verify rescue_noise_centers behavior:
  - Nearby noise points get assigned to nearest cluster
  - Distant noise points remain unassigned (-1)
  - Edge cases: no noise, all noise
  - Distance scale factor controls rescue radius
"""

from __future__ import annotations

import numpy as np
import pytest

from core.local_clusters_core import rescue_noise_centers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_two_clusters_with_noise(
    rng: np.random.Generator,
    *,
    noise_coords: np.ndarray | None = None,
    n_per_cluster: int = 50,
    cluster_std: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Create 2 tight clusters at [0,0,0] and [1,0,0] plus optional noise."""
    c0 = rng.normal(0, cluster_std, (n_per_cluster, 3)).astype(np.float32)
    c1 = rng.normal([1, 0, 0], cluster_std, (n_per_cluster, 3)).astype(np.float32)

    parts_coords = [c0, c1]
    parts_labels = [
        np.zeros(n_per_cluster, dtype=np.int32),
        np.ones(n_per_cluster, dtype=np.int32),
    ]

    if noise_coords is not None:
        parts_coords.append(noise_coords.astype(np.float32))
        parts_labels.append(np.full(noise_coords.shape[0], -1, dtype=np.int32))

    coords = np.vstack(parts_coords)
    labels = np.concatenate(parts_labels)
    return coords, labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRescueNoiseCenters:
    def test_rescue_assigns_nearby_noise(self) -> None:
        """Noise point very close to cluster 0 should be assigned to cluster 0."""
        rng = np.random.default_rng(42)
        noise_near = np.array([[0.02, 0.0, 0.0]], dtype=np.float32)
        coords, labels = _make_two_clusters_with_noise(rng, noise_coords=noise_near)

        result = rescue_noise_centers(coords, labels, k=8, rescue_distance_scale=2.0)
        assert result[-1] == 0, "Nearby noise should be assigned to cluster 0"

    def test_rescue_ignores_distant_noise(self) -> None:
        """Noise point far from any cluster should remain unassigned (-1)."""
        rng = np.random.default_rng(42)
        noise_far = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
        coords, labels = _make_two_clusters_with_noise(rng, noise_coords=noise_far)

        result = rescue_noise_centers(coords, labels, k=8, rescue_distance_scale=2.0)
        assert result[-1] == -1, "Distant noise should remain unassigned"

    def test_rescue_empty_input(self) -> None:
        """When there are no noise points, labels should be unchanged."""
        rng = np.random.default_rng(42)
        coords, labels = _make_two_clusters_with_noise(rng, noise_coords=None)

        result = rescue_noise_centers(coords, labels, k=8, rescue_distance_scale=2.0)
        np.testing.assert_array_equal(result, labels)

    def test_rescue_all_noise(self) -> None:
        """When all points are noise (no clusters), labels should be unchanged."""
        rng = np.random.default_rng(42)
        coords = rng.normal(0, 1, (20, 3)).astype(np.float32)
        labels = np.full(20, -1, dtype=np.int32)

        result = rescue_noise_centers(coords, labels, k=8, rescue_distance_scale=2.0)
        np.testing.assert_array_equal(result, labels)

    def test_rescue_respects_scale_factor(self) -> None:
        """With a tight scale factor, fewer noise points should be rescued."""
        rng = np.random.default_rng(42)
        # Place noise at moderate distance from cluster 0
        noise_moderate = np.array([
            [0.05, 0.0, 0.0],
            [0.08, 0.0, 0.0],
            [0.12, 0.0, 0.0],
        ], dtype=np.float32)
        coords, labels = _make_two_clusters_with_noise(rng, noise_coords=noise_moderate)

        result_tight = rescue_noise_centers(coords, labels, k=8, rescue_distance_scale=0.5)
        result_loose = rescue_noise_centers(coords, labels, k=8, rescue_distance_scale=5.0)

        rescued_tight = int(np.count_nonzero(result_tight[-3:] >= 0))
        rescued_loose = int(np.count_nonzero(result_loose[-3:] >= 0))
        assert rescued_tight <= rescued_loose, (
            f"Tight scale should rescue fewer points: tight={rescued_tight}, loose={rescued_loose}"
        )
