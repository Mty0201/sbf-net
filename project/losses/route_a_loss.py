"""Route A: SemanticBoundaryLoss + local within-basin direction coherence.

Basin coherence regulariser design:
- Groups valid direction points by (scene_idx, support_id) to prevent
  cross-scene contamination when samples are concatenated in a batch.
- For each basin group, finds each point's nearest same-basin neighbour
  via pairwise cdist (capped at max_points_per_basin points to bound cost).
- Only enforces coherence between pairs with spatial distance < local_radius,
  so the constraint is genuinely local, not a full-basin pull.
- Returns zero with gradient when no valid pairs exist (no silent skip).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .semantic_boundary_loss import SemanticBoundaryLoss

# Safety cap: large enough to avoid real id collision across scenes.
_SCENE_BASIN_STRIDE = 100_000


class RouteASemanticBoundaryLoss(SemanticBoundaryLoss):
    """SemanticBoundaryLoss + local within-basin direction coherence term."""

    def __init__(
        self,
        coherence_weight: float = 0.1,
        local_radius: float = 0.30,
        max_points_per_basin: int = 100,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.coherence_weight = float(coherence_weight)
        self.local_radius = float(local_radius)
        self.max_points_per_basin = int(max_points_per_basin)

    # ------------------------------------------------------------------
    # Basin coherence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_unique_basin_ids(
        support_id: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Shift per-scene support ids so they are globally unique in the batch."""
        unique = support_id.clone().long()
        starts = torch.cat(
            [torch.zeros(1, dtype=offset.dtype, device=offset.device), offset[:-1]]
        )
        for scene_idx, (start, end) in enumerate(
            zip(starts.tolist(), offset.tolist())
        ):
            start, end = int(start), int(end)
            unique[start:end] += scene_idx * _SCENE_BASIN_STRIDE
        return unique

    def _basin_coherence_loss(
        self,
        dir_pred_unit: torch.Tensor,
        coord: torch.Tensor,
        support_id: torch.Tensor,
        offset: torch.Tensor,
        dir_valid_mask: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not dir_valid_mask.any():
            return reference.sum() * 0.0

        unique_basin_id = self._make_unique_basin_ids(support_id, offset)

        # Restrict to direction-valid points only.
        valid_idx = torch.where(dir_valid_mask)[0]
        valid_basin = unique_basin_id[valid_idx]
        valid_coord = coord[valid_idx]
        valid_dir = dir_pred_unit[valid_idx]

        all_basins = torch.unique(valid_basin)
        coherence_terms: list[torch.Tensor] = []

        for bid in all_basins:
            in_basin = valid_basin == bid
            n = int(in_basin.sum().item())
            if n < 2:
                continue

            idx_b = torch.where(in_basin)[0]

            # Cap the number of points to bound cdist cost.
            if n > self.max_points_per_basin:
                perm = torch.randperm(n, device=dir_pred_unit.device)
                idx_b = idx_b[perm[: self.max_points_per_basin]]

            c = valid_coord[idx_b]   # (K, 3)
            d = valid_dir[idx_b]     # (K, 3)

            # Pairwise distances; self-distance set to inf.
            pdist = torch.cdist(c, c)                   # (K, K)
            pdist.fill_diagonal_(float("inf"))

            # Nearest same-basin neighbour for each point.
            nn_dist, nn_idx = pdist.min(dim=1)          # (K,)

            # Only use locally close pairs.
            local_mask = nn_dist < self.local_radius
            if not local_mask.any():
                continue

            cos = (
                (d[local_mask] * d[nn_idx[local_mask]])
                .sum(dim=1)
                .clamp(-1.0, 1.0)
            )
            coherence_terms.append(1.0 - cos.mean())

        if not coherence_terms:
            return reference.sum() * 0.0

        return torch.stack(coherence_terms).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        seg_logits: torch.Tensor,
        edge_pred: torch.Tensor,
        segment: torch.Tensor,
        edge: torch.Tensor,
        coord: torch.Tensor | None = None,
        support_id: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        base = super().forward(seg_logits, edge_pred, segment, edge)

        can_compute_coherence = (
            self.coherence_weight > 0.0
            and coord is not None
            and support_id is not None
            and offset is not None
        )

        if not can_compute_coherence:
            base["loss_coherence"] = edge_pred.sum() * 0.0
            return base

        # Reconstruct dir_valid_mask from edge GT (same logic as base).
        edge_f = edge.float()
        dist_gt = edge_f[:, 3].clamp_min(0.0)
        valid_gt = edge_f[:, 5].clamp(0.0, 1.0)
        dir_valid_mask = (valid_gt > 0.5) & (dist_gt > self.tau_dir)

        dir_pred_unit = F.normalize(edge_pred[:, 0:3], dim=1, eps=1e-6)

        loss_coherence = self._basin_coherence_loss(
            dir_pred_unit=dir_pred_unit,
            coord=coord,
            support_id=support_id,
            offset=offset,
            dir_valid_mask=dir_valid_mask,
            reference=edge_pred,
        )

        base["loss_coherence"] = loss_coherence
        base["loss"] = base["loss"] + self.coherence_weight * loss_coherence

        return base
