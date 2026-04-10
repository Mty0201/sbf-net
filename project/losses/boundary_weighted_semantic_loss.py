"""GT-support-weighted semantic CE + Lovasz (CR-K ablation).

Ablation of CR-I: removes boundary head and aux MSE entirely.
Uses GT support (edge[:,3]) directly as CE per-point weight.
Same semantic-only model as CR-A — zero extra parameters.

Purpose: isolate whether the mIoU gain (if any) comes from
GT boundary weighting alone, or requires the aux MSE signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class BoundaryWeightedSemanticLoss(nn.Module):
    """Semantic CE weighted by GT support + Lovasz. No aux loss."""

    def __init__(
        self,
        boundary_ce_weight: float = 10.0,
    ):
        super().__init__()
        self.boundary_ce_weight = float(boundary_ce_weight)
        self.lovasz_loss = LovaszLoss(
            mode="multiclass", ignore_index=-1, loss_weight=1.0
        )

    def forward(
        self,
        seg_logits: torch.Tensor,
        segment: torch.Tensor,
        edge: torch.Tensor,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        edge = edge.float()

        support_gt = edge[:, 3].float().clamp(0.0, 1.0)

        # Support-weighted semantic CE (same as CR-I Term 1)
        ce_per_point = F.cross_entropy(
            seg_logits, segment, ignore_index=-1, reduction="none"
        )
        boundary_mask = (support_gt > 0.5).float()
        point_weight = 1.0 + boundary_mask * support_gt * (self.boundary_ce_weight - 1.0)
        valid_semantic = segment != -1
        loss_ce_weighted = (
            (ce_per_point * point_weight * valid_semantic.float()).sum()
            / valid_semantic.float().sum().clamp_min(1.0)
        )

        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        total = loss_ce_weighted + loss_lovasz

        # Monitoring
        with torch.no_grad():
            high_support = support_gt > 0.5
            boundary_ce_frac = (
                (ce_per_point * high_support.float()).sum()
                / ce_per_point.sum().clamp_min(1e-6)
            )

        return dict(
            loss=total,
            loss_semantic=total,
            loss_ce_weighted=loss_ce_weighted,
            loss_lovasz=loss_lovasz,
            boundary_ce_frac=boundary_ce_frac,
        )
