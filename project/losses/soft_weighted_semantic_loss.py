"""GT-support fully-soft-weighted semantic CE + Lovasz (CR-O).

Difference from CR-K (BoundaryWeightedSemanticLoss):
  CR-K uses a *truncated* soft band — points with support <= 0.5 get
  weight 1.0 unchanged. Only support > 0.5 receives `1 + s * 9`.
  CR-O removes the truncation: every point's weight scales linearly
  with support_gt across the full [0, 1] range.

Per-point weight:
    w(s) = 1 + s * boundary_ce_weight_minus_one
where s = clamp(support_gt, 0, 1) and boundary_ce_weight_minus_one
defaults to 9.0, giving:
    s=0.0  -> w=1   (background, far from any boundary)
    s=0.5  -> w=5.5 (transition zone — now also upweighted)
    s=1.0  -> w=10  (boundary core)

Same semantic-only model as CR-A (no boundary head, no aux loss).
This is the minimal smooth extension of CR-A: replace `weight=1` with
`weight = 1 + s * 9`. Tests whether continuous boundary proximity
weighting on its own moves val_mIoU within seed noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class SoftWeightedSemanticLoss(nn.Module):
    """Semantic CE with fully continuous support weighting + Lovasz."""

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

        ce_per_point = F.cross_entropy(
            seg_logits, segment, ignore_index=-1, reduction="none"
        )
        # Fully continuous: every point's weight = 1 + s * (K - 1)
        point_weight = 1.0 + support_gt * (self.boundary_ce_weight - 1.0)
        valid_semantic = segment != -1
        loss_ce_weighted = (
            (ce_per_point * point_weight * valid_semantic.float()).sum()
            / valid_semantic.float().sum().clamp_min(1.0)
        )

        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        total = loss_ce_weighted + loss_lovasz

        with torch.no_grad():
            high_support = support_gt > 0.5
            mid_support = (support_gt > 0) & (support_gt <= 0.5)
            boundary_ce_frac = (
                (ce_per_point * high_support.float()).sum()
                / ce_per_point.sum().clamp_min(1e-6)
            )
            transition_ce_frac = (
                (ce_per_point * mid_support.float()).sum()
                / ce_per_point.sum().clamp_min(1e-6)
            )
            mean_weight = point_weight.mean()

        return dict(
            loss=total,
            loss_semantic=total,
            loss_ce_weighted=loss_ce_weighted,
            loss_lovasz=loss_lovasz,
            boundary_ce_frac=boundary_ce_frac,
            transition_ce_frac=transition_ce_frac,
            mean_point_weight=mean_weight,
        )
