"""Focal MSE boundary loss: semantic + focal MSE on support.

CR-H experiment. Replaces BCE (which has an irreducible entropy lower bound
on continuous targets) with focal MSE:
  - MSE: per-point regression, lower bound = 0, gradient vanishes once
    support is learned. Focal weighting (pos_alpha) handles class imbalance.

Dice was removed after CR-H 100-epoch analysis: Dice loss (~0.55) dominated
the aux term and never decayed, causing aux to consume >55% of the total
loss budget in late training and starving semantic optimization.
With MSE-only, aux_weighted ≈ 0.02 at convergence vs semantic ≈ 0.14,
keeping a healthy ~14% ratio.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pointcept.models.losses.lovasz import LovaszLoss


class FocalMSEBoundaryLoss(nn.Module):
    """Two-term loss: global semantic (CE+Lovasz) + focal MSE on support."""

    def __init__(
        self,
        aux_weight: float = 0.3,
        pos_alpha: float = 9.0,
    ):
        super().__init__()
        self.aux_weight = float(aux_weight)
        self.pos_alpha = float(pos_alpha)
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_loss = LovaszLoss(
            mode="multiclass", ignore_index=-1, loss_weight=1.0
        )

    def forward(
        self,
        seg_logits: torch.Tensor,
        support_pred: torch.Tensor,
        segment: torch.Tensor,
        edge: torch.Tensor,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        edge = edge.float()

        support_gt = edge[:, 3].float().clamp(0.0, 1.0)

        # === Term 1: Global semantic (CE + Lovasz) ===
        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz

        # === Term 2: Focal MSE on support ===
        support_prob = torch.sigmoid(support_pred[:, 0])
        mse_per_point = (support_prob - support_gt) ** 2
        point_weight = 1.0 + support_gt * self.pos_alpha
        loss_mse = (point_weight * mse_per_point).mean()

        loss_aux = loss_mse
        loss_aux_weighted = self.aux_weight * loss_aux
        total = loss_semantic + loss_aux_weighted

        # Monitoring metrics (detached)
        with torch.no_grad():
            high_support = support_gt > 0.5
            high_support_sum = high_support.float().sum().clamp_min(1e-6)
            aux_prob_mean = support_prob.mean()
            aux_prob_boundary_mean = (
                (support_prob * high_support.float()).sum() / high_support_sum
            )

        return dict(
            loss=total,
            loss_semantic=loss_semantic,
            loss_ce=loss_ce,
            loss_lovasz=loss_lovasz,
            loss_aux=loss_aux,
            loss_aux_weighted=loss_aux_weighted,
            loss_mse=loss_mse,
            valid_ratio=(support_gt > 0.5).float().mean(),
            support_positive_ratio=(support_gt > 1e-3).float().mean(),
            aux_prob_mean=aux_prob_mean,
            aux_prob_boundary_mean=aux_prob_boundary_mean,
        )
