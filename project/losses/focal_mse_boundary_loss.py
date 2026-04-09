"""Focal MSE + Dice boundary loss: semantic + MSE/Dice combo on support.

CR-H experiment. Replaces BCE (which has an irreducible entropy lower bound
on continuous targets) with MSE + soft Dice combination:
  - MSE: per-point regression, lower bound = 0, provides stable early
    gradients. Focal weighting (pos_alpha) handles class imbalance.
  - Dice: global overlap metric, naturally handles extreme imbalance,
    prevents all-zero collapse. Provides precise late-stage signal even
    after MSE gradients vanish.

The two terms are complementary:
  - Early training: MSE dominates (stable per-point gradients)
  - Late training: Dice dominates (MSE vanishes, Dice keeps refining)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class FocalMSEBoundaryLoss(nn.Module):
    """Two-term loss: global semantic (CE+Lovasz) + MSE/Dice on support."""

    def __init__(
        self,
        aux_weight: float = 0.3,
        pos_alpha: float = 9.0,
        dice_weight: float = 1.0,
        mse_weight: float = 1.0,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.aux_weight = float(aux_weight)
        self.pos_alpha = float(pos_alpha)
        self.dice_weight = float(dice_weight)
        self.mse_weight = float(mse_weight)
        self.dice_smooth = float(dice_smooth)
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

        # === Term 2a: Focal MSE on support ===
        support_prob = torch.sigmoid(support_pred[:, 0])
        mse_per_point = (support_prob - support_gt) ** 2
        point_weight = 1.0 + support_gt * self.pos_alpha
        loss_mse = (point_weight * mse_per_point).mean()

        # === Term 2b: Soft Dice on support ===
        smooth = self.dice_smooth
        intersection = (support_prob * support_gt).sum()
        dice = (2.0 * intersection + smooth) / (
            support_prob.sum() + support_gt.sum() + smooth
        )
        loss_dice = 1.0 - dice

        # Combined aux
        loss_aux = self.mse_weight * loss_mse + self.dice_weight * loss_dice
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
            loss_dice=loss_dice,
            dice_score=dice,
            valid_ratio=(support_gt > 0.5).float().mean(),
            support_positive_ratio=(support_gt > 1e-3).float().mean(),
            aux_prob_mean=aux_prob_mean,
            aux_prob_boundary_mean=aux_prob_boundary_mean,
        )
