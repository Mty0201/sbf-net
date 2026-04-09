"""Boundary-upweighted semantic + Focal MSE/Dice aux on support (CR-I).

CR-I = CR-H (proven aux) + BFANet-inspired semantic boundary upweight.

Two mechanisms, different roles:
  - Semantic upweight: CE per-point weight = 1 + support_gt * (K - 1).
    Continuous Gaussian support gives smooth weighting — points closer to
    the semantic boundary get proportionally higher CE weight.  No hard
    cutoff artifacts from binary valid.
  - Focal MSE + Dice aux: identical to CR-H.  Focal MSE provides stable
    per-point gradients (lower bound = 0, no BCE entropy floor).  Dice
    provides global overlap signal immune to ~2% boundary sparsity.
    Already validated in real training: fast separation, Dice improving,
    mIoU not degraded.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class BoundaryUpweightLoss(nn.Module):
    """Support-weighted semantic CE + Lovasz + Focal MSE/Dice aux."""

    def __init__(
        self,
        aux_weight: float = 0.3,
        boundary_ce_weight: float = 10.0,
        pos_alpha: float = 9.0,
        dice_weight: float = 1.0,
        mse_weight: float = 1.0,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.aux_weight = float(aux_weight)
        self.boundary_ce_weight = float(boundary_ce_weight)
        self.pos_alpha = float(pos_alpha)
        self.dice_weight = float(dice_weight)
        self.mse_weight = float(mse_weight)
        self.dice_smooth = float(dice_smooth)
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

        # Continuous support target: edge column 4, Gaussian decay from boundary
        support_gt = edge[:, 4].float().clamp(0.0, 1.0)

        # === Term 1: Support-weighted semantic CE ===
        ce_per_point = F.cross_entropy(
            seg_logits, segment, ignore_index=-1, reduction="none"
        )
        # Truncated weighting: only support > 0.5 gets upweighted, tail stays at 1.0
        boundary_mask = (support_gt > 0.5).float()
        point_weight = 1.0 + boundary_mask * support_gt * (self.boundary_ce_weight - 1.0)
        valid_semantic = segment != -1
        loss_ce_weighted = (
            (ce_per_point * point_weight * valid_semantic.float()).sum()
            / valid_semantic.float().sum().clamp_min(1.0)
        )

        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce_weighted + loss_lovasz

        # === Term 2a: Focal MSE on support (from CR-H) ===
        support_prob = torch.sigmoid(support_pred[:, 0])
        mse_per_point = (support_prob - support_gt) ** 2
        mse_point_weight = 1.0 + support_gt * self.pos_alpha
        loss_mse = (mse_point_weight * mse_per_point).mean()

        # === Term 2b: Soft Dice on support (from CR-H) ===
        smooth = self.dice_smooth
        intersection = (support_prob * support_gt).sum()
        dice = (2.0 * intersection + smooth) / (
            support_prob.sum() + support_gt.sum() + smooth
        )
        loss_dice = 1.0 - dice

        # Combined aux (same as CR-H)
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
            boundary_ce_frac = (
                (ce_per_point * high_support.float()).sum()
                / ce_per_point.sum().clamp_min(1e-6)
            )

        return dict(
            loss=total,
            loss_semantic=loss_semantic,
            loss_ce_weighted=loss_ce_weighted,
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
            boundary_ce_frac=boundary_ce_frac,
        )
