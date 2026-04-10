"""Support-weighted binary boundary BCE + boundary-weighted semantic CE (CR-L).

BFANet-inspired binary boundary classification, enhanced with continuous
support weighting. Replaces continuous support regression (CR-H/I) with
a clean binary task: "is this point a core boundary point?"

Target: (support_gt > 0.9) → binary {0, 1}.
Sample weight: base + support_gt * K. Three semantic levels:
  - Core boundary (support > 0.9, positive): highest weight
  - Transition zone (0 < support ≤ 0.9, negative): medium weight,
    hard negatives that also reflect boundary proximity
  - Background (support = 0, negative): base weight only

Binary target means BCE converges cleanly to 0 (no continuous residual).
Support weighting handles per-point BCE imbalance. Global Dice (unweighted,
BFANet-style) provides ratio-level overlap pressure — background prob→0
quickly so Dice denominator stays clean.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class BoundaryBinaryLoss(nn.Module):
    """Support-weighted binary BCE + global Dice aux + GT-support-weighted semantic CE."""

    def __init__(
        self,
        aux_weight: float = 0.3,
        boundary_ce_weight: float = 10.0,
        sample_weight_scale: float = 9.0,
        boundary_threshold: float = 0.9,
        pos_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.aux_weight = float(aux_weight)
        self.boundary_ce_weight = float(boundary_ce_weight)
        self.sample_weight_scale = float(sample_weight_scale)
        self.boundary_threshold = float(boundary_threshold)
        self.pos_weight = float(pos_weight)
        self.dice_weight = float(dice_weight)
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

        support_gt = edge[:, 3].float().clamp(0.0, 1.0)

        # === Term 1: Support-weighted semantic CE (same as CR-I) ===
        ce_per_point = F.cross_entropy(
            seg_logits, segment, ignore_index=-1, reduction="none"
        )
        ce_mask = (support_gt > 0.5).float()
        ce_point_weight = 1.0 + ce_mask * support_gt * (self.boundary_ce_weight - 1.0)
        valid_semantic = segment != -1
        loss_ce_weighted = (
            (ce_per_point * ce_point_weight * valid_semantic.float()).sum()
            / valid_semantic.float().sum().clamp_min(1.0)
        )

        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce_weighted + loss_lovasz

        # === Term 2: Support-weighted binary BCE ===
        binary_target = (support_gt > self.boundary_threshold).float()
        support_logit = support_pred[:, 0]

        # Per-point sample weight: background=1, transition=1+s*K, core≈1+K
        sample_weight = 1.0 + support_gt * self.sample_weight_scale

        # pos_weight inside BCE for positive class balance
        pw = torch.tensor(self.pos_weight, device=support_logit.device)
        bce_per_point = F.binary_cross_entropy_with_logits(
            support_logit, binary_target, pos_weight=pw, reduction="none"
        )
        loss_bce = (sample_weight * bce_per_point).mean()

        # === Term 2b: Local Dice on support > 0, unweighted ===
        support_prob = torch.sigmoid(support_logit)
        local_mask = support_gt > 0
        local_prob = support_prob[local_mask]
        local_target = binary_target[local_mask]
        smooth = self.dice_smooth
        intersection = (local_prob * local_target).sum()
        dice_score = (2.0 * intersection + smooth) / (
            local_prob.sum() + local_target.sum() + smooth
        )
        loss_dice = 1.0 - dice_score

        loss_aux = loss_bce + self.dice_weight * loss_dice
        loss_aux_weighted = self.aux_weight * loss_aux
        total = loss_semantic + loss_aux_weighted

        # Monitoring metrics (detached)
        with torch.no_grad():
            positive = binary_target > 0.5
            negative = ~positive
            pos_sum = positive.float().sum().clamp_min(1e-6)
            neg_sum = negative.float().sum().clamp_min(1e-6)
            prob_pos_mean = (support_prob * positive.float()).sum() / pos_sum
            prob_neg_mean = (support_prob * negative.float()).sum() / neg_sum
            boundary_ce_frac = (
                (ce_per_point * (support_gt > 0.5).float()).sum()
                / ce_per_point.sum().clamp_min(1e-6)
            )

        return dict(
            loss=total,
            loss_semantic=loss_semantic,
            loss_ce_weighted=loss_ce_weighted,
            loss_lovasz=loss_lovasz,
            loss_aux=loss_aux,
            loss_aux_weighted=loss_aux_weighted,
            loss_bce=loss_bce,
            loss_dice=loss_dice,
            dice_score=dice_score,
            prob_pos_mean=prob_pos_mean,
            prob_neg_mean=prob_neg_mean,
            positive_ratio=positive.float().mean(),
            boundary_ce_frac=boundary_ce_frac,
        )
