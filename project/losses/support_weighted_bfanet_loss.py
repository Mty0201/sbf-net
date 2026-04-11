"""Continuous support-weighted BFANet loss (CR-V base).

Byte-identical to ``PureBFANetLoss`` except the semantic CE per-point
weight is driven by a continuous ``s_weight`` field in [0, 1] instead of
a hard 0/1 boundary mask. The boundary branch (BCE + global Dice) still
uses the hard ``boundary_mask`` — CR-V decouples "what gets a wider
loss-weighting footprint" (semantic CE) from "what the boundary head
classifies" (hard r=0.06 m mask).

The single-line delta from PureBFANetLoss:

    # OLD (PureBFANetLoss):
    ce_point_weight = 1.0 + boundary_mask * (self.boundary_ce_weight - 1.0)

    # NEW (SupportWeightedBFANetLoss):
    ce_point_weight = 1.0 + s_weight * (self.boundary_ce_weight - 1.0)

``s_weight`` is precomputed offline by
``scripts/data/generate_support_weight.py``:

    s_weight = 1.0                         if d <= core_radius (0.06 m)
               exp(-k * (d - core) / width) if core < d <= outer (0.12 m)
               0.0                         otherwise

where ``d`` is the per-point physical distance to the nearest point of a
different semantic class. Core subset (``s_weight >= 1 - eps``) agrees
with ``boundary_mask_r060`` by construction.

If ``s_weight`` is not present in the batch (preprocessing not run, or
running on an older dataset), the loss falls back to ``boundary_mask``
and reproduces PureBFANetLoss exactly. This keeps CR-V runnable on any
dataset that CR-Q / CR-P can already train on.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class SupportWeightedBFANetLoss(nn.Module):
    """Continuous support-weighted CE + unweighted BCE + global Dice (CR-V)."""

    def __init__(
        self,
        aux_weight: float = 1.0,
        boundary_ce_weight: float = 10.0,
        boundary_threshold: float = 0.5,
        pos_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.aux_weight = float(aux_weight)
        self.boundary_ce_weight = float(boundary_ce_weight)
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
        edge: torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
        s_weight: torch.Tensor | None = None,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()

        # Hard boundary mask for the binary aux branch (BCE + Dice).
        if boundary_mask is not None:
            boundary_mask = boundary_mask.reshape(-1).float()
        elif edge is not None:
            support_gt = edge.float()[:, 3].clamp(0.0, 1.0)
            boundary_mask = (support_gt > self.boundary_threshold).float()
        else:
            raise RuntimeError(
                "SupportWeightedBFANetLoss requires either boundary_mask "
                "or edge to derive the binary boundary target."
            )

        # Continuous weight for the semantic CE branch. Falls back to the
        # hard boundary mask (reproducing PureBFANetLoss exactly) when the
        # precomputed field is absent.
        if s_weight is not None:
            s_weight = s_weight.reshape(-1).float().clamp(0.0, 1.0)
        else:
            s_weight = boundary_mask

        # === Term 1: Continuous-weight semantic CE ===
        ce_per_point = F.cross_entropy(
            seg_logits, segment, ignore_index=-1, reduction="none"
        )
        ce_point_weight = 1.0 + s_weight * (self.boundary_ce_weight - 1.0)
        loss_ce_weighted = (ce_per_point * ce_point_weight).mean()

        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce_weighted + loss_lovasz

        # === Term 2: Unweighted binary BCE on the hard mask ===
        binary_target = boundary_mask
        support_logit = support_pred[:, 0]
        support_prob = torch.sigmoid(support_logit)

        bce_per_point = F.binary_cross_entropy_with_logits(
            support_logit, binary_target, reduction="none"
        )
        loss_bce = bce_per_point.mean()

        # === Term 2b: Global Dice on the hard mask ===
        smooth = self.dice_smooth
        intersection = (support_prob * binary_target).sum()
        dice_score = (2.0 * intersection + smooth) / (
            support_prob.sum() + binary_target.sum() + smooth
        )
        loss_dice = 1.0 - dice_score

        loss_aux = loss_bce + self.dice_weight * loss_dice
        loss_aux_weighted = self.aux_weight * loss_aux
        total = loss_semantic + loss_aux_weighted

        with torch.no_grad():
            positive = binary_target > 0.5
            negative = ~positive
            pos_sum = positive.float().sum().clamp_min(1e-6)
            neg_sum = negative.float().sum().clamp_min(1e-6)
            prob_pos_mean = (support_prob * positive.float()).sum() / pos_sum
            prob_neg_mean = (support_prob * negative.float()).sum() / neg_sum
            boundary_ce_frac = (
                (ce_per_point * boundary_mask).sum()
                / ce_per_point.sum().clamp_min(1e-6)
            )
            ce_weight_mean = ce_point_weight.mean()
            ce_weight_max = ce_point_weight.max()

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
            ce_weight_mean=ce_weight_mean,
            ce_weight_max=ce_weight_max,
        )
