"""Pure BFANet control loss (CR-N).

Faithful BFANet-style reproduction as a clean control against CR-L. Same
dual-branch model, but the loss strips CR-L's deviations from BFANet:

  1. Semantic CE uses a hard-mask 10x upweight on boundary-positive voxels
     (support > threshold -> weight=10, else weight=1). CR-L upweights by
     continuous support via sample_weight_scale; CR-N uses the same 0/1
     mask as the boundary target.
  2. Boundary branch is unweighted BCE + **global** Dice. CR-L uses a
     support-scaled per-point sample weight plus local Dice restricted to
     the support>0 region. CR-N removes both.
  3. Boundary target is still `support > boundary_threshold` hard binary
     (threshold=0.5), identical to CR-L.

Total loss: L = (CE_weighted + Lovasz) + aux_weight * (BCE + dice_weight * Dice)

This is the faithful BFANet baseline. Contrasting CR-N vs CR-L isolates
"does support weighting + local Dice help?"; CR-N vs CR-A isolates
"does the BFANet binary aux head help at all on this dataset?".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class PureBFANetLoss(nn.Module):
    """Hard-mask 10x semantic CE + unweighted BCE + global Dice (CR-N)."""

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
        edge: torch.Tensor,
        boundary_mask: torch.Tensor | None = None,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        edge = edge.float()

        # Prefer precomputed radius-based BFANet boundary mask (r=0.06 m).
        # Fallback: threshold continuous support — legacy CR-L-style derivation.
        if boundary_mask is not None:
            boundary_mask = boundary_mask.reshape(-1).float()
        else:
            support_gt = edge[:, 3].float().clamp(0.0, 1.0)
            boundary_mask = (support_gt > self.boundary_threshold).float()

        # === Term 1: Hard-mask 10x weighted semantic CE (BFANet original) ===
        # BFANet normalizes by total point count (ignore_index points contribute 0
        # via F.cross_entropy(ignore_index=-1, reduction="none")), matching
        # train.py line 150-153 where semantic_loss * sem_weight is followed by
        # an unmasked .mean() over all points including -100.
        ce_per_point = F.cross_entropy(
            seg_logits, segment, ignore_index=-1, reduction="none"
        )
        ce_point_weight = 1.0 + boundary_mask * (self.boundary_ce_weight - 1.0)
        loss_ce_weighted = (ce_per_point * ce_point_weight).mean()
        valid_semantic = segment != -1

        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce_weighted + loss_lovasz

        # === Term 2: Unweighted binary BCE (pos_weight=1, no sample weight) ===
        # BFANet writes this as BCELoss(sigmoid(margin), target) in fp32
        # (network/BFANet.py line 166-168). We use binary_cross_entropy_with_logits
        # because F.binary_cross_entropy is NOT AMP-safe — under torch.autocast
        # the fp16 sigmoid output can underflow and PyTorch explicitly forbids
        # it with "unsafe to autocast". BCEWithLogits is mathematically and
        # gradient-wise identical to BCELoss(sigmoid(.)) (log-sum-exp trick
        # internally), so this is a surface-form deviation forced by AMP
        # compatibility, not a numerical difference.
        binary_target = boundary_mask
        support_logit = support_pred[:, 0]
        support_prob = torch.sigmoid(support_logit)  # kept for Dice + monitoring

        bce_per_point = F.binary_cross_entropy_with_logits(
            support_logit, binary_target, reduction="none"
        )
        loss_bce = bce_per_point.mean()

        # === Term 2b: Global Dice over the whole output (BFANet original) ===
        smooth = self.dice_smooth
        intersection = (support_prob * binary_target).sum()
        dice_score = (2.0 * intersection + smooth) / (
            support_prob.sum() + binary_target.sum() + smooth
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
                (ce_per_point * boundary_mask).sum()
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
