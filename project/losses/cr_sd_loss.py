"""CR-SD loss for DecoupledBFANetSegmentorV1.

Margin branch: BCE + global Dice on boundary_mask (CR-P / pure_bfanet_v4 recipe).
Replaces the original binary Lovász which plateaued at ~0.95 on ~7% positive ratio.
boundary_mask comes from precomputed r=0.06m radius search (~7.4% positive),
no edge/support column needed.

Semantic branch: s_weight continuous 10x CE upweight + multiclass Lovász (CR-V recipe).
Falls back to hard boundary_mask when s_weight is absent.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class CRSDLoss(nn.Module):
    def __init__(
        self,
        aux_weight: float = 1.0,
        boundary_ce_weight: float = 5.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.aux_weight = float(aux_weight)
        self.boundary_ce_weight = float(boundary_ce_weight)
        self.dice_weight = float(dice_weight)
        self.dice_smooth = float(dice_smooth)
        self.lovasz_multi = LovaszLoss(
            mode="multiclass", loss_weight=1.0, ignore_index=-1
        )

    def _compute_sem_loss(self, seg_logits, segment, sw):
        ce_per_point = F.cross_entropy(
            seg_logits, segment, ignore_index=-1, reduction="none"
        )
        ce_point_weight = 1.0 + sw * (self.boundary_ce_weight - 1.0)
        loss_ce = (ce_per_point * ce_point_weight).mean()
        loss_lovasz = self.lovasz_multi(seg_logits, segment)
        return loss_ce, loss_lovasz

    def _compute_marg_loss(self, marg_logits, binary_target, valid_mask):
        marg_valid = marg_logits.view(-1)[valid_mask]
        loss_bce = F.binary_cross_entropy_with_logits(
            marg_valid, binary_target, reduction="none"
        ).mean()
        marg_prob = torch.sigmoid(marg_valid)
        smooth = self.dice_smooth
        intersection = (marg_prob * binary_target).sum()
        dice_score = (2.0 * intersection + smooth) / (
            marg_prob.sum() + binary_target.sum() + smooth
        )
        loss_dice = 1.0 - dice_score
        return loss_bce, loss_dice, dice_score, marg_prob

    def forward(
        self,
        seg_logits: torch.Tensor,
        segment: torch.Tensor,
        marg_logits: torch.Tensor,
        boundary_mask: torch.Tensor,
        s_weight: torch.Tensor | None = None,
        seg_logits_v1: torch.Tensor | None = None,
        marg_logits_v1: torch.Tensor | None = None,
        alpha_mean: torch.Tensor | None = None,
        alpha_std: torch.Tensor | None = None,
        alpha_abs_max: torch.Tensor | None = None,
        w_fro: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        valid_mask = segment != -1

        if s_weight is not None:
            sw = s_weight.reshape(-1).float().clamp(0.0, 1.0)
        else:
            sw = boundary_mask.float().view(-1)

        binary_target = boundary_mask.float().view(-1)[valid_mask]

        # === v2 (post-fusion) losses ===
        loss_ce, loss_lovasz = self._compute_sem_loss(seg_logits, segment, sw)
        loss_marg_bce, loss_marg_dice, dice_score, marg_prob = (
            self._compute_marg_loss(marg_logits, binary_target, valid_mask)
        )
        loss_marg = loss_marg_bce + self.dice_weight * loss_marg_dice

        total = loss_ce + loss_lovasz + self.aux_weight * loss_marg

        # === v1 (pre-fusion) losses ===
        if seg_logits_v1 is not None:
            loss_ce_v1, loss_lovasz_v1 = self._compute_sem_loss(
                seg_logits_v1, segment, sw,
            )
            total = total + loss_ce_v1 + loss_lovasz_v1

        if marg_logits_v1 is not None:
            loss_marg_bce_v1, loss_marg_dice_v1, _, _ = (
                self._compute_marg_loss(marg_logits_v1, binary_target, valid_mask)
            )
            loss_marg_v1 = loss_marg_bce_v1 + self.dice_weight * loss_marg_dice_v1
            total = total + self.aux_weight * loss_marg_v1

        # Monitoring metrics (v2 only)
        with torch.no_grad():
            positive = binary_target > 0.5
            negative = ~positive
            pos_sum = positive.float().sum().clamp_min(1e-6)
            neg_sum = negative.float().sum().clamp_min(1e-6)
            prob_pos_mean = (marg_prob * positive.float()).sum() / pos_sum
            prob_neg_mean = (marg_prob * negative.float()).sum() / neg_sum

        out = dict(
            loss=total,
            loss_ce_weighted=loss_ce,
            loss_lovasz=loss_lovasz,
            loss_marg_bce=loss_marg_bce,
            loss_marg_dice=loss_marg_dice,
            dice_score=dice_score,
            prob_pos_mean=prob_pos_mean,
            prob_neg_mean=prob_neg_mean,
            positive_ratio=positive.float().mean(),
        )
        if alpha_mean is not None:
            out["alpha_mean"] = alpha_mean
        if alpha_std is not None:
            out["alpha_std"] = alpha_std
        if alpha_abs_max is not None:
            out["alpha_abs_max"] = alpha_abs_max
        if w_fro is not None:
            out["w_fro"] = w_fro
        return out
