"""Unweighted boundary cue loss: semantic + plain BCE on valid label.

CR-F baseline. Same as BoundaryProximityCueLoss but without support
confidence weighting — all points have equal weight in the BCE term.
Tests whether the support weighting helps or hurts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class UnweightedBoundaryCueLoss(nn.Module):
    """Two-term loss: global semantic (CE+Lovasz) + unweighted BCE auxiliary."""

    def __init__(self, aux_weight: float = 0.3):
        super().__init__()
        self.aux_weight = float(aux_weight)
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

        valid_gt = edge[:, 4].float().clamp(0.0, 1.0)

        # Term 1: Global semantic
        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz

        # Term 2: Plain BCE — no support weighting
        support_logit = support_pred[:, 0]
        loss_aux = F.binary_cross_entropy_with_logits(
            support_logit, valid_gt, reduction="mean"
        )

        loss_aux_weighted = self.aux_weight * loss_aux
        total = loss_semantic + loss_aux_weighted

        with torch.no_grad():
            support_prob = torch.sigmoid(support_logit)
            valid_sum = valid_gt.sum().clamp_min(1e-6)
            aux_prob_mean = support_prob.mean()
            aux_prob_boundary_mean = (support_prob * valid_gt).sum() / valid_sum

        return dict(
            loss=total,
            loss_semantic=loss_semantic,
            loss_ce=loss_ce,
            loss_lovasz=loss_lovasz,
            loss_aux=loss_aux,
            loss_aux_weighted=loss_aux_weighted,
            valid_ratio=valid_gt.mean(),
            aux_prob_mean=aux_prob_mean,
            aux_prob_boundary_mean=aux_prob_boundary_mean,
        )
