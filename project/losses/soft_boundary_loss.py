"""Soft boundary loss: semantic + BCE with support as continuous target.

CR-G experiment. Uses edge_support (Gaussian decay, σ=0.02m) as the BCE
target instead of the binary valid label. This eliminates the hard valid
boundary that has no geometric meaning, replacing it with a smooth
distance-aware signal that naturally concentrates gradient on points
closest to the true semantic boundary.

Key differences from prior losses:
  - vs CR-B (SmoothL1+Tversky): BCE gradient is σ(z)-t, no sigmoid
    saturation fight; no Tversky reintroducing hard boundary.
  - vs CR-C (weighted BCE on valid): target is continuous support, not
    binary valid. No meaningless valid boundary for the network to learn.
  - vs CR-F (unweighted BCE on valid): same elimination of valid boundary,
    plus support concentrates gradient near true edge.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class SoftBoundaryLoss(nn.Module):
    """Two-term loss: global semantic (CE+Lovasz) + BCE with support as target."""

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

        support_gt = edge[:, 3].float().clamp(0.0, 1.0)

        # Term 1: Global semantic
        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz

        # Term 2: BCE with support as continuous target
        support_logit = support_pred[:, 0]
        loss_aux = F.binary_cross_entropy_with_logits(
            support_logit, support_gt, reduction="mean"
        )

        loss_aux_weighted = self.aux_weight * loss_aux
        total = loss_semantic + loss_aux_weighted

        with torch.no_grad():
            support_prob = torch.sigmoid(support_logit)
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
            valid_ratio=(support_gt > 0.5).float().mean(),
            support_positive_ratio=(support_gt > 1e-3).float().mean(),
            aux_prob_mean=aux_prob_mean,
            aux_prob_boundary_mean=aux_prob_boundary_mean,
        )
