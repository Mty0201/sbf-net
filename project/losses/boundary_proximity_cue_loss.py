"""Boundary proximity cue loss: confidence-weighted BCE auxiliary supervision.

Replaces the SmoothL1+Tversky regression in RedesignedSupportFocusLoss with a
classification-based auxiliary loss that treats:
  - valid (edge[:, 5]) as a binary boundary/not-boundary label
  - support (edge[:, 4]) as a confidence weight for boundary points

This reinterpretation aligns the auxiliary task with semantic discrimination
(detecting class transitions) rather than geometric distance regression.
See docs/canonical/route_redesign_discussion.md Section 5.3 Option L1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class BoundaryProximityCueLoss(nn.Module):
    """Two-term loss: global semantic (CE + Lovasz) + confidence-weighted BCE auxiliary."""

    def __init__(self, aux_weight: float = 0.3):
        super().__init__()
        self.aux_weight = float(aux_weight)

        # Term 1: Global semantic — identical to CR-A and CR-B
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

        # Ground-truth extraction (same columns as RedesignedSupportFocusLoss)
        support_gt = edge[:, 4].float().clamp(0.0, 1.0)
        valid_gt = edge[:, 5].float().clamp(0.0, 1.0)

        # === Term 1: Global semantic (CE + Lovasz) ===
        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz

        # === Term 2: Confidence-weighted BCE (Option L1) ===
        # Target: valid (binary — is this point near a semantic boundary?)
        # Weight: support confidence for boundary points, 1.0 for non-boundary
        support_logit = support_pred[:, 0]
        aux_target = valid_gt
        confidence_weight = torch.where(
            valid_gt > 0.5, support_gt, torch.ones_like(support_gt)
        )

        loss_aux = F.binary_cross_entropy_with_logits(
            support_logit, aux_target, weight=confidence_weight, reduction="mean"
        )

        # === Total loss ===
        loss_aux_weighted = self.aux_weight * loss_aux
        total_loss = loss_semantic + loss_aux_weighted

        # Monitoring metrics (detached — no gradient contribution)
        with torch.no_grad():
            support_prob = torch.sigmoid(support_logit)
            valid_sum = valid_gt.sum().clamp_min(1e-6)
            aux_prob_mean = support_prob.mean()
            aux_prob_boundary_mean = (support_prob * valid_gt).sum() / valid_sum

        return dict(
            loss=total_loss,
            loss_semantic=loss_semantic,
            loss_ce=loss_ce,
            loss_lovasz=loss_lovasz,
            loss_aux=loss_aux,
            loss_aux_weighted=loss_aux_weighted,
            valid_ratio=valid_gt.mean(),
            support_positive_ratio=(support_gt > 1e-3).float().mean(),
            aux_prob_mean=aux_prob_mean,
            aux_prob_boundary_mean=aux_prob_boundary_mean,
        )
