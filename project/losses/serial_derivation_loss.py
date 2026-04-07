"""Serial derivation loss: semantic + support BCE + offset smooth-L1.

Extends BoundaryProximityCueLoss (CR-C) with a third term that supervises
the 3D boundary offset predicted by module g. The offset ground truth is
computed as dir_gt * dist_gt from existing edge.npy columns.

See docs/canonical/part2_serial_derivation_discussion.md Section 7.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class SerialDerivationLoss(nn.Module):
    """Three-term loss: global semantic (CE+Lovasz) + confidence-weighted BCE + offset smooth-L1."""

    def __init__(self, aux_weight: float = 0.3, offset_weight: float = 1.0):
        super().__init__()
        self.aux_weight = float(aux_weight)
        self.offset_weight = float(offset_weight)

        # Term 1: Global semantic — identical to CR-A / CR-B / CR-C
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_loss = LovaszLoss(
            mode="multiclass", ignore_index=-1, loss_weight=1.0
        )

    def forward(
        self,
        seg_logits: torch.Tensor,
        support_pred: torch.Tensor,
        offset_pred: torch.Tensor,
        segment: torch.Tensor,
        edge: torch.Tensor,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        edge = edge.float()

        # Ground-truth extraction
        dir_gt = edge[:, 0:3]  # (N, 3) unit direction to boundary
        dist_gt = edge[:, 3:4]  # (N, 1) distance to boundary
        support_gt = edge[:, 4].clamp(0.0, 1.0)  # (N,)
        valid_gt = edge[:, 5].clamp(0.0, 1.0)  # (N,)
        offset_gt = dir_gt * dist_gt  # (N, 3) displacement to boundary

        # === Term 1: Global semantic (CE + Lovasz) ===
        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz

        # === Term 2: Confidence-weighted BCE for support (same as CR-C) ===
        support_logit = support_pred[:, 0]
        confidence_weight = 1.0 + support_gt
        loss_aux = F.binary_cross_entropy_with_logits(
            support_logit, valid_gt, weight=confidence_weight, reduction="mean"
        )

        # === Term 3: Offset smooth-L1 on valid points only ===
        valid_mask = valid_gt > 0.5
        if valid_mask.any():
            loss_offset = F.smooth_l1_loss(
                offset_pred[valid_mask], offset_gt[valid_mask], reduction="mean"
            )
        else:
            loss_offset = offset_pred.sum() * 0.0  # zero but keeps grad graph

        # === Total ===
        loss_aux_weighted = self.aux_weight * loss_aux
        loss_offset_weighted = self.offset_weight * loss_offset
        total = loss_semantic + loss_aux_weighted + loss_offset_weighted

        # Monitoring metrics (detached)
        with torch.no_grad():
            support_prob = torch.sigmoid(support_logit)
            valid_sum = valid_gt.sum().clamp_min(1e-6)
            aux_prob_mean = support_prob.mean()
            aux_prob_boundary_mean = (support_prob * valid_gt).sum() / valid_sum
            if valid_mask.any():
                offset_mae = (
                    (offset_pred[valid_mask] - offset_gt[valid_mask]).abs().mean()
                )
            else:
                offset_mae = torch.tensor(0.0, device=seg_logits.device)

        return dict(
            loss=total,
            loss_semantic=loss_semantic,
            loss_ce=loss_ce,
            loss_lovasz=loss_lovasz,
            loss_aux=loss_aux,
            loss_aux_weighted=loss_aux_weighted,
            loss_offset=loss_offset,
            loss_offset_weighted=loss_offset_weighted,
            valid_ratio=valid_gt.mean(),
            support_positive_ratio=(support_gt > 1e-3).float().mean(),
            aux_prob_mean=aux_prob_mean,
            aux_prob_boundary_mean=aux_prob_boundary_mean,
            offset_mae=offset_mae,
        )
