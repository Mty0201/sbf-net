"""Serial derivation only loss: semantic + offset smooth-L1, no support BCE.

Two-term loss for CR-E baseline. Tests whether pure serial derivation
(offset from semantic logits) suffices without the support/proximity
BCE regularizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class SerialDerivationOnlyLoss(nn.Module):
    """Two-term loss: global semantic (CE+Lovasz) + offset smooth-L1."""

    def __init__(self, offset_weight: float = 1.0):
        super().__init__()
        self.offset_weight = float(offset_weight)
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_loss = LovaszLoss(
            mode="multiclass", ignore_index=-1, loss_weight=1.0
        )

    def forward(
        self,
        seg_logits: torch.Tensor,
        offset_pred: torch.Tensor,
        segment: torch.Tensor,
        edge: torch.Tensor,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        edge = edge.float()

        dir_gt = edge[:, 0:3]
        dist_gt = edge[:, 3:4]
        valid_gt = edge[:, 5].clamp(0.0, 1.0)
        offset_gt = dir_gt * dist_gt

        # Term 1: Global semantic
        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz

        # Term 2: Offset smooth-L1 on valid points only
        valid_mask = valid_gt > 0.5
        if valid_mask.any():
            loss_offset = F.smooth_l1_loss(
                offset_pred[valid_mask], offset_gt[valid_mask], reduction="mean"
            )
        else:
            loss_offset = offset_pred.sum() * 0.0

        loss_offset_weighted = self.offset_weight * loss_offset
        total = loss_semantic + loss_offset_weighted

        with torch.no_grad():
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
            loss_offset=loss_offset,
            loss_offset_weighted=loss_offset_weighted,
            valid_ratio=valid_gt.mean(),
            offset_mae=offset_mae,
        )
