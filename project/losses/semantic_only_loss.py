"""Minimal semantic-only loss."""

from __future__ import annotations

import torch
import torch.nn as nn

from pointcept.models.losses.lovasz import LovaszLoss


class SemanticOnlyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_loss = LovaszLoss(
            mode="multiclass",
            ignore_index=-1,
            loss_weight=1.0,
        )

    def forward(
        self,
        seg_logits: torch.Tensor,
        segment: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        loss_semantic = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        total_loss = loss_semantic + loss_lovasz
        return dict(
            loss=total_loss,
            loss_semantic=loss_semantic,
            loss_lovasz=loss_lovasz,
        )
