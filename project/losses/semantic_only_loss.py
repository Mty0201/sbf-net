"""Minimal semantic-only loss."""

from __future__ import annotations

import torch
import torch.nn as nn


class SemanticOnlyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        seg_logits: torch.Tensor,
        segment: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        loss_semantic = self.semantic_loss(seg_logits, segment)
        return dict(
            loss=loss_semantic,
            loss_semantic=loss_semantic,
        )
