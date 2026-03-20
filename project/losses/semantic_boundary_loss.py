"""Minimal semantic + edge loss with mask gating."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class SemanticBoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_loss = LovaszLoss(
            mode="multiclass",
            ignore_index=-1,
            loss_weight=1.0,
        )
        self.mask_loss = nn.BCEWithLogitsLoss()

    @staticmethod
    def _zero_like(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def forward(
        self,
        seg_logits: torch.Tensor,
        edge_pred: torch.Tensor,
        segment: torch.Tensor,
        edge: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        edge = edge.float()

        vec_pred = edge_pred[:, 0:3]
        vec_gt = edge[:, 0:3]

        strength_pred = edge_pred[:, 3]
        strength_gt = edge[:, 3]

        mask_pred = edge_pred[:, 4]
        mask_gt = edge[:, 4].float()

        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz
        loss_mask = self.mask_loss(mask_pred, mask_gt)

        positive_mask = mask_gt > 0.5
        if positive_mask.any():
            loss_vec = F.mse_loss(vec_pred[positive_mask], vec_gt[positive_mask])
            loss_strength = F.mse_loss(
                strength_pred[positive_mask], strength_gt[positive_mask]
            )
        else:
            loss_vec = self._zero_like(edge_pred)
            loss_strength = self._zero_like(edge_pred)

        total_loss = loss_semantic + loss_mask + loss_vec + loss_strength

        return dict(
            loss=total_loss,
            loss_semantic=loss_semantic,
            loss_mask=loss_mask,
            loss_vec=loss_vec,
            loss_strength=loss_strength,
        )
