"""Minimal semantic + boundary support/offset loss aligned with edge.npy."""

from __future__ import annotations

import torch
import torch.nn as nn

from pointcept.models.losses.lovasz import LovaszLoss


class SemanticBoundaryLoss(nn.Module):
    """Use edge.npy as [vec_x, vec_y, vec_z, edge_support, edge_valid]."""

    def __init__(self):
        super().__init__()
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_loss = LovaszLoss(
            mode="multiclass",
            ignore_index=-1,
            loss_weight=1.0,
        )
        self.support_regression_loss = nn.SmoothL1Loss(reduction="none")
        self.vec_regression_loss = nn.SmoothL1Loss(reduction="none")

    @staticmethod
    def _weighted_mean(
        value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        weight_sum = weight.sum()
        if weight_sum.item() <= 0:
            return value.sum() * 0.0
        return (value * weight).sum() / (weight_sum + eps)

    @staticmethod
    def _soft_dice_loss(
        pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        intersection = torch.sum(pred * target)
        denominator = torch.sum(pred) + torch.sum(target)
        return 1.0 - (2.0 * intersection + eps) / (denominator + eps)

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
        support_logit = edge_pred[:, 3]
        support_pred = torch.sigmoid(support_logit)
        support_gt = edge[:, 3].float().clamp(0.0, 1.0)
        valid_gt = edge[:, 4].float().clamp(0.0, 1.0)
        support_target = support_gt * valid_gt

        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz
        loss_support_reg = self._weighted_mean(
            self.support_regression_loss(support_pred, support_target),
            valid_gt,
        )
        loss_support_overlap = self._soft_dice_loss(
            support_pred * valid_gt,
            support_target,
        )

        vec_error = self.vec_regression_loss(vec_pred, vec_gt).mean(dim=1)
        loss_vec = self._weighted_mean(vec_error, support_target)

        loss_support = loss_support_reg + loss_support_overlap
        total_loss = loss_semantic + loss_support + loss_vec

        return dict(
            loss=total_loss,
            loss_semantic=loss_semantic,
            loss_support=loss_support,
            loss_support_reg=loss_support_reg,
            loss_support_overlap=loss_support_overlap,
            loss_vec=loss_vec,
            # Legacy aliases kept only because the unchanged trainer still logs them.
            loss_mask=loss_support,
            loss_strength=loss_support_reg,
        )
