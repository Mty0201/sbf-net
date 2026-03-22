"""Minimal semantic + boundary support/offset loss aligned with edge.npy."""

from __future__ import annotations

import torch
import torch.nn as nn

from pointcept.models.losses.lovasz import LovaszLoss


class SemanticBoundaryLoss(nn.Module):
    """Use edge.npy as [vec_x, vec_y, vec_z, edge_support, edge_valid]."""

    SUPPORT_POSITIVE_EPS = 1e-3

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

    @staticmethod
    def _safe_mean(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if mask.numel() == 0:
            return reference.sum() * 0.0
        return mask.float().mean()

    @staticmethod
    def _masked_mean(
        value: torch.Tensor, mask: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        if not mask.any():
            return reference.sum() * 0.0
        return value[mask].mean()

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
        support_positive_mask = support_gt > self.SUPPORT_POSITIVE_EPS
        active_vec_mask = support_target > self.SUPPORT_POSITIVE_EPS

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
        valid_ratio = valid_gt.mean()
        support_positive_ratio = self._safe_mean(support_positive_mask, edge_pred)
        active_vec_ratio = self._safe_mean(active_vec_mask, edge_pred)
        vec_gt_norm = torch.norm(vec_gt, dim=1)
        vec_gt_norm_active_mean = self._masked_mean(
            vec_gt_norm, active_vec_mask, edge_pred
        )
        vec_error_unweighted_active = self._masked_mean(
            vec_error, active_vec_mask, edge_pred
        )
        vec_error_weighted_active = self._masked_mean(
            vec_error * support_target,
            active_vec_mask,
            edge_pred,
        )

        loss_support = loss_support_reg + loss_support_overlap
        loss_edge = loss_support + loss_vec
        total_loss = loss_semantic + loss_edge

        return dict(
            loss=total_loss,
            loss_semantic=loss_semantic,
            loss_edge=loss_edge,
            loss_support=loss_support,
            loss_support_reg=loss_support_reg,
            loss_support_overlap=loss_support_overlap,
            loss_vec=loss_vec,
            valid_ratio=valid_ratio,
            support_positive_ratio=support_positive_ratio,
            active_vec_ratio=active_vec_ratio,
            vec_gt_norm_active_mean=vec_gt_norm_active_mean,
            vec_error_unweighted_active=vec_error_unweighted_active,
            vec_error_weighted_active=vec_error_weighted_active,
            # Legacy aliases kept only because the unchanged trainer still logs them.
            loss_mask=loss_support,
            loss_strength=loss_support_reg,
        )
