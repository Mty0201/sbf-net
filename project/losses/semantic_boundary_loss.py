"""Minimal semantic + boundary direction/distance/support loss aligned with edge.npy.

Distance supervision is linearly rescaled by ``dist_scale`` for optimization, while
reported ``dist_error`` stays in the original physical unit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class SemanticBoundaryLoss(nn.Module):
    """Use edge.npy as [dir_x, dir_y, dir_z, edge_dist, edge_support, edge_valid]."""

    SUPPORT_POSITIVE_EPS = 1e-3

    def __init__(
        self,
        tau_dir: float = 1e-3,
        dist_scale: float = 0.08,
        support_weight: float = 1.0,
        support_cover_weight: float = 0.25,
        support_reg_weight: float = 1.0,
        support_tversky_alpha: float = 0.3,
        support_tversky_beta: float = 0.7,
        dir_weight: float = 1.0,
        dist_weight: float = 1.0,
        support_weighted_edge: bool = False,
    ):
        super().__init__()
        self.tau_dir = float(tau_dir)
        self.dist_scale = float(dist_scale)
        if self.dist_scale <= 0:
            raise ValueError("dist_scale must be positive.")
        self.support_weight = float(support_weight)
        self.support_cover_weight = float(support_cover_weight)
        self.support_reg_weight = float(support_reg_weight)
        self.support_tversky_alpha = float(support_tversky_alpha)
        self.support_tversky_beta = float(support_tversky_beta)
        self.dir_weight = float(dir_weight)
        self.dist_weight = float(dist_weight)
        self.support_weighted_edge = bool(support_weighted_edge)

        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_loss = LovaszLoss(
            mode="multiclass",
            ignore_index=-1,
            loss_weight=1.0,
        )
        self.support_regression_loss = nn.SmoothL1Loss(reduction="none")
        self.dist_regression_loss = nn.SmoothL1Loss(reduction="none")

    @staticmethod
    def _weighted_mean(
        value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        weight_sum = weight.sum()
        if weight_sum.item() <= 0:
            return value.sum() * 0.0
        return (value * weight).sum() / (weight_sum + eps)

    @staticmethod
    def _tversky_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        beta: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        target = target.float()
        true_positive = torch.sum(pred * target)
        false_positive = torch.sum(pred * (1.0 - target))
        false_negative = torch.sum((1.0 - pred) * target)
        return 1.0 - (true_positive + eps) / (
            true_positive + alpha * false_positive + beta * false_negative + eps
        )

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

    @staticmethod
    def _normalize_direction(direction: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return F.normalize(direction, dim=1, eps=eps)

    def forward(
        self,
        seg_logits: torch.Tensor,
        edge_pred: torch.Tensor,
        segment: torch.Tensor,
        edge: torch.Tensor,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        edge = edge.float()

        dir_pred_raw = edge_pred[:, 0:3]
        dist_pred = edge_pred[:, 3]
        support_logit = edge_pred[:, 4]

        dir_gt = edge[:, 0:3]
        dist_gt = edge[:, 3].float().clamp_min(0.0)
        support_gt = edge[:, 4].float().clamp(0.0, 1.0)
        valid_gt = edge[:, 5].float().clamp(0.0, 1.0)

        support_pred = torch.sigmoid(support_logit)
        dist_pred_scaled = dist_pred / self.dist_scale
        dist_gt_scaled = dist_gt / self.dist_scale
        support_target = support_gt * valid_gt
        support_region_gt = (valid_gt > 0.5).float()
        support_positive_mask = support_gt > self.SUPPORT_POSITIVE_EPS
        dir_valid_mask = (valid_gt > 0.5) & (dist_gt > self.tau_dir)
        dist_valid_mask = valid_gt > 0.5

        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz

        loss_support_reg = self._weighted_mean(
            self.support_regression_loss(support_pred, support_target),
            valid_gt,
        )
        loss_support_cover = self._tversky_loss(
            support_pred,
            support_region_gt,
            alpha=self.support_tversky_alpha,
            beta=self.support_tversky_beta,
        )
        # Roll support back to a continuous-field-first objective while keeping the
        # current coverage term as auxiliary regularization.
        loss_support = (
            self.support_reg_weight * loss_support_reg
            + self.support_cover_weight * loss_support_cover
        )

        # Keep the model prediction in physical distance units and only normalize inside
        # the supervision space so evaluator metrics can stay directly interpretable.
        dist_error = self.dist_regression_loss(dist_pred_scaled, dist_gt_scaled)
        if self.support_weighted_edge:
            edge_weight = support_gt * valid_gt
            loss_dist = self._weighted_mean(dist_error, edge_weight)
        else:
            loss_dist = self._weighted_mean(dist_error, valid_gt)

        dir_pred_unit = self._normalize_direction(dir_pred_raw)
        dir_gt_unit = self._normalize_direction(dir_gt)
        dir_cosine_all = torch.sum(dir_pred_unit * dir_gt_unit, dim=1).clamp(-1.0, 1.0)
        dir_error = 1.0 - dir_cosine_all
        if self.support_weighted_edge:
            dir_weight = support_gt * valid_gt
            dir_weight = dir_weight * (dist_gt > self.tau_dir).float()
            loss_dir = self._weighted_mean(dir_error, dir_weight)
        else:
            loss_dir = self._masked_mean(dir_error, dir_valid_mask, edge_pred)

        loss_edge = (
            self.support_weight * loss_support
            + self.dir_weight * loss_dir
            + self.dist_weight * loss_dist
        )
        total_loss = loss_semantic + loss_edge

        valid_ratio = valid_gt.mean()
        support_positive_ratio = self._safe_mean(support_positive_mask, edge_pred)
        dir_valid_ratio = self._safe_mean(dir_valid_mask, edge_pred)
        dist_gt_valid_mean = self._masked_mean(dist_gt, dist_valid_mask, edge_pred)
        dir_cosine = self._masked_mean(dir_cosine_all, dir_valid_mask, edge_pred)
        dist_error_valid = self._masked_mean(
            torch.abs(dist_pred - dist_gt),
            dist_valid_mask,
            edge_pred,
        )
        dist_error_scaled_valid = self._masked_mean(
            torch.abs(dist_pred_scaled - dist_gt_scaled),
            dist_valid_mask,
            edge_pred,
        )

        return dict(
            loss=total_loss,
            loss_semantic=loss_semantic,
            loss_edge=loss_edge,
            loss_support=loss_support,
            loss_support_reg=loss_support_reg,
            loss_support_cover=loss_support_cover,
            loss_dir=loss_dir,
            loss_dist=loss_dist,
            valid_ratio=valid_ratio,
            support_positive_ratio=support_positive_ratio,
            dir_valid_ratio=dir_valid_ratio,
            dist_gt_valid_mean=dist_gt_valid_mean,
            dir_cosine=dir_cosine,
            dist_error=dist_error_valid,
            dist_error_scaled=dist_error_scaled_valid,
            support_cover=1.0 - loss_support_cover,
        )
