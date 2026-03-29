"""Axis + Side + Support loss: decomposes direction into unsigned axis and binary side.

Instead of regressing a signed unit-vector direction (which has hard discontinuities
at support-element boundaries where directions flip ~180°), this loss decomposes the
direction signal into:

  - **axis**: the unsigned direction axis, supervised with sign-invariant cosine loss
    ``1 - |cos(pred, gt)|``.  Continuous across support-element boundaries.
  - **side**: binary label indicating which side of the support element the point is on,
    supervised with BCE.  Derived at training time via hemisphere convention.
  - **support / magnitude**: reuses the existing support prediction and loss unchanged.

Channel layout of ``edge_pred`` (5 channels, same as existing heads):
  [axis_pred(3), side_logit(1), support_logit(1)]

GT ``edge.npy`` layout (6 columns, unchanged):
  [dir_x, dir_y, dir_z, dist, support, valid]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class AxisSideSemanticBoundaryLoss(nn.Module):
    """Semantic + boundary axis/side/support loss."""

    SUPPORT_POSITIVE_EPS = 1e-3

    def __init__(
        self,
        tau_dir: float = 1e-3,
        support_weight: float = 1.0,
        support_cover_weight: float = 0.25,
        support_reg_weight: float = 1.0,
        support_tversky_alpha: float = 0.3,
        support_tversky_beta: float = 0.7,
        axis_weight: float = 1.0,
        side_weight: float = 1.0,
        side_support_threshold: float = 0.0,
    ):
        super().__init__()
        self.tau_dir = float(tau_dir)
        self.support_weight = float(support_weight)
        self.support_cover_weight = float(support_cover_weight)
        self.support_reg_weight = float(support_reg_weight)
        self.support_tversky_alpha = float(support_tversky_alpha)
        self.support_tversky_beta = float(support_tversky_beta)
        self.axis_weight = float(axis_weight)
        self.side_weight = float(side_weight)
        self.side_support_threshold = float(side_support_threshold)

        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_loss = LovaszLoss(
            mode="multiclass",
            ignore_index=-1,
            loss_weight=1.0,
        )
        self.support_regression_loss = nn.SmoothL1Loss(reduction="none")

    # ------------------------------------------------------------------
    # helpers (same as SemanticBoundaryLoss where applicable)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # side GT derivation via hemisphere convention
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_side_gt(dir_gt: torch.Tensor) -> torch.Tensor:
        """Derive binary side label from signed direction using hemisphere convention.

        Convention: for each direction vector, find the component with the largest
        absolute value.  If that component is positive -> side = 1, else side = 0.

        Within a single support-element basin, directions are smooth so the dominant
        component is consistent -> side labels are stable.  Across the support element,
        directions flip ~180 deg -> side labels flip, which is exactly what we want.

        Returns float tensor of shape (N,) with values in {0, 1}.
        """
        abs_dir = dir_gt.abs()
        max_comp_idx = abs_dir.argmax(dim=1)  # (N,)
        max_comp_val = dir_gt.gather(1, max_comp_idx.unsqueeze(1)).squeeze(1)  # (N,)
        return (max_comp_val > 0).float()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

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

        # Parse predictions: [axis(3), side_logit(1), support_logit(1)]
        axis_pred_raw = edge_pred[:, 0:3]
        side_logit = edge_pred[:, 3]
        support_logit = edge_pred[:, 4]

        # Parse GT (unchanged edge.npy layout)
        dir_gt = edge[:, 0:3]
        dist_gt = edge[:, 3].float().clamp_min(0.0)
        support_gt = edge[:, 4].float().clamp(0.0, 1.0)
        valid_gt = edge[:, 5].float().clamp(0.0, 1.0)

        support_pred = torch.sigmoid(support_logit)
        support_target = support_gt * valid_gt
        support_region_gt = (valid_gt > 0.5).float()
        support_positive_mask = support_gt > self.SUPPORT_POSITIVE_EPS
        axis_valid_mask = (valid_gt > 0.5) & (dist_gt > self.tau_dir)

        # --- semantic loss (unchanged) ---
        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz

        # --- support loss (unchanged) ---
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
        loss_support = (
            self.support_reg_weight * loss_support_reg
            + self.support_cover_weight * loss_support_cover
        )

        # --- axis loss: sign-invariant cosine ---
        axis_pred_unit = self._normalize_direction(axis_pred_raw)
        dir_gt_unit = self._normalize_direction(dir_gt)
        axis_cosine_all = torch.sum(axis_pred_unit * dir_gt_unit, dim=1).clamp(-1.0, 1.0)
        axis_error = 1.0 - axis_cosine_all.abs()  # sign-invariant!
        # Weight axis loss by support_gt so direction supervision concentrates
        # near edges (high support) and fades away from them (low support).
        axis_weight_map = support_gt * valid_gt
        loss_axis = self._weighted_mean(axis_error, axis_weight_map)

        # --- side loss: BCE on hemisphere convention ---
        side_gt = self._derive_side_gt(dir_gt)
        side_pred = torch.sigmoid(side_logit)
        side_bce = F.binary_cross_entropy_with_logits(
            side_logit,
            side_gt,
            reduction="none",
        )
        # Side is only meaningful where axis is valid AND support is above threshold
        side_valid_mask = axis_valid_mask.clone()
        if self.side_support_threshold > 0:
            side_valid_mask = side_valid_mask & (support_gt > self.side_support_threshold)
        # Weight side loss by support_gt (same rationale as axis).
        side_weight_map = support_gt * valid_gt
        loss_side = self._weighted_mean(side_bce, side_weight_map)

        # --- combine ---
        loss_edge = (
            self.support_weight * loss_support
            + self.axis_weight * loss_axis
            + self.side_weight * loss_side
        )
        total_loss = loss_semantic + loss_edge

        # --- metrics ---
        valid_ratio = valid_gt.mean()
        support_positive_ratio = self._safe_mean(support_positive_mask, edge_pred)
        axis_valid_ratio = self._safe_mean(axis_valid_mask, edge_pred)
        axis_cosine = self._masked_mean(axis_cosine_all.abs(), axis_valid_mask, edge_pred)
        side_accuracy = self._masked_mean(
            ((side_pred > 0.5).float() == side_gt).float(),
            side_valid_mask,
            edge_pred,
        )
        # Also report the signed cosine for comparison with previous experiments
        dir_cosine = self._masked_mean(axis_cosine_all, axis_valid_mask, edge_pred)

        return dict(
            loss=total_loss,
            loss_semantic=loss_semantic,
            loss_edge=loss_edge,
            loss_support=loss_support,
            loss_support_reg=loss_support_reg,
            loss_support_cover=loss_support_cover,
            loss_axis=loss_axis,
            loss_side=loss_side,
            valid_ratio=valid_ratio,
            support_positive_ratio=support_positive_ratio,
            axis_valid_ratio=axis_valid_ratio,
            axis_cosine=axis_cosine,
            side_accuracy=side_accuracy,
            dir_cosine=dir_cosine,
            support_cover=1.0 - loss_support_cover,
        )
