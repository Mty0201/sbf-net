"""Redesigned support-focused loss replacing Phase 7 SupportGuidedSemanticFocusLoss.

Fixes three confirmed problems from Phase 9 analysis:
1. Replaces BCE (entropy-floor saturated) with SmoothL1 + Tversky for support supervision
2. Replaces 1+-base focus term (93% redundant CE) with optional Lovasz-on-boundary
3. Supports both Variant C (focus_mode="none") and Variant A (focus_mode="lovasz")

The SmoothL1 + Tversky pattern is copied from SemanticBoundaryLoss (the support-only
baseline loss), which demonstrated effective support learning: 68% loss reduction over
300 epochs vs. 6.7% with BCE.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pointcept.models.losses.lovasz import LovaszLoss


class RedesignedSupportFocusLoss(nn.Module):
    """Three-term loss: global semantic, SmoothL1+Tversky support, optional Lovasz focus."""

    def __init__(
        self,
        support_reg_weight: float = 1.0,
        support_cover_weight: float = 0.2,
        support_tversky_alpha: float = 0.3,
        support_tversky_beta: float = 0.7,
        focus_mode: str = "none",
        focus_weight: float = 0.5,
        boundary_threshold: float = 0.1,
    ):
        super().__init__()
        if focus_mode not in ("none", "lovasz"):
            raise ValueError(
                f"focus_mode must be 'none' or 'lovasz', got '{focus_mode}'"
            )

        self.support_reg_weight = float(support_reg_weight)
        self.support_cover_weight = float(support_cover_weight)
        self.support_tversky_alpha = float(support_tversky_alpha)
        self.support_tversky_beta = float(support_tversky_beta)
        self.focus_mode = focus_mode
        self.focus_weight = float(focus_weight)
        self.boundary_threshold = float(boundary_threshold)

        # Term 1: Global semantic
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_loss = LovaszLoss(
            mode="multiclass", ignore_index=-1, loss_weight=1.0
        )

        # Term 2: Support regression
        self.support_regression_loss = nn.SmoothL1Loss(reduction="none")

        # Term 3: Optional boundary-subset Lovasz focus
        if self.focus_mode == "lovasz":
            self.boundary_lovasz = LovaszLoss(
                mode="multiclass", ignore_index=-1, loss_weight=1.0
            )

    # -- Static helpers copied from SemanticBoundaryLoss --

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

        # Ground-truth extraction
        support_gt = edge[:, 4].float().clamp(0.0, 1.0)
        valid_gt = edge[:, 5].float().clamp(0.0, 1.0)

        # Support prediction: apply sigmoid to raw logit (D-05)
        support_logit = support_pred[:, 0]
        support_prob = torch.sigmoid(support_logit)

        # === Term 1: Global semantic (CE + Lovasz) ===
        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz

        # === Term 2: Support SmoothL1 + Tversky (D-01, D-05, D-06) ===
        support_target = support_gt * valid_gt  # continuous regression target
        support_region_gt = (valid_gt > 0.5).float()  # binary Tversky target (D-06)

        loss_support_reg = self._weighted_mean(
            self.support_regression_loss(support_prob, support_target), valid_gt
        )
        loss_support_cover = self._tversky_loss(
            support_prob,
            support_region_gt,
            alpha=self.support_tversky_alpha,
            beta=self.support_tversky_beta,
        )
        loss_support = (
            self.support_reg_weight * loss_support_reg
            + self.support_cover_weight * loss_support_cover
        )

        # === Term 3: Focus (conditional on focus_mode) ===
        if self.focus_mode == "lovasz":
            # D-09, D-10: Lovasz on boundary subset only
            boundary_mask = (support_gt > self.boundary_threshold) & (segment != -1)
            if boundary_mask.sum() > 0:
                seg_logits_boundary = seg_logits[boundary_mask]
                segment_boundary = segment[boundary_mask]
                loss_focus = self.boundary_lovasz(
                    seg_logits_boundary, segment_boundary
                )
            else:
                loss_focus = seg_logits.new_tensor(0.0)
        else:
            loss_focus = seg_logits.new_tensor(0.0)

        # === Total loss ===
        # D-02 (Variant C: no focus), D-13 (Variant A: with focus)
        total_loss = loss_semantic + loss_support
        if self.focus_mode != "none":
            total_loss = total_loss + self.focus_weight * loss_focus

        return dict(
            loss=total_loss,
            loss_semantic=loss_semantic,
            loss_ce=loss_ce,
            loss_lovasz=loss_lovasz,
            loss_support=loss_support,
            loss_support_reg=loss_support_reg,
            loss_support_cover=loss_support_cover,
            loss_focus=loss_focus,
            valid_ratio=valid_gt.mean(),
            support_positive_ratio=(support_gt > 1e-3).float().mean(),
            support_cover=1.0 - loss_support_cover,
        )
