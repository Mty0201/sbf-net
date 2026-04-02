"""Support-guided semantic focus loss for the Phase 7 active route.

Design note — CE term overlap and intentional weighting:
    loss_semantic includes both CE and Lovasz over all points globally.
    loss_focus is an additional pointwise CE weighted by ground-truth boundary
    proximity. This means CE appears in both loss_semantic and loss_focus,
    which is intentional: the focus term is additive boundary emphasis on top
    of the global semantic supervision, not a replacement for the Lovasz
    component. Lovasz stays global (not support-weighted) to preserve its
    region-level smoothing effect across the entire prediction. The net effect
    is that boundary-region points receive stronger CE gradient while Lovasz
    continues to shape the global class distribution.

Design note — ground-truth guidance (not prediction-guided):
    The focus weighting is intentionally derived from ground-truth support_gt
    (edge[:, 4]), not from model prediction support_pred. This ensures the
    boundary emphasis is deterministic during training and does not create a
    feedback loop where the model's own predictions amplify its own errors.
    The model learns to predict support_pred via loss_support (BCE), but that
    prediction is not used to modulate the semantic loss gradient.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class SupportGuidedSemanticFocusLoss(nn.Module):
    """Three-term loss: global semantic, support BCE, support-guided semantic focus."""

    def __init__(
        self,
        support_loss_weight: float = 1.0,
        focus_loss_weight: float = 1.0,
        focus_lambda: float = 1.0,
        focus_gamma: float = 1.0,
    ):
        super().__init__()
        self.support_loss_weight = float(support_loss_weight)
        self.focus_loss_weight = float(focus_loss_weight)
        self.focus_lambda = float(focus_lambda)
        self.focus_gamma = float(focus_gamma)

        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_loss = LovaszLoss(
            mode="multiclass",
            ignore_index=-1,
            loss_weight=1.0,
        )

    def forward(self, seg_logits: torch.Tensor, support_pred: torch.Tensor,
                segment: torch.Tensor, edge: torch.Tensor,
                **_extra) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        edge = edge.float()

        # --- Ground-truth extraction ---
        support_gt = edge[:, 4].float().clamp(0.0, 1.0)
        valid_gt = edge[:, 5].float().clamp(0.0, 1.0)

        # === Term 1: Global semantic loss (CE + Lovasz) ===
        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz

        # === Term 2: Support supervision (BCE with logits) ===
        support_logit = support_pred[:, 0]
        support_target = support_gt * valid_gt
        loss_support_raw = F.binary_cross_entropy_with_logits(
            support_logit, support_target, reduction="none"
        )
        # Weighted mean over valid points only
        valid_sum = valid_gt.sum().clamp_min(1e-6)
        loss_support = (loss_support_raw * valid_gt).sum() / valid_sum

        # === Term 3: Support-guided semantic focus ===
        # NOTE: focus_weight uses ground-truth support_gt, not support_pred.
        # This is intentional — see module docstring for rationale.
        point_ce = F.cross_entropy(
            seg_logits, segment, ignore_index=-1, reduction="none"
        )
        focus_weight = 1.0 + self.focus_lambda * (support_gt * valid_gt).pow(
            self.focus_gamma
        )
        # Exclude ignore-index points from weighting
        valid_sem = segment != -1
        focus_weight = focus_weight * valid_sem.float()
        loss_focus = (point_ce * focus_weight).sum() / focus_weight.sum().clamp_min(
            1e-6
        )

        # === Total loss ===
        total_loss = (
            loss_semantic
            + self.support_loss_weight * loss_support
            + self.focus_loss_weight * loss_focus
        )

        return dict(
            loss=total_loss,
            loss_semantic=loss_semantic,
            loss_ce=loss_ce,
            loss_lovasz=loss_lovasz,
            loss_support=loss_support,
            loss_focus=loss_focus,
            valid_ratio=valid_gt.mean(),
            support_positive_ratio=(support_gt > 1e-3).float().mean(),
        )
