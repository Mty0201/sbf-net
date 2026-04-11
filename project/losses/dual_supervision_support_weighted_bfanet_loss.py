"""Dual-supervision wrapper around SupportWeightedBFANetLoss (CR-V).

CR-V is a single-variable delta from CR-P: same dual-stream architecture
(g v4 cross-stream fusion attention), same boundary target (hard
``boundary_mask_r060``), same v1/v2 supervision scheme. The only
difference is the per-point semantic CE weight — CR-P uses
``1 + boundary_mask * 9`` (binary), CR-V uses ``1 + s_weight * 9``
(continuous, exp-decay across a 6 cm buffer shell beyond the core).

Weighting: ``loss = v1_weight * L_v1 + v2_weight * L_v2`` with defaults
1:1 matching CR-P.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .support_weighted_bfanet_loss import SupportWeightedBFANetLoss


class DualSupervisionSupportWeightedBFANetLoss(nn.Module):
    """v1 + v2 CR-V supervision wrapper."""

    def __init__(
        self,
        aux_weight: float = 1.0,
        boundary_ce_weight: float = 10.0,
        boundary_threshold: float = 0.5,
        pos_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
        v1_weight: float = 1.0,
        v2_weight: float = 1.0,
    ):
        super().__init__()
        self.v1_weight = float(v1_weight)
        self.v2_weight = float(v2_weight)
        self.base = SupportWeightedBFANetLoss(
            aux_weight=aux_weight,
            boundary_ce_weight=boundary_ce_weight,
            boundary_threshold=boundary_threshold,
            pos_weight=pos_weight,
            dice_weight=dice_weight,
            dice_smooth=dice_smooth,
        )

    def forward(
        self,
        seg_logits: torch.Tensor,
        support_pred: torch.Tensor,
        segment: torch.Tensor,
        edge: torch.Tensor | None = None,
        seg_logits_v2: torch.Tensor | None = None,
        support_pred_v2: torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
        s_weight: torch.Tensor | None = None,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        if seg_logits_v2 is None or support_pred_v2 is None:
            raise RuntimeError(
                "DualSupervisionSupportWeightedBFANetLoss requires "
                "seg_logits_v2 and support_pred_v2 in the model output. "
                "Make sure the trainer forwards these kwargs and that the "
                "model emits them (BoundaryGatedSemanticModelV4)."
            )

        out_v1 = self.base(
            seg_logits=seg_logits,
            support_pred=support_pred,
            segment=segment,
            edge=edge,
            boundary_mask=boundary_mask,
            s_weight=s_weight,
        )
        out_v2 = self.base(
            seg_logits=seg_logits_v2,
            support_pred=support_pred_v2,
            segment=segment,
            edge=edge,
            boundary_mask=boundary_mask,
            s_weight=s_weight,
        )

        total = self.v1_weight * out_v1["loss"] + self.v2_weight * out_v2["loss"]

        result: dict[str, torch.Tensor] = {"loss": total}
        for key, value in out_v1.items():
            if key == "loss":
                result["v1_loss"] = value
            else:
                result[f"v1_{key}"] = value
        for key, value in out_v2.items():
            if key == "loss":
                result["v2_loss"] = value
            else:
                result[f"v2_{key}"] = value
        return result
