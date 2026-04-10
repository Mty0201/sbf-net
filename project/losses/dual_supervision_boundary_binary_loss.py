"""Dual-supervision wrapper around BoundaryBinaryLoss (CR-M).

Runs a single ``BoundaryBinaryLoss`` instance on both the v1 (pre-attention)
and v2 (post-attention) outputs of ``BoundaryGatedSemanticModelV4``, sums the
totals at equal weight, and exposes every internal metric under prefixed
``v1_`` / ``v2_`` keys so the trainer's dynamic metric dispatch picks them
up automatically.

At step 0 the g v4 residual deltas are zero and the v2 heads are cloned
from v1, so ``v1_loss`` and ``v2_loss`` must match within numerical noise.
That acts as a canary that the CR-L baseline equivalence property holds
before training begins.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .boundary_binary_loss import BoundaryBinaryLoss


class DualSupervisionBoundaryBinaryLoss(nn.Module):
    """v1 + v2 full-loss supervision wrapper for CR-M.

    Constructor kwargs are passed through to an internal
    ``BoundaryBinaryLoss`` instance, which is stateless and reused for both
    branches.
    """

    def __init__(
        self,
        aux_weight: float = 0.3,
        boundary_ce_weight: float = 10.0,
        sample_weight_scale: float = 9.0,
        boundary_threshold: float = 0.5,
        pos_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.base = BoundaryBinaryLoss(
            aux_weight=aux_weight,
            boundary_ce_weight=boundary_ce_weight,
            sample_weight_scale=sample_weight_scale,
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
        edge: torch.Tensor,
        seg_logits_v2: torch.Tensor | None = None,
        support_pred_v2: torch.Tensor | None = None,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        """Compute the dual-supervision loss.

        ``seg_logits`` / ``support_pred`` are the v1 (pre-attention)
        predictions (aliased by the model to the trainer-compat keys).
        ``seg_logits_v2`` / ``support_pred_v2`` come from the additive
        trainer extension — they must be provided when this loss is used.
        """
        if seg_logits_v2 is None or support_pred_v2 is None:
            raise RuntimeError(
                "DualSupervisionBoundaryBinaryLoss requires seg_logits_v2 "
                "and support_pred_v2 in the model output. Make sure the "
                "trainer forwards these kwargs and that the model emits "
                "them (BoundaryGatedSemanticModelV4)."
            )

        out_v1 = self.base(
            seg_logits=seg_logits,
            support_pred=support_pred,
            segment=segment,
            edge=edge,
        )
        out_v2 = self.base(
            seg_logits=seg_logits_v2,
            support_pred=support_pred_v2,
            segment=segment,
            edge=edge,
        )

        total = out_v1["loss"] + out_v2["loss"]

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
