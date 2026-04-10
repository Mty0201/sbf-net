"""Dual-supervision wrapper around PureBFANetLoss (CR-P).

Runs a single ``PureBFANetLoss`` instance on both the v1 (pre-attention)
and v2 (post-attention) outputs of ``BoundaryGatedSemanticModelV4`` with
equal weights — BFANet's original ``train.py`` lines 187-188 sum all
eight loss terms (sem_v1 + dice_v1 + margin_v1 + margin_dice_v1 +
sem_v2 + dice_v2 + margin_v2 + margin_dice_v2) directly without any
v1/v2 scaling, so defaults are ``v1_weight=1.0`` and ``v2_weight=1.0``.

Internal telemetry exposes every ``PureBFANetLoss`` metric under prefixed
``v1_`` / ``v2_`` keys, picked up by the trainer's dynamic metric dispatch.

At step 0 the g v4 residual deltas are zero and the v2 heads are cloned
from v1, so ``v1_loss`` and ``v2_loss`` must match within numerical noise.

Loss design is BFANet-faithful (CR-N semantics), doubled over v1/v2:
  - Semantic CE with hard-mask 10x upweight on radius-search boundary
  - Unweighted BCE (pos_weight=1.0, no sample_weight_scale)
  - Global Dice over the whole output
  - Radius-search boundary mask from precomputed ``boundary_mask_r060.npy``

This is the "full BFANet reproduction" control: faithful boundary
detection + faithful loss + faithful dual-layer supervision.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .pure_bfanet_loss import PureBFANetLoss


class DualSupervisionPureBFANetLoss(nn.Module):
    """v1 + v2 BFANet-faithful supervision wrapper for CR-P.

    Weighting: ``loss = v1_weight * L_v1 + v2_weight * L_v2``.
    Defaults (``v1_weight=1.0``, ``v2_weight=1.0``) match BFANet's original
    dual-supervision scheme where all eight loss terms are summed directly
    without per-stream scaling (``train.py`` lines 187-188).
    """

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
        self.base = PureBFANetLoss(
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
        edge: torch.Tensor,
        seg_logits_v2: torch.Tensor | None = None,
        support_pred_v2: torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        """Compute the dual-supervision BFANet-faithful loss.

        ``seg_logits`` / ``support_pred`` are the v1 (pre-attention)
        predictions. ``seg_logits_v2`` / ``support_pred_v2`` come from
        the additive trainer extension — they must be provided when
        this loss is used. ``boundary_mask`` is the precomputed
        radius-search target; ``edge`` is kept for the PureBFANetLoss
        fallback path and is ignored when ``boundary_mask`` is present.
        """
        if seg_logits_v2 is None or support_pred_v2 is None:
            raise RuntimeError(
                "DualSupervisionPureBFANetLoss requires seg_logits_v2 "
                "and support_pred_v2 in the model output. Make sure the "
                "trainer forwards these kwargs and that the model emits "
                "them (BoundaryGatedSemanticModelV4)."
            )

        out_v1 = self.base(
            seg_logits=seg_logits,
            support_pred=support_pred,
            segment=segment,
            edge=edge,
            boundary_mask=boundary_mask,
        )
        out_v2 = self.base(
            seg_logits=seg_logits_v2,
            support_pred=support_pred_v2,
            segment=segment,
            edge=edge,
            boundary_mask=boundary_mask,
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
