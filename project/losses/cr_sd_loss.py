"""CR-SD loss for DecoupledBFANetSegmentorV1.

Computes the four loss terms that the segmentor used to compute internally:
  - semantic CE on seg_logits vs segment
  - multiclass Lovasz on seg_logits vs segment
  - margin BCE-with-logits on marg_logits vs boundary_mask (valid-masked)
  - binary Lovasz on marg_logits vs boundary_mask (valid-masked)

Only points with segment != -1 are used for the margin terms, matching the
segmentor's original in-place logic. Returns a dict with `loss` plus per-term
entries so the trainer can log them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pointcept.models.losses.lovasz import LovaszLoss


class CRSDLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_multi = LovaszLoss(
            mode="multiclass", loss_weight=1.0, ignore_index=-1
        )
        self.margin_bce = nn.BCEWithLogitsLoss(reduction="none")
        self.margin_lovasz = LovaszLoss(mode="binary", loss_weight=1.0)

    def forward(
        self,
        seg_logits: torch.Tensor,
        segment: torch.Tensor,
        marg_logits: torch.Tensor,
        boundary_mask: torch.Tensor,
        alpha_mean: torch.Tensor | None = None,
        alpha_std: torch.Tensor | None = None,
        alpha_abs_max: torch.Tensor | None = None,
        w_fro: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        valid_mask = segment != -1
        boundary_valid = boundary_mask.float().view(-1)[valid_mask]
        marg_valid = marg_logits.view(-1)[valid_mask]

        loss_semantic = self.ce_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_multi(seg_logits, segment)
        loss_marg_bce = self.margin_bce(marg_valid, boundary_valid).mean()
        loss_marg_lovasz = self.margin_lovasz(marg_valid, boundary_valid)

        total = loss_semantic + loss_lovasz + loss_marg_bce + loss_marg_lovasz
        out = dict(
            loss=total,
            loss_semantic=loss_semantic,
            loss_lovasz=loss_lovasz,
            loss_marg_bce=loss_marg_bce,
            loss_marg_lovasz=loss_marg_lovasz,
        )
        if alpha_mean is not None:
            out["alpha_mean"] = alpha_mean
        if alpha_std is not None:
            out["alpha_std"] = alpha_std
        if alpha_abs_max is not None:
            out["alpha_abs_max"] = alpha_abs_max
        if w_fro is not None:
            out["w_fro"] = w_fro
        return out
