"""ZAHA-flavored evaluator for models that emit ``support_pred`` but whose
dataset provides ``boundary_mask`` / ``s_weight`` instead of an ``edge`` tensor.

Differences vs ``RedesignedSupportFocusEvaluator``:
- Uses ``boundary_mask`` (binary) for the boundary-region semantic split.
- Uses ``s_weight`` (continuous, already clamped to [0, 1] upstream) as the
  support regression target. Falls back to ``boundary_mask`` when ``s_weight``
  is absent.
- ``support_pred`` is optional — semantic metrics are always produced so the
  evaluator also works as a drop-in for semantic-only runs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from project.losses import SemanticOnlyLoss


class ZAHASupportFocusEvaluator:
    """Support-focused evaluator keyed on ``boundary_mask`` / ``s_weight``."""

    def __init__(self, boundary_metric_threshold: float = 0.5):
        self.boundary_metric_threshold = float(boundary_metric_threshold)
        self.loss_fn = SemanticOnlyLoss()

    @staticmethod
    def _safe_zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    @staticmethod
    def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        if denominator.item() == 0:
            return numerator.new_tensor(0.0)
        return numerator / denominator

    def _compute_per_class_stats(
        self,
        pred: torch.Tensor,
        segment: torch.Tensor,
        num_classes: int,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        intersection = torch.zeros(num_classes, device=pred.device)
        union = torch.zeros(num_classes, device=pred.device)
        target = torch.zeros(num_classes, device=pred.device)

        for class_id in range(num_classes):
            pred_i = pred == class_id
            gt_i = segment == class_id
            if mask is not None:
                pred_i = pred_i & mask
                gt_i = gt_i & mask
            intersection[class_id] = (pred_i & gt_i).sum()
            union[class_id] = (pred_i | gt_i).sum()
            target[class_id] = gt_i.sum()
        return dict(intersection=intersection, union=union, target=target)

    def _metrics_from_stats(
        self,
        intersection: torch.Tensor,
        union: torch.Tensor,
        target: torch.Tensor,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        valid_iou = union > 0
        valid_acc = target > 0

        iou_class = torch.zeros_like(intersection)
        acc_class = torch.zeros_like(intersection)
        iou_class[valid_iou] = intersection[valid_iou] / union[valid_iou]
        acc_class[valid_acc] = intersection[valid_acc] / target[valid_acc]

        mIoU = (
            iou_class[valid_iou].mean()
            if valid_iou.any()
            else self._safe_zero(reference)
        )
        mAcc = (
            acc_class[valid_acc].mean()
            if valid_acc.any()
            else self._safe_zero(reference)
        )
        allAcc = self._safe_div(intersection.sum(), target.sum())
        return dict(
            mIoU=mIoU, mAcc=mAcc, allAcc=allAcc,
            iou_per_class=iou_class, acc_per_class=acc_class,
        )

    def __call__(
        self,
        seg_logits: torch.Tensor,
        segment: torch.Tensor,
        support_pred: torch.Tensor | None = None,
        marg_logits: torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
        s_weight: torch.Tensor | None = None,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        # Accept ``marg_logits`` as an alias for ``support_pred`` — the
        # CR-SD / CR-SDE segmentors expose the margin head under that name.
        if support_pred is None and marg_logits is not None:
            support_pred = marg_logits
        segment = segment.reshape(-1).long()
        pred = seg_logits.argmax(dim=1)
        num_classes = seg_logits.shape[1]

        # === 1. Global semantic metrics ===
        global_stats = self._compute_per_class_stats(pred, segment, num_classes)
        global_metrics = self._metrics_from_stats(
            global_stats["intersection"],
            global_stats["union"],
            global_stats["target"],
            seg_logits,
        )
        loss_dict = self.loss_fn(seg_logits=seg_logits, segment=segment)

        result = dict(
            val_mIoU=global_metrics["mIoU"],
            val_mAcc=global_metrics["mAcc"],
            val_allAcc=global_metrics["allAcc"],
            semantic_intersection=global_stats["intersection"],
            semantic_union=global_stats["union"],
            semantic_target=global_stats["target"],
            semantic_iou_per_class=global_metrics["iou_per_class"],
            semantic_acc_per_class=global_metrics["acc_per_class"],
            val_loss_semantic=loss_dict["loss_semantic"],
        )

        # === 2. Boundary-region semantic metrics (needs boundary_mask) ===
        if boundary_mask is not None:
            bmask_bool = boundary_mask.reshape(-1).float() >= self.boundary_metric_threshold
            result["boundary_point_ratio"] = bmask_bool.float().mean()
            if bmask_bool.sum() > 0:
                boundary_stats = self._compute_per_class_stats(
                    pred, segment, num_classes, mask=bmask_bool
                )
                boundary_metrics = self._metrics_from_stats(
                    boundary_stats["intersection"],
                    boundary_stats["union"],
                    boundary_stats["target"],
                    seg_logits,
                )
                result["val_boundary_mIoU"] = boundary_metrics["mIoU"]
                result["val_boundary_mAcc"] = boundary_metrics["mAcc"]
            else:
                result["val_boundary_mIoU"] = seg_logits.new_tensor(0.0)
                result["val_boundary_mAcc"] = seg_logits.new_tensor(0.0)

        # === 3. Support-head metrics (needs support_pred) ===
        if support_pred is not None:
            prob = torch.sigmoid(support_pred[:, 0].reshape(-1))

            target_soft: torch.Tensor | None = None
            if s_weight is not None:
                target_soft = s_weight.reshape(-1).float().clamp(0.0, 1.0)
            elif boundary_mask is not None:
                target_soft = boundary_mask.reshape(-1).float().clamp(0.0, 1.0)

            if target_soft is not None:
                reg_raw = F.smooth_l1_loss(prob, target_soft, reduction="mean")
                result["support_reg_error"] = reg_raw

                target_binary = (target_soft > 0.5).float()
                if target_binary.sum() > 0:
                    tp = (prob * target_binary).sum()
                    fp = (prob * (1.0 - target_binary)).sum()
                    fn = ((1.0 - prob) * target_binary).sum()
                    result["support_cover"] = tp / (tp + 0.3 * fp + 0.7 * fn + 1e-6)
                else:
                    result["support_cover"] = seg_logits.new_tensor(0.0)
                result["support_positive_ratio"] = target_binary.mean()

        return result
