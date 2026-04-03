"""Redesigned support-focused evaluator replacing Phase 7 SupportGuidedSemanticFocusEvaluator.

Reports SmoothL1 regression error instead of BCE for support metrics,
matching the loss redesign from Phase 10. All other metrics (global semantic,
boundary-region semantic, Tversky coverage) remain identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from project.losses import SemanticOnlyLoss


class RedesignedSupportFocusEvaluator:
    """Evaluator for the redesigned support-focused loss route."""

    def __init__(self, boundary_metric_threshold: float = 0.2):
        self.boundary_metric_threshold = float(boundary_metric_threshold)
        self.loss_fn = SemanticOnlyLoss()
        self.support_regression_loss = nn.SmoothL1Loss(reduction="none")

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
        """Compute per-class intersection/union/target, optionally masked."""
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
        """Compute mIoU, mAcc, allAcc from per-class stats."""
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
        support_pred: torch.Tensor,
        segment: torch.Tensor,
        edge: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        edge = edge.float()
        pred = seg_logits.argmax(dim=1)
        num_classes = seg_logits.shape[1]

        # --- Ground-truth extraction ---
        support_gt = edge[:, 4].float().clamp(0.0, 1.0)
        valid_gt = edge[:, 5].float().clamp(0.0, 1.0)

        # === 1. Global semantic metrics ===
        global_stats = self._compute_per_class_stats(pred, segment, num_classes)
        global_metrics = self._metrics_from_stats(
            global_stats["intersection"],
            global_stats["union"],
            global_stats["target"],
            seg_logits,
        )

        # val_loss_semantic from SemanticOnlyLoss
        loss_dict = self.loss_fn(seg_logits=seg_logits, segment=segment)

        # === 2. Boundary-region semantic metrics ===
        boundary_mask = support_gt >= self.boundary_metric_threshold
        boundary_point_ratio = boundary_mask.float().mean()

        if boundary_mask.sum() > 0:
            boundary_stats = self._compute_per_class_stats(
                pred, segment, num_classes, mask=boundary_mask
            )
            boundary_metrics = self._metrics_from_stats(
                boundary_stats["intersection"],
                boundary_stats["union"],
                boundary_stats["target"],
                seg_logits,
            )
            val_boundary_mIoU = boundary_metrics["mIoU"]
            val_boundary_mAcc = boundary_metrics["mAcc"]
        else:
            val_boundary_mIoU = seg_logits.new_tensor(0.0)
            val_boundary_mAcc = seg_logits.new_tensor(0.0)

        # === 3. Support metrics (SmoothL1 regression error, NOT BCE) ===
        prob = torch.sigmoid(support_pred[:, 0])
        support_target = support_gt * valid_gt
        valid_sum = valid_gt.sum()

        if valid_sum > 0:
            reg_raw = F.smooth_l1_loss(prob, support_target, reduction="none")
            support_reg_error = (reg_raw * valid_gt).sum() / valid_sum.clamp_min(1e-6)
        else:
            support_reg_error = seg_logits.new_tensor(0.0)

        # support_cover: Tversky overlap on sigmoid probabilities (D-06 consistent)
        target_binary = (valid_gt > 0.5).float()
        if target_binary.sum() > 0:
            tp = (prob * target_binary).sum()
            fp = (prob * (1.0 - target_binary)).sum()
            fn = ((1.0 - prob) * target_binary).sum()
            support_cover = tp / (tp + 0.3 * fp + 0.7 * fn + 1e-6)
        else:
            support_cover = seg_logits.new_tensor(0.0)

        valid_ratio = valid_gt.mean()
        support_positive_ratio = (support_gt > 1e-3).float().mean()

        return dict(
            # Global semantic
            val_mIoU=global_metrics["mIoU"],
            val_mAcc=global_metrics["mAcc"],
            val_allAcc=global_metrics["allAcc"],
            semantic_intersection=global_stats["intersection"],
            semantic_union=global_stats["union"],
            semantic_target=global_stats["target"],
            semantic_iou_per_class=global_metrics["iou_per_class"],
            semantic_acc_per_class=global_metrics["acc_per_class"],
            val_loss_semantic=loss_dict["loss_semantic"],
            # Boundary-region semantic
            val_boundary_mIoU=val_boundary_mIoU,
            val_boundary_mAcc=val_boundary_mAcc,
            boundary_point_ratio=boundary_point_ratio,
            # Support metrics (SmoothL1 replaces BCE)
            support_reg_error=support_reg_error,
            support_cover=support_cover,
            valid_ratio=valid_ratio,
            support_positive_ratio=support_positive_ratio,
        )
