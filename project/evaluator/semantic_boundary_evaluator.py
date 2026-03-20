"""Minimal validation evaluator aligned with support/vec/valid edge targets."""

from __future__ import annotations

import torch

from project.losses import SemanticBoundaryLoss


class SemanticBoundaryEvaluator:
    """Keep trainer-facing legacy keys while evaluating support/vec semantics."""

    def __init__(self):
        self.loss_fn = SemanticBoundaryLoss()

    @staticmethod
    def _safe_zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    @staticmethod
    def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        if denominator.item() == 0:
            return numerator.new_tensor(0.0)
        return numerator / denominator

    def _compute_semantic_stats(
        self, seg_logits: torch.Tensor, segment: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        pred = seg_logits.argmax(dim=1)
        num_classes = seg_logits.shape[1]

        intersection = torch.zeros(num_classes, device=seg_logits.device)
        union = torch.zeros(num_classes, device=seg_logits.device)
        target = torch.zeros(num_classes, device=seg_logits.device)

        for class_id in range(num_classes):
            pred_i = pred == class_id
            gt_i = segment == class_id
            intersection[class_id] = (pred_i & gt_i).sum()
            union[class_id] = (pred_i | gt_i).sum()
            target[class_id] = gt_i.sum()

        return dict(
            semantic_intersection=intersection,
            semantic_union=union,
            semantic_target=target,
        )

    def _compute_semantic_metrics_from_stats(
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

        val_mIoU = (
            iou_class[valid_iou].mean()
            if valid_iou.any()
            else self._safe_zero(reference)
        )
        val_mAcc = (
            acc_class[valid_acc].mean()
            if valid_acc.any()
            else self._safe_zero(reference)
        )
        val_allAcc = self._safe_div(intersection.sum(), target.sum())

        return dict(
            val_mIoU=val_mIoU,
            val_mAcc=val_mAcc,
            val_allAcc=val_allAcc,
            semantic_iou_per_class=iou_class,
            semantic_acc_per_class=acc_class,
        )

    def _compute_edge_metrics(
        self, edge_pred: torch.Tensor, edge: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        edge = edge.float()
        support_gt = edge[:, 3].float().clamp(0.0, 1.0)
        valid_gt = edge[:, 4].float().clamp(0.0, 1.0)
        support_target = support_gt * valid_gt
        support_pred = torch.sigmoid(edge_pred[:, 3])
        support_region_gt = support_target > 0

        support_overlap = 1.0 - self.loss_fn._soft_dice_loss(
            support_pred * valid_gt,
            support_target,
        )
        if valid_gt.any():
            support_error = self.loss_fn._weighted_mean(
                (support_pred - support_target) ** 2,
                valid_gt,
            )
        else:
            support_error = self._safe_zero(edge_pred)

        if support_region_gt.any():
            vec_error_masked = torch.mean(
                ((edge_pred[:, 0:3] - edge[:, 0:3]) ** 2)[support_region_gt]
            )
        else:
            vec_error_masked = self._safe_zero(edge_pred)

        return dict(
            support_overlap=support_overlap,
            support_error=support_error,
            vec_error_masked=vec_error_masked,
        )

    def __call__(
        self,
        seg_logits: torch.Tensor,
        edge_pred: torch.Tensor,
        segment: torch.Tensor,
        edge: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_dict = self.loss_fn(
            seg_logits=seg_logits,
            edge_pred=edge_pred,
            segment=segment,
            edge=edge,
        )
        semantic_stats = self._compute_semantic_stats(seg_logits, segment)
        semantic_metrics = self._compute_semantic_metrics_from_stats(
            intersection=semantic_stats["semantic_intersection"],
            union=semantic_stats["semantic_union"],
            target=semantic_stats["semantic_target"],
            reference=seg_logits,
        )
        edge_metrics = self._compute_edge_metrics(edge_pred, edge)

        return dict(
            val_mIoU=semantic_metrics["val_mIoU"],
            val_mAcc=semantic_metrics["val_mAcc"],
            val_allAcc=semantic_metrics["val_allAcc"],
            semantic_intersection=semantic_stats["semantic_intersection"],
            semantic_union=semantic_stats["semantic_union"],
            semantic_target=semantic_stats["semantic_target"],
            semantic_iou_per_class=semantic_metrics["semantic_iou_per_class"],
            semantic_acc_per_class=semantic_metrics["semantic_acc_per_class"],
            # Legacy metric names are preserved only for trainer compatibility.
            val_loss_mask=loss_dict["loss_support"],
            val_loss_vec=loss_dict["loss_vec"],
            val_loss_strength=loss_dict["loss_support_reg"],
            mask_precision=edge_metrics["support_overlap"],
            mask_recall=edge_metrics["support_overlap"],
            mask_f1=edge_metrics["support_overlap"],
            vec_error_masked=edge_metrics["vec_error_masked"],
            strength_error_masked=edge_metrics["support_error"],
        )
