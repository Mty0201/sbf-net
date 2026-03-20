"""Minimal validation evaluator for semantic boundary stage-1 metrics."""

from __future__ import annotations

import torch

from project.losses import SemanticBoundaryLoss


class SemanticBoundaryEvaluator:
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
        mask_gt = edge[:, 4].float()
        positive_mask = mask_gt > 0.5

        mask_logit = edge_pred[:, 4]
        mask_prob = torch.sigmoid(mask_logit)
        mask_pred = mask_prob >= 0.5
        mask_gt_bool = positive_mask

        tp = (mask_pred & mask_gt_bool).sum().float()
        fp = (mask_pred & ~mask_gt_bool).sum().float()
        fn = (~mask_pred & mask_gt_bool).sum().float()

        mask_precision = self._safe_div(tp, tp + fp)
        mask_recall = self._safe_div(tp, tp + fn)
        mask_f1 = self._safe_div(
            2.0 * mask_precision * mask_recall, mask_precision + mask_recall
        )

        if positive_mask.any():
            vec_error_masked = torch.mean(
                (edge_pred[:, 0:3][positive_mask] - edge[:, 0:3][positive_mask]) ** 2
            )
            strength_error_masked = torch.mean(
                (edge_pred[:, 3][positive_mask] - edge[:, 3][positive_mask]) ** 2
            )
        else:
            vec_error_masked = self._safe_zero(edge_pred)
            strength_error_masked = self._safe_zero(edge_pred)

        return dict(
            mask_precision=mask_precision,
            mask_recall=mask_recall,
            mask_f1=mask_f1,
            vec_error_masked=vec_error_masked,
            strength_error_masked=strength_error_masked,
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
            val_loss_mask=loss_dict["loss_mask"],
            val_loss_vec=loss_dict["loss_vec"],
            val_loss_strength=loss_dict["loss_strength"],
            mask_precision=edge_metrics["mask_precision"],
            mask_recall=edge_metrics["mask_recall"],
            mask_f1=edge_metrics["mask_f1"],
            vec_error_masked=edge_metrics["vec_error_masked"],
            strength_error_masked=edge_metrics["strength_error_masked"],
        )
