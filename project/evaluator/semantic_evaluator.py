"""Minimal semantic-only validation evaluator."""

from __future__ import annotations

import torch

from project.losses import SemanticOnlyLoss


class SemanticEvaluator:
    def __init__(self):
        self.loss_fn = SemanticOnlyLoss()

    @staticmethod
    def _safe_zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    @staticmethod
    def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        if denominator.item() == 0:
            return numerator.new_tensor(0.0)
        return numerator / denominator

    def __call__(
        self,
        seg_logits: torch.Tensor,
        segment: torch.Tensor,
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

        valid_iou = union > 0
        valid_acc = target > 0

        iou_class = torch.zeros_like(intersection)
        acc_class = torch.zeros_like(intersection)
        iou_class[valid_iou] = intersection[valid_iou] / union[valid_iou]
        acc_class[valid_acc] = intersection[valid_acc] / target[valid_acc]

        val_mIoU = iou_class[valid_iou].mean() if valid_iou.any() else self._safe_zero(seg_logits)
        val_mAcc = acc_class[valid_acc].mean() if valid_acc.any() else self._safe_zero(seg_logits)
        val_allAcc = self._safe_div(intersection.sum(), target.sum())

        loss_dict = self.loss_fn(seg_logits=seg_logits, segment=segment)

        return dict(
            val_mIoU=val_mIoU,
            val_mAcc=val_mAcc,
            val_allAcc=val_allAcc,
            semantic_intersection=intersection,
            semantic_union=union,
            semantic_target=target,
            semantic_iou_per_class=iou_class,
            semantic_acc_per_class=acc_class,
            val_loss_semantic=loss_dict["loss_semantic"],
        )
