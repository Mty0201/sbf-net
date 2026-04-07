"""Minimal validation evaluator aligned with direction/dist/support/valid edge targets.

The evaluator keeps ``dist_error`` in the original physical distance unit even though
training uses a linearly rescaled distance target for ``loss_dist``.
"""

from __future__ import annotations

import torch

from project.losses import SemanticBoundaryLoss


class SemanticBoundaryEvaluator:
    """Keep trainer-facing evaluation simple while exposing new edge semantics."""

    def __init__(
        self,
        tau_dir: float = 1e-3,
        dist_scale: float = 0.08,
        support_weight: float = 1.0,
        support_cover_weight: float = 1.0,
        support_reg_weight: float = 0.25,
        support_tversky_alpha: float = 0.3,
        support_tversky_beta: float = 0.7,
        dir_weight: float = 1.0,
        dist_weight: float = 1.0,
    ):
        self.loss_fn = SemanticBoundaryLoss(
            tau_dir=tau_dir,
            dist_scale=dist_scale,
            support_weight=support_weight,
            support_cover_weight=support_cover_weight,
            support_reg_weight=support_reg_weight,
            support_tversky_alpha=support_tversky_alpha,
            support_tversky_beta=support_tversky_beta,
            dir_weight=dir_weight,
            dist_weight=dist_weight,
        )

    @staticmethod
    def _safe_zero(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    @staticmethod
    def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        if denominator.item() == 0:
            return numerator.new_tensor(0.0)
        return numerator / denominator

    @staticmethod
    def _masked_mean(
        value: torch.Tensor, mask: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        if not mask.any():
            return reference.sum() * 0.0
        return value[mask].mean()

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
        vec_gt = edge[:, 0:3]
        support_gt = edge[:, 3].float().clamp(0.0, 1.0)
        valid_gt = edge[:, 4].float().clamp(0.0, 1.0)
        dist_gt = torch.linalg.norm(vec_gt, dim=1).clamp_min(0.0)
        dir_gt = torch.nn.functional.normalize(vec_gt, dim=1, eps=1e-6)

        dir_pred_raw = edge_pred[:, 0:3]
        dist_pred = edge_pred[:, 3]
        support_pred = torch.sigmoid(edge_pred[:, 4])

        support_target = support_gt * valid_gt
        support_region_gt = (valid_gt > 0.5).float()
        dir_valid_mask = (valid_gt > 0.5) & (dist_gt > self.loss_fn.tau_dir)
        dist_valid_mask = valid_gt > 0.5

        support_cover = 1.0 - self.loss_fn._tversky_loss(
            support_pred,
            support_region_gt,
            alpha=self.loss_fn.support_tversky_alpha,
            beta=self.loss_fn.support_tversky_beta,
        )
        support_error = self.loss_fn._weighted_mean(
            (support_pred - support_target) ** 2,
            valid_gt,
        )

        dir_pred_unit = self.loss_fn._normalize_direction(dir_pred_raw)
        dir_gt_unit = self.loss_fn._normalize_direction(dir_gt)
        dir_cosine_all = torch.sum(dir_pred_unit * dir_gt_unit, dim=1).clamp(-1.0, 1.0)
        dir_cosine = self._masked_mean(dir_cosine_all, dir_valid_mask, edge_pred)
        dist_error = self._masked_mean(
            torch.abs(dist_pred - dist_gt),
            dist_valid_mask,
            edge_pred,
        )
        dist_error_scaled = self._masked_mean(
            torch.abs((dist_pred / self.loss_fn.dist_scale) - (dist_gt / self.loss_fn.dist_scale)),
            dist_valid_mask,
            edge_pred,
        )

        return dict(
            support_cover=support_cover,
            support_error=support_error,
            dir_cosine=dir_cosine,
            dist_error=dist_error,
            dist_error_scaled=dist_error_scaled,
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
            val_loss_edge=loss_dict["loss_edge"],
            val_loss_support=loss_dict["loss_support"],
            val_loss_dir=loss_dict["loss_dir"],
            val_loss_dist=loss_dict["loss_dist"],
            val_loss_support_reg=loss_dict["loss_support_reg"],
            val_loss_support_cover=loss_dict["loss_support_cover"],
            valid_ratio=loss_dict["valid_ratio"],
            support_positive_ratio=loss_dict["support_positive_ratio"],
            dir_valid_ratio=loss_dict["dir_valid_ratio"],
            dist_gt_valid_mean=loss_dict["dist_gt_valid_mean"],
            support_cover=edge_metrics["support_cover"],
            support_error=edge_metrics["support_error"],
            dir_cosine=edge_metrics["dir_cosine"],
            dist_error=edge_metrics["dist_error"],
            dist_error_scaled=edge_metrics["dist_error_scaled"],
        )
