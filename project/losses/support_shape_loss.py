"""Support-first semantic boundary loss with pairwise ordinal shape constraint.

Removes direction and distance as prediction targets.  The only boundary
prediction is the scalar support field.  On top of the existing reg + cover
supervision, an **ordinal shape constraint** enforces that spatially nearby
points with smaller dist-to-edge have higher support predictions, shaping
the support field into a ridge along the true edge.

Channel layout of ``edge_pred`` (5 channels, model unchanged):
  [unused(3), unused(1), support_logit(1)]

GT ``edge.npy`` layout (6 columns, unchanged):
  [dir_x, dir_y, dir_z, dist, support, valid]

Only channel 4 (support) of ``edge_pred`` is used by this loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class SupportShapeLoss(nn.Module):
    """Semantic + support (reg + cover + ordinal shape) loss."""

    SUPPORT_POSITIVE_EPS = 1e-3

    def __init__(
        self,
        support_weight: float = 1.0,
        support_reg_weight: float = 1.0,
        support_cover_weight: float = 0.2,
        support_tversky_alpha: float = 0.3,
        support_tversky_beta: float = 0.7,
        ordinal_weight: float = 0.5,
        ordinal_radius: float = 0.15,
        ordinal_min_dist_diff: float = 0.02,
        ordinal_margin: float = 0.05,
        ordinal_narrowband_min: float = 0.1,
        ordinal_n_samples: int = 512,
        ordinal_max_pairs: int = 2048,
    ):
        super().__init__()
        self.support_weight = float(support_weight)
        self.support_reg_weight = float(support_reg_weight)
        self.support_cover_weight = float(support_cover_weight)
        self.support_tversky_alpha = float(support_tversky_alpha)
        self.support_tversky_beta = float(support_tversky_beta)
        self.ordinal_weight = float(ordinal_weight)
        self.ordinal_radius = float(ordinal_radius)
        self.ordinal_min_dist_diff = float(ordinal_min_dist_diff)
        self.ordinal_margin = float(ordinal_margin)
        self.ordinal_narrowband_min = float(ordinal_narrowband_min)
        self.ordinal_n_samples = int(ordinal_n_samples)
        self.ordinal_max_pairs = int(ordinal_max_pairs)

        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_loss = LovaszLoss(
            mode="multiclass",
            ignore_index=-1,
            loss_weight=1.0,
        )
        self.support_regression_loss = nn.SmoothL1Loss(reduction="none")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_mean(
        value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        weight_sum = weight.sum()
        if weight_sum.item() <= 0:
            return value.sum() * 0.0
        return (value * weight).sum() / (weight_sum + eps)

    @staticmethod
    def _tversky_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        beta: float,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        target = target.float()
        true_positive = torch.sum(pred * target)
        false_positive = torch.sum(pred * (1.0 - target))
        false_negative = torch.sum((1.0 - pred) * target)
        return 1.0 - (true_positive + eps) / (
            true_positive + alpha * false_positive + beta * false_negative + eps
        )

    @staticmethod
    def _safe_mean(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if mask.numel() == 0:
            return reference.sum() * 0.0
        return mask.float().mean()

    # ------------------------------------------------------------------
    # ordinal pair sampling and loss
    # ------------------------------------------------------------------

    def _ordinal_loss_single_scene(
        self,
        support_pred: torch.Tensor,
        coord: torch.Tensor,
        dist_gt: torch.Tensor,
        support_gt: torch.Tensor,
        valid_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Compute ordinal shape loss for one scene.

        Returns (loss_value, n_pairs_used).
        """
        device = support_pred.device
        zero = support_pred.sum() * 0.0

        # 1. Identify narrowband points
        narrowband_mask = (valid_gt > 0.5) & (
            support_gt > self.ordinal_narrowband_min
        )
        nb_idx = narrowband_mask.nonzero(as_tuple=True)[0]

        if nb_idx.numel() < 4:
            return zero, 0

        # 2. Subsample for efficiency
        n = nb_idx.numel()
        n_sub = min(n, self.ordinal_n_samples)
        perm = torch.randperm(n, device=device)[:n_sub]
        sub_idx = nb_idx[perm]

        sub_coord = coord[sub_idx]  # (n_sub, 3)
        sub_dist = dist_gt[sub_idx]  # (n_sub,)

        # 3. Pairwise spatial distances (n_sub x n_sub)
        spatial_dist = torch.cdist(
            sub_coord.unsqueeze(0), sub_coord.unsqueeze(0)
        ).squeeze(0)

        # 4. Pairwise dist-to-edge differences
        # dist_diff[i,j] = sub_dist[i] - sub_dist[j]
        dist_diff = sub_dist.unsqueeze(1) - sub_dist.unsqueeze(0)

        # 5. Valid pair mask: spatially close + meaningful dist difference
        valid_pair = (
            (spatial_dist < self.ordinal_radius)
            & (dist_diff.abs() > self.ordinal_min_dist_diff)
        )
        # Upper triangle only (avoid duplicates and self-pairs)
        valid_pair = valid_pair.triu(diagonal=1)

        pair_ij = valid_pair.nonzero(as_tuple=False)  # (M, 2)
        if pair_ij.shape[0] == 0:
            return zero, 0

        # 6. Subsample pairs if too many
        M = pair_ij.shape[0]
        if M > self.ordinal_max_pairs:
            sel = torch.randperm(M, device=device)[: self.ordinal_max_pairs]
            pair_ij = pair_ij[sel]
            M = self.ordinal_max_pairs

        i_local = pair_ij[:, 0]
        j_local = pair_ij[:, 1]

        dist_i = sub_dist[i_local]
        dist_j = sub_dist[j_local]

        # Determine closer/farther (smaller dist = closer to edge)
        i_closer = dist_i < dist_j  # bool (M,)

        # Map back to original indices
        orig_i = sub_idx[i_local]
        orig_j = sub_idx[j_local]

        closer_orig = torch.where(i_closer, orig_i, orig_j)
        farther_orig = torch.where(i_closer, orig_j, orig_i)

        # 7. Margin ranking loss: support(closer) should exceed support(farther)
        sup_closer = support_pred[closer_orig]
        sup_farther = support_pred[farther_orig]

        # loss = max(0, margin - (sup_closer - sup_farther))
        violation = self.ordinal_margin - (sup_closer - sup_farther)
        pair_loss = F.relu(violation)

        return pair_loss.mean(), M

    def _ordinal_loss(
        self,
        support_pred: torch.Tensor,
        coord: torch.Tensor,
        offset: torch.Tensor,
        dist_gt: torch.Tensor,
        support_gt: torch.Tensor,
        valid_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ordinal loss across all scenes in the batch.

        Returns (loss, n_pairs_ratio).
        """
        zero = support_pred.sum() * 0.0

        # Parse scene boundaries from offset
        offsets = offset.long().cpu().tolist()
        starts = [0] + offsets[:-1]
        ends = offsets

        total_loss = zero
        total_pairs = 0
        n_scenes = 0

        for start, end in zip(starts, ends):
            if end <= start:
                continue
            sl = slice(start, end)
            scene_loss, n_pairs = self._ordinal_loss_single_scene(
                support_pred[sl],
                coord[sl],
                dist_gt[sl],
                support_gt[sl],
                valid_gt[sl],
            )
            if n_pairs > 0:
                total_loss = total_loss + scene_loss
                total_pairs += n_pairs
                n_scenes += 1

        if n_scenes > 0:
            total_loss = total_loss / n_scenes

        # Report avg pairs per scene as a diagnostic
        avg_pairs = torch.tensor(
            total_pairs / max(n_scenes, 1),
            device=support_pred.device,
            dtype=support_pred.dtype,
        )

        return total_loss, avg_pairs

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        seg_logits: torch.Tensor,
        edge_pred: torch.Tensor,
        segment: torch.Tensor,
        edge: torch.Tensor,
        coord: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        edge = edge.float()

        # Parse predictions — only support is used
        support_logit = edge_pred[:, 4]
        support_pred = torch.sigmoid(support_logit)

        # Parse GT
        dist_gt = torch.linalg.norm(edge[:, 0:3], dim=1).clamp_min(0.0)
        support_gt = edge[:, 3].float().clamp(0.0, 1.0)
        valid_gt = edge[:, 4].float().clamp(0.0, 1.0)

        support_target = support_gt * valid_gt
        support_region_gt = (valid_gt > 0.5).float()
        support_positive_mask = support_gt > self.SUPPORT_POSITIVE_EPS

        # --- semantic loss ---
        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz

        # --- support reg loss ---
        loss_support_reg = self._weighted_mean(
            self.support_regression_loss(support_pred, support_target),
            valid_gt,
        )

        # --- support cover loss ---
        loss_support_cover = self._tversky_loss(
            support_pred,
            support_region_gt,
            alpha=self.support_tversky_alpha,
            beta=self.support_tversky_beta,
        )

        loss_support = (
            self.support_reg_weight * loss_support_reg
            + self.support_cover_weight * loss_support_cover
        )

        # --- ordinal shape loss ---
        zero = support_pred.sum() * 0.0
        if coord is not None and offset is not None and self.ordinal_weight > 0:
            loss_ordinal, ordinal_pairs = self._ordinal_loss(
                support_pred, coord, offset, dist_gt, support_gt, valid_gt,
            )
        else:
            loss_ordinal = zero
            ordinal_pairs = zero

        # --- combine ---
        loss_edge = (
            self.support_weight * loss_support
            + self.ordinal_weight * loss_ordinal
        )
        total_loss = loss_semantic + loss_edge

        # --- metrics ---
        valid_ratio = valid_gt.mean()
        support_positive_ratio = self._safe_mean(support_positive_mask, edge_pred)

        return dict(
            loss=total_loss,
            loss_semantic=loss_semantic,
            loss_edge=loss_edge,
            loss_support=loss_support,
            loss_support_reg=loss_support_reg,
            loss_support_cover=loss_support_cover,
            loss_ordinal=loss_ordinal,
            valid_ratio=valid_ratio,
            support_positive_ratio=support_positive_ratio,
            support_cover=1.0 - loss_support_cover,
            ordinal_pairs=ordinal_pairs,
        )
