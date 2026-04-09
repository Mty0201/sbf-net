"""Serial derivation loss: semantic + soft boundary BCE + tolerant offset.

Three-term loss for CR-D. Combines:
  - Global semantic (CE + Lovasz)
  - Soft boundary BCE (CR-G style): support branch boundary detection vs GT.
  - Tolerant offset loss from module g:
    a) Cosine direction loss on valid points only (~2%): coarse directional
       alignment, tolerant of magnitude errors.
    b) Local consistency: within each patch, same-side neighbors (same GT
       class) should predict similar offset directions.

See docs/canonical/part2_serial_derivation_discussion.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.losses.lovasz import LovaszLoss


class SerialDerivationLoss(nn.Module):
    """Three-term loss: semantic + soft boundary BCE + tolerant offset."""

    def __init__(
        self,
        aux_weight: float = 0.3,
        offset_weight: float = 1.0,
        consistency_weight: float = 0.5,
        patch_size: int = 48,
    ):
        super().__init__()
        self.aux_weight = float(aux_weight)
        self.offset_weight = float(offset_weight)
        self.consistency_weight = float(consistency_weight)
        self.patch_size = patch_size

        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_loss = LovaszLoss(
            mode="multiclass", ignore_index=-1, loss_weight=1.0
        )

    def _cosine_direction_loss(self, pred, gt, valid_mask):
        """Cosine direction loss on valid points only. Returns 0 if no valid points."""
        if not valid_mask.any():
            return pred.sum() * 0.0

        p = pred[valid_mask]  # (M, 3)
        g = gt[valid_mask]    # (M, 3)

        # Normalize to unit vectors
        p_norm = F.normalize(p, dim=-1, eps=1e-6)
        g_norm = F.normalize(g, dim=-1, eps=1e-6)

        # 1 - cos(pred, gt), range [0, 2]
        cos_sim = (p_norm * g_norm).sum(dim=-1)  # (M,)
        return (1.0 - cos_sim).mean()

    def _local_consistency_loss(self, offset_pred, segment):
        """Same-side neighbors within patches should have similar offset directions."""
        N = offset_pred.shape[0]
        K = self.patch_size

        if N < K:
            return offset_pred.sum() * 0.0

        # Truncate to multiple of K (drop tail points)
        n_patches = N // K
        n_used = n_patches * K

        pred_patches = offset_pred[:n_used].reshape(n_patches, K, 3)    # (P, K, 3)
        seg_patches = segment[:n_used].reshape(n_patches, K)            # (P, K)

        # Normalize directions
        pred_norm = F.normalize(pred_patches, dim=-1, eps=1e-6)  # (P, K, 3)

        # Pairwise cosine similarity within each patch: (P, K, K)
        cos_matrix = torch.bmm(pred_norm, pred_norm.transpose(1, 2))

        # Same-side mask: same GT class label
        same_side = seg_patches.unsqueeze(2) == seg_patches.unsqueeze(1)  # (P, K, K)

        # Exclude self-pairs and ignore_index (-1)
        valid_seg = (seg_patches >= 0).unsqueeze(2) & (seg_patches >= 0).unsqueeze(1)
        diag_mask = ~torch.eye(K, dtype=torch.bool, device=offset_pred.device).unsqueeze(0)
        pair_mask = same_side & valid_seg & diag_mask  # (P, K, K)

        if not pair_mask.any():
            return offset_pred.sum() * 0.0

        # Same-side pairs should have cosine similarity close to 1
        loss = (1.0 - cos_matrix[pair_mask]).mean()
        return loss

    def forward(
        self,
        seg_logits: torch.Tensor,
        support_pred: torch.Tensor,
        offset_pred: torch.Tensor,
        segment: torch.Tensor,
        edge: torch.Tensor,
        **_extra,
    ) -> dict[str, torch.Tensor]:
        segment = segment.reshape(-1).long()
        edge = edge.float()

        # Ground-truth extraction
        offset_gt = edge[:, 0:3]  # (N, 3) displacement to boundary
        support_gt = edge[:, 3].clamp(0.0, 1.0)
        valid_gt = edge[:, 4].clamp(0.0, 1.0)

        # === Term 1: Global semantic (CE + Lovasz) ===
        loss_ce = self.semantic_loss(seg_logits, segment)
        loss_lovasz = self.lovasz_loss(seg_logits, segment)
        loss_semantic = loss_ce + loss_lovasz

        # === Term 2: Soft boundary BCE (CR-G style, pos_weight) ===
        support_logit = support_pred[:, 0]
        with torch.no_grad():
            pos_mask = support_gt > 0.5
            neg_ratio = (~pos_mask).float().mean()
            pos_ratio = pos_mask.float().mean().clamp_min(1e-6)
            pw = (neg_ratio / pos_ratio).sqrt()
        loss_aux = F.binary_cross_entropy_with_logits(
            support_logit, support_gt, pos_weight=pw, reduction="mean"
        )

        # === Term 3a: Cosine direction loss (valid points only) ===
        valid_mask = valid_gt > 0.5
        loss_direction = self._cosine_direction_loss(offset_pred, offset_gt, valid_mask)

        # === Term 3b: Local consistency (same-side patch neighbors) ===
        loss_local = self._local_consistency_loss(offset_pred, segment)

        # === Total ===
        loss_aux_weighted = self.aux_weight * loss_aux
        loss_offset_total = self.offset_weight * loss_direction + self.consistency_weight * loss_local
        total = loss_semantic + loss_aux_weighted + loss_offset_total

        # Monitoring metrics (detached)
        with torch.no_grad():
            support_prob = torch.sigmoid(support_logit)
            high_support = support_gt > 0.5
            high_support_sum = high_support.float().sum().clamp_min(1e-6)
            aux_prob_mean = support_prob.mean()
            aux_prob_boundary_mean = (
                (support_prob * high_support.float()).sum() / high_support_sum
            )
            if valid_mask.any():
                p_norm = F.normalize(offset_pred[valid_mask], dim=-1, eps=1e-6)
                g_norm = F.normalize(offset_gt[valid_mask], dim=-1, eps=1e-6)
                cos_mean = (p_norm * g_norm).sum(dim=-1).mean()
            else:
                cos_mean = torch.tensor(0.0, device=seg_logits.device)

        return dict(
            loss=total,
            loss_semantic=loss_semantic,
            loss_ce=loss_ce,
            loss_lovasz=loss_lovasz,
            loss_aux=loss_aux,
            loss_aux_weighted=loss_aux_weighted,
            loss_direction=loss_direction,
            loss_local_consistency=loss_local,
            loss_offset_total=loss_offset_total,
            pos_weight=pw,
            valid_ratio=valid_gt.mean(),
            support_positive_ratio=(support_gt > 1e-3).float().mean(),
            aux_prob_mean=aux_prob_mean,
            aux_prob_boundary_mean=aux_prob_boundary_mean,
            offset_cos_mean=cos_mean,
        )
