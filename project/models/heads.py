"""Minimal task heads for semantic segmentation and boundary field supervision."""

import torch
import torch.nn as nn
from torch_cluster import knn


class ResidualFeatureAdapter(nn.Module):
    """Lightweight identity-friendly adapter for task-specific feature branches."""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        residual_scale=1.0,
        zero_init_last=True,
    ):
        super().__init__()
        hidden_channels = int(hidden_channels or in_channels)
        self.residual_scale = float(residual_scale)
        self.adapter = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )
        if zero_init_last:
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, feat):
        return feat + self.residual_scale * self.adapter(feat)


class SemanticHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.proj = (
            nn.Linear(in_channels, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, feat):
        return self.proj(feat)


class SupportHead(nn.Module):
    """Predict boundary support from shared features — no direction or distance."""

    def __init__(self, in_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
        )
        self.support_head = nn.Linear(in_channels, 1)

    def forward(self, feat):
        feat = self.stem(feat)
        return self.support_head(feat)


class EdgeHead(nn.Module):
    """Predict boundary direction, distance, and support from shared features."""

    def __init__(self, in_channels, out_channels=5):
        super().__init__()
        if out_channels != 5:
            raise ValueError(
                "EdgeHead expects 5 output channels arranged as dir(3) + dist(1) + support(1)."
            )
        self.stem = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
        )
        self.support_head = nn.Linear(in_channels, 1)
        self.dist_head = nn.Linear(in_channels, 1)
        self.dir_head = nn.Linear(in_channels, 3)

    def forward(self, feat):
        feat = self.stem(feat)
        return dict(
            support_pred=self.support_head(feat),
            dist_pred=self.dist_head(feat),
            dir_pred=self.dir_head(feat),
        )


class SupportConditionedEdgeHead(nn.Module):
    """Stage-2 edge head with support-first conditioning for direction."""

    def __init__(self, in_channels, out_channels=5):
        super().__init__()
        if out_channels != 5:
            raise ValueError(
                "SupportConditionedEdgeHead expects 5 output channels arranged as dir(3) + dist(1) + support(1)."
            )
        self.support_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
        )
        self.support_head = nn.Linear(in_channels, 1)
        self.direction_private_tower = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
        )
        self.dir_head = nn.Linear(in_channels, 3)
        self.dist_head = nn.Linear(in_channels, 1)

    def forward(self, feat):
        support_feat = self.support_mlp(feat)
        dir_input = torch.cat([feat, support_feat], dim=1)
        dir_feat = self.direction_private_tower(dir_input)
        return dict(
            support_pred=self.support_head(support_feat),
            dist_pred=self.dist_head(support_feat),
            dir_pred=self.dir_head(dir_feat),
        )


class BoundaryOffsetModule(nn.Module):
    """Learnable module g: derives boundary offset field from semantic logits.

    Takes softmax(seg_logits) and coord, performs KNN local aggregation to
    capture local semantic gradients, and predicts a 3D offset vector per point.
    offset = displacement from current point to nearest boundary.
    """

    def __init__(self, num_classes, k=16, hidden_dim=64):
        super().__init__()
        self.k = k
        # Edge feature: [feat_i, feat_j - feat_i, coord_j - coord_i]
        # dim = (num_classes + 3) + (num_classes + 3) + 3 = 2 * num_classes + 9
        edge_feat_dim = 2 * num_classes + 9
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, seg_logits, coord, offset):
        """
        Args:
            seg_logits: (N, num_classes) raw logits from semantic head
            coord: (N, 3) point coordinates
            offset: (B,) cumulative point counts per batch element

        Returns:
            offset_pred: (N, 3) predicted displacement to nearest boundary
        """
        # Soft semantic probabilities (differentiable)
        sem_prob = seg_logits.softmax(dim=-1)  # (N, C)
        feat = torch.cat([sem_prob, coord], dim=-1)  # (N, C+3)

        # Build batch vector from offset for torch_cluster
        N = coord.shape[0]
        batch_vec = torch.zeros(N, dtype=torch.long, device=coord.device)
        for i in range(offset.shape[0]):
            start = 0 if i == 0 else int(offset[i - 1])
            batch_vec[start : int(offset[i])] = i

        # KNN on coord (spatial neighbors only)
        # Returns (2, N*K): row 0 = source (neighbor), row 1 = target (center)
        edge_index = knn(coord, coord, self.k, batch_vec, batch_vec)
        row, col = edge_index[0], edge_index[1]  # row=neighbor, col=center

        # Edge features: [feat_center, feat_neighbor - feat_center, coord_diff]
        feat_center = feat[col]  # (N*K, C+3)
        feat_neighbor = feat[row]  # (N*K, C+3)
        coord_diff = coord[row] - coord[col]  # (N*K, 3)

        edge_feat = torch.cat(
            [feat_center, feat_neighbor - feat_center, coord_diff], dim=-1
        )  # (N*K, 2*(C+3)+3)

        # Shared MLP on edge features
        edge_feat = self.edge_mlp(edge_feat)  # (N*K, hidden_dim)

        # Max-pool over K neighbors per point
        pooled = torch.full(
            (N, edge_feat.shape[1]), float("-inf"), device=edge_feat.device
        )
        pooled.scatter_reduce_(
            0, col.unsqueeze(-1).expand_as(edge_feat), edge_feat, reduce="amax"
        )
        # Replace -inf (points with no neighbors, shouldn't happen) with 0
        pooled = pooled.clamp_min(0.0)

        # Offset prediction
        return self.offset_head(pooled)  # (N, 3)
