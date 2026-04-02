"""Minimal task heads for semantic segmentation and boundary field supervision."""

import torch
import torch.nn as nn


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
