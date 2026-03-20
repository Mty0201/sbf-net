"""Minimal task heads for semantic segmentation and boundary support/offset."""

import torch.nn as nn


class SemanticHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.proj = (
            nn.Linear(in_channels, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, feat):
        return self.proj(feat)


class EdgeHead(nn.Module):
    """Predict boundary support and boundary offset from shared backbone features."""

    def __init__(self, in_channels, out_channels=4):
        super().__init__()
        if out_channels != 4:
            raise ValueError(
                "EdgeHead expects 4 output channels arranged as vec(3) + support(1)."
            )
        self.stem = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
        )
        self.support_head = nn.Linear(in_channels, 1)
        self.vec_head = nn.Linear(in_channels, 3)

    def forward(self, feat):
        feat = self.stem(feat)
        return dict(
            support_pred=self.support_head(feat),
            vec_pred=self.vec_head(feat),
        )
