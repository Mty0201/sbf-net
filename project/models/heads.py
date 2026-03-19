"""Minimal task heads for semantic boundary modeling."""

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
    def __init__(self, in_channels, out_channels=5):
        super().__init__()
        self.proj = (
            nn.Linear(in_channels, out_channels)
            if out_channels > 0
            else nn.Identity()
        )

    def forward(self, feat):
        return self.proj(feat)
