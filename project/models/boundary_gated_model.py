"""Boundary-gated semantic segmentation + support auxiliary model.

Extends SharedBackboneSemanticSupportModel with BoundaryGatingModule (g v3):
boundary adapter features produce a per-point gate that modulates semantic
features before the semantic head. This is the boundary→semantic direction
(BFANet-inspired), replacing the failed semantic→boundary direction (g v1/v2).

Architecture:
    backbone → feat
    feat → boundary_adapter → boundary_feat → support_head → support_pred
                                    ↓
                              g(boundary_feat) → gate
                                    ↓
    feat → semantic_adapter → semantic_feat * (1 + gate) → semantic_head → seg_logits

No extra loss for g — semantic loss gradients flow through the gate into
boundary_feat and backbone, training g to enhance semantic features at
boundary regions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import pointcept.models  # noqa: F401
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point

from .heads import (
    BoundaryGatingModule,
    ResidualFeatureAdapter,
    SemanticHead,
    SupportHead,
)


ADAPTER_TYPES = {
    "ResidualFeatureAdapter": ResidualFeatureAdapter,
}


@MODELS.register_module()
class BoundaryGatedSemanticModel(nn.Module):
    """Semantic segmentation with boundary-gated feature enhancement.

    Same dual-head structure as SharedBackboneSemanticSupportModel, plus
    BoundaryGatingModule that uses boundary_feat to modulate semantic_feat.
    """

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        gate_patch_size=48,
        gate_num_heads=4,
        enable_flash=True,
        semantic_adapter_cfg: dict | None = None,
        boundary_adapter_cfg: dict | None = None,
        backbone=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.semantic_adapter = self._build_adapter(
            backbone_out_channels, semantic_adapter_cfg
        )
        self.boundary_adapter = self._build_adapter(
            backbone_out_channels, boundary_adapter_cfg
        )
        self.semantic_head = SemanticHead(backbone_out_channels, num_classes)
        self.support_head = SupportHead(backbone_out_channels)
        self.boundary_gate = BoundaryGatingModule(
            backbone_out_channels,
            patch_size=gate_patch_size,
            num_heads=gate_num_heads,
            enable_flash=enable_flash,
        )

    @staticmethod
    def _build_adapter(in_channels, adapter_cfg: dict | None):
        if adapter_cfg is None:
            return nn.Identity()
        adapter_cfg = dict(adapter_cfg)
        adapter_type = adapter_cfg.pop("type", "ResidualFeatureAdapter")
        if adapter_type not in ADAPTER_TYPES:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")
        adapter_cls = ADAPTER_TYPES[adapter_type]
        return adapter_cls(in_channels, **adapter_cfg)

    @staticmethod
    def _extract_feat(backbone_output):
        point = backbone_output
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        return point, feat

    def forward(self, input_dict, return_point=False):
        backbone_output = self.backbone(Point(input_dict))
        point, feat = self._extract_feat(backbone_output)

        boundary_feat = self.boundary_adapter(feat)
        support_pred = self.support_head(boundary_feat)

        # Boundary→semantic gating (with patch self-attention)
        gate = self.boundary_gate(boundary_feat, point)
        semantic_feat = self.semantic_adapter(feat)
        semantic_feat = semantic_feat * (1.0 + gate)

        seg_logits = self.semantic_head(semantic_feat)

        output = dict(
            seg_logits=seg_logits,
            support_pred=support_pred,
        )
        if return_point:
            output["point"] = point
        return output
