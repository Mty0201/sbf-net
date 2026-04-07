"""Semantic segmentation + support head + serial boundary offset derivation.

Extends SharedBackboneSemanticSupportModel with module g that derives
boundary offset from semantic logits. The backbone only learns semantic
features — the geometric field is derived serially from semantic output.

See docs/canonical/part2_serial_derivation_discussion.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import pointcept.models  # noqa: F401  # ensure Pointcept model registry is populated
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point

from .heads import (
    BoundaryOffsetModule,
    ResidualFeatureAdapter,
    SemanticHead,
    SupportHead,
)


ADAPTER_TYPES = {
    "ResidualFeatureAdapter": ResidualFeatureAdapter,
}


@MODELS.register_module()
class SerialDerivationModel(nn.Module):
    """Semantic segmentation with support head and serial boundary offset derivation.

    Architecture:
        backbone → semantic_head → seg_logits
        backbone → support_head → support_pred  (CR-C route, separate branch)
        g(softmax(seg_logits), coord) → offset_pred  (serial derivation)

    The offset module g takes soft semantic predictions (not backbone features)
    and derives the geometric boundary field. Gradients from offset supervision
    flow through g back to backbone via seg_logits, reinforcing boundary-region
    semantic quality without gradient competition.
    """

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        offset_k=16,
        offset_hidden_dim=64,
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

        # Serial derivation module g
        self.offset_module = BoundaryOffsetModule(
            num_classes=num_classes,
            k=offset_k,
            hidden_dim=offset_hidden_dim,
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
        coord = input_dict["coord"]  # (N, 3)
        batch_offset = input_dict["offset"]  # (B,) cumulative counts

        backbone_output = self.backbone(Point(input_dict))
        point, feat = self._extract_feat(backbone_output)

        semantic_feat = self.semantic_adapter(feat)
        boundary_feat = self.boundary_adapter(feat)
        seg_logits = self.semantic_head(semantic_feat)
        support_pred = self.support_head(boundary_feat)

        # Serial derivation: offset from semantic logits (NOT backbone features)
        offset_pred = self.offset_module(seg_logits, coord, batch_offset)

        output = dict(
            seg_logits=seg_logits,
            support_pred=support_pred,
            offset_pred=offset_pred,
        )
        if return_point:
            output["point"] = point
        return output
