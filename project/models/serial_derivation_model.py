"""Semantic segmentation + support head + boundary offset module g.

Extends SharedBackboneSemanticSupportModel with module g that derives a
3D boundary offset field from semantic logits. Trained with tolerant
cosine direction loss + local consistency, so gradients nudge backbone
to produce better semantic predictions at boundary regions.

See docs/canonical/part2_serial_derivation_discussion.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import pointcept.models  # noqa: F401  # ensure Pointcept model registry is populated
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point

from .heads import (
    BoundaryConsistencyModule,
    ResidualFeatureAdapter,
    SemanticHead,
    SupportHead,
)


ADAPTER_TYPES = {
    "ResidualFeatureAdapter": ResidualFeatureAdapter,
}


@MODELS.register_module()
class SerialDerivationModel(nn.Module):
    """Semantic segmentation with support head and boundary offset module g.

    Architecture:
        backbone → semantic_head → seg_logits
        backbone → support_head → support_pred  (BCE vs GT, boundary detection)
        g(softmax(seg_logits), coord) → offset_pred  (tolerant offset derivation)

    Module g derives a 3D boundary offset field from semantic logits.
    Trained with cosine direction loss (tolerant, valid points only) +
    local consistency (same-side patch neighbors have similar directions).
    Gradients flow through seg_logits back to backbone.
    """

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        offset_channels=64,
        offset_patch_size=48,
        offset_num_heads=4,
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

        # Boundary consistency module g
        self.consistency_module = BoundaryConsistencyModule(
            num_classes=num_classes,
            channels=offset_channels,
            patch_size=offset_patch_size,
            num_heads=offset_num_heads,
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
        coord = input_dict["coord"]  # (N, 3)

        backbone_output = self.backbone(Point(input_dict))
        point, feat = self._extract_feat(backbone_output)

        semantic_feat = self.semantic_adapter(feat)
        boundary_feat = self.boundary_adapter(feat)
        seg_logits = self.semantic_head(semantic_feat)
        support_pred = self.support_head(boundary_feat)

        # Boundary offset: 3D field from semantic logits (NOT backbone features)
        # point carries serialized_order/inverse/grid_coord from backbone
        offset_pred = self.consistency_module(seg_logits, coord, point)

        output = dict(
            seg_logits=seg_logits,
            support_pred=support_pred,
            offset_pred=offset_pred,
        )
        if return_point:
            output["point"] = point
        return output
