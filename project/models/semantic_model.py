"""Shared-backbone semantic-only prediction model."""

import torch
import torch.nn as nn

import pointcept.models  # noqa: F401
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point

from .heads import SemanticHead


@MODELS.register_module()
class SharedBackboneSemanticModel(nn.Module):
    def __init__(self, num_classes, backbone_out_channels, backbone=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.semantic_head = SemanticHead(backbone_out_channels, num_classes)

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
        seg_logits = self.semantic_head(feat)
        output = dict(seg_logits=seg_logits)
        if return_point:
            output["point"] = point
        return output
