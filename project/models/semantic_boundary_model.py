"""Shared-backbone semantic segmentation + boundary support/offset model."""

import torch
import torch.nn as nn

import pointcept.models  # noqa: F401  # ensure Pointcept model registry is populated
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point

from .heads import EdgeHead, SemanticHead


@MODELS.register_module()
class SharedBackboneSemanticBoundaryModel(nn.Module):
    """Edge output follows edge.npy semantics: vec(3) + support(1)."""

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        edge_out_channels=4,
        backbone=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.semantic_head = SemanticHead(backbone_out_channels, num_classes)
        self.edge_head = EdgeHead(backbone_out_channels, edge_out_channels)

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
        edge_output = self.edge_head(feat)
        support_pred = edge_output["support_pred"]
        vec_pred = edge_output["vec_pred"]
        # Keep the compact tensor for the unchanged trainer/loss entrypoints.
        edge_pred = torch.cat([vec_pred, support_pred], dim=1)

        output = dict(
            seg_logits=seg_logits,
            edge_pred=edge_pred,
            support_pred=support_pred,
            vec_pred=vec_pred,
        )
        if return_point:
            output["point"] = point
        return output
