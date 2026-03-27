"""Shared-backbone semantic segmentation + boundary field model."""

from __future__ import annotations

import torch
import torch.nn as nn

import pointcept.models  # noqa: F401  # ensure Pointcept model registry is populated
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point

from .heads import EdgeHead, SemanticHead, SupportConditionedEdgeHead


EDGE_HEAD_TYPES = {
    "EdgeHead": EdgeHead,
    "SupportConditionedEdgeHead": SupportConditionedEdgeHead,
}


@MODELS.register_module()
class SharedBackboneSemanticBoundaryModel(nn.Module):
    """Edge output follows edge.npy semantics: dir(3) + dist(1) + support(1)."""

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        edge_out_channels=4,
        edge_head_cfg: dict | None = None,
        backbone=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.semantic_head = SemanticHead(backbone_out_channels, num_classes)
        edge_head_cfg = dict(edge_head_cfg or {})
        edge_head_type = edge_head_cfg.pop("type", "EdgeHead")
        if edge_head_type not in EDGE_HEAD_TYPES:
            raise ValueError(f"Unsupported edge head type: {edge_head_type}")
        edge_head_cls = EDGE_HEAD_TYPES[edge_head_type]
        self.edge_head = edge_head_cls(
            backbone_out_channels,
            edge_out_channels,
            **edge_head_cfg,
        )

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
        dist_pred = edge_output["dist_pred"]
        dir_pred = edge_output["dir_pred"]
        # Keep a compact tensor entrypoint for the unchanged trainer/loss plumbing.
        edge_pred = torch.cat([dir_pred, dist_pred, support_pred], dim=1)

        output = dict(
            seg_logits=seg_logits,
            edge_pred=edge_pred,
            support_pred=support_pred,
            dist_pred=dist_pred,
            dir_pred=dir_pred,
        )
        if return_point:
            output["point"] = point
        return output
