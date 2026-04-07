"""Single-branch semantic segmentation + serial boundary offset derivation.

Unlike SerialDerivationModel, this model has NO support head and no
boundary adapter. The only auxiliary signal is the offset field derived
serially from semantic logits via module g.

CR-E baseline: tests whether pure serial derivation suffices without
the support/proximity BCE regularizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import pointcept.models  # noqa: F401
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point

from .heads import BoundaryOffsetModule, SemanticHead


@MODELS.register_module()
class SerialDerivationOnlyModel(nn.Module):
    """Semantic segmentation with serial boundary offset derivation only.

    Architecture:
        backbone → semantic_head → seg_logits
        g(softmax(seg_logits), coord) → offset_pred

    No support head, no boundary adapter. Single-branch backbone.
    """

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        offset_k=16,
        offset_hidden_dim=64,
        backbone=None,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.semantic_head = SemanticHead(backbone_out_channels, num_classes)
        self.offset_module = BoundaryOffsetModule(
            num_classes=num_classes,
            k=offset_k,
            hidden_dim=offset_hidden_dim,
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
        coord = input_dict["coord"]
        batch_offset = input_dict["offset"]

        backbone_output = self.backbone(Point(input_dict))
        point, feat = self._extract_feat(backbone_output)

        seg_logits = self.semantic_head(feat)
        offset_pred = self.offset_module(seg_logits, coord, batch_offset)

        output = dict(
            seg_logits=seg_logits,
            offset_pred=offset_pred,
        )
        if return_point:
            output["point"] = point
        return output
