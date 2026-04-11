"""CR-SD segmentor: PTv3 backbone + SubtractiveDecoupling + dual heads."""

import torch
import torch.nn as nn

import pointcept.models  # noqa: F401 — triggers registry population
from pointcept.models.builder import MODELS, build_model
from pointcept.models.losses import LovaszLoss
from pointcept.models.utils.structure import Point

from .subtractive_decoupling import SubtractiveDecoupling


@MODELS.register_module()
class DecoupledBFANetSegmentorV1(nn.Module):
    """PT-v3m1 backbone + SubtractiveDecoupling + seg/marg heads.

    Loss is computed inside the segmentor and returned under ``loss``:
      - semantics: CE + multiclass Lovasz on seg_logits vs segment
      - margin:    BCE + binary Lovasz on marg_logits vs boundary_mask

    Only the segment-valid mask (``segment != -1``) is applied before loss
    reduction. ``boundary_mask`` is already voxel-aligned by
    ``InjectIndexValidKeys``.
    """

    def __init__(self, num_classes=8, backbone_out_channels=64, backbone=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.decoupling = SubtractiveDecoupling(
            channels=backbone_out_channels,
            hidden=256,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
        )
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.marg_head = nn.Linear(backbone_out_channels, 1)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.lovasz_multi = LovaszLoss(
            mode="multiclass", loss_weight=1.0, ignore_index=-1
        )
        self.margin_bce = nn.BCEWithLogitsLoss(reduction="none")
        self.margin_lovasz = LovaszLoss(mode="binary", loss_weight=1.0)

    @staticmethod
    def _extract_feat(backbone_output):
        point = backbone_output
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat(
                    [parent.feat, point.feat[inverse]], dim=-1
                )
                point = parent
            feat = point.feat
        else:
            feat = point
        return point, feat

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        point, _ = self._extract_feat(point)

        point = self.decoupling(point)

        seg_feat = point.point_seg.feat
        marg_feat = point.feat

        seg_logits = self.seg_head(seg_feat)
        marg_logits = self.marg_head(marg_feat)

        if "segment" in input_dict:
            segment = input_dict["segment"]
            valid_mask = segment != -1
            boundary_mask = input_dict["boundary_mask"].float().view(-1)
            boundary_valid = boundary_mask[valid_mask]
            marg_valid = marg_logits.view(-1)[valid_mask]

            loss = self.ce_loss(seg_logits, segment)
            loss = loss + self.lovasz_multi(seg_logits, segment)
            loss = loss + self.margin_bce(marg_valid, boundary_valid).mean()
            loss = loss + self.margin_lovasz(marg_valid, boundary_valid)

            if self.training:
                return dict(loss=loss)
            return dict(loss=loss, seg_logits=seg_logits)

        return dict(seg_logits=seg_logits)
