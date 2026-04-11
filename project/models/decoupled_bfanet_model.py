"""CR-SD segmentor: PTv3 backbone + SubtractiveDecoupling + dual heads."""

import torch
import torch.nn as nn

import pointcept.models  # noqa: F401 — triggers registry population
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point

from .subtractive_decoupling import SubtractiveDecoupling


@MODELS.register_module()
class DecoupledBFANetSegmentorV1(nn.Module):
    """PT-v3m1 backbone + SubtractiveDecoupling + seg/marg heads.

    Returns the two logit tensors; loss is computed externally by
    ``CRSDLoss`` so the trainer's loss dispatch stays the single source of
    truth for backprop. ``boundary_mask`` is already voxel-aligned by
    ``InjectIndexValidKeys`` and is consumed by the loss directly from the
    batch.
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

        with torch.no_grad():
            alpha = self.decoupling.alpha.detach()
            w = self.decoupling.proj_to_semantic.weight.detach()
            alpha_mean = alpha.mean()
            alpha_std = alpha.std()
            alpha_abs_max = alpha.abs().max()
            w_fro = w.norm(p="fro")

        return dict(
            seg_logits=seg_logits,
            marg_logits=marg_logits,
            alpha_mean=alpha_mean,
            alpha_std=alpha_std,
            alpha_abs_max=alpha_abs_max,
            w_fro=w_fro,
        )
