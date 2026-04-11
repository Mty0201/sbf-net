"""CR-SDE segmentor: PTv3 backbone + SubtractiveDecoupling + GatedSegRefiner + dual heads.

Same skeleton as ``DecoupledBFANetSegmentorV1`` (CR-SD). The only difference is
the ``GatedSegRefiner`` block inserted between the decoupling module and the
segmentation head. The refiner reads the margin stream to build a gate and
writes a zero-initialised residual back onto the seg stream.

``DecoupledBFANetSegmentorV1`` is unchanged — CR-SDE is a physically separate
segmentor so the CR-SD baseline stays clean.
"""

import torch
import torch.nn as nn

import pointcept.models  # noqa: F401 — triggers registry population
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point

from .g_refiner import GatedSegRefiner
from .subtractive_decoupling import SubtractiveDecoupling


@MODELS.register_module()
class DecoupledBFANetSegmentorGRef(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        backbone_out_channels: int = 64,
        backbone: dict | None = None,
        refiner_num_heads: int = 4,
        refiner_patch_size: int = 1024,
    ) -> None:
        super().__init__()
        self.backbone = build_model(backbone)
        self.decoupling = SubtractiveDecoupling(
            channels=backbone_out_channels,
            hidden=256,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
        )
        self.refiner = GatedSegRefiner(
            channels=backbone_out_channels,
            num_heads=refiner_num_heads,
            patch_size=refiner_patch_size,
            enable_flash=True,
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

        point_marg = self.decoupling(point)
        point_seg = point_marg.point_seg

        point_seg, g_diag = self.refiner(point_seg, point_marg)

        seg_logits = self.seg_head(point_seg.feat)
        marg_logits = self.marg_head(point_marg.feat)

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
            **g_diag,
        )
