"""CR-SDE segmentor: PTv3 backbone + SubtractiveDecoupling + CrossStreamFusion + dual heads.

Same skeleton as ``DecoupledBFANetSegmentorV1`` (CR-SD). After subtractive
decoupling produces two divergent streams, a BFANet-style cross-stream
fusion attention (g v4) lets them exchange information before the logit
heads. This prevents late-stage gradient conflict between the two branches
by re-aligning their representations through a shared fused query.

``DecoupledBFANetSegmentorV1`` is unchanged — CR-SDE is a physically separate
segmentor so the CR-SD baseline stays clean.
"""

import torch
import torch.nn as nn

import pointcept.models  # noqa: F401 — triggers registry population
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point

from .gv4 import CrossStreamFusionAttention
from .subtractive_decoupling import SubtractiveDecoupling


@MODELS.register_module()
class DecoupledBFANetSegmentorGRef(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        backbone_out_channels: int = 64,
        backbone: dict | None = None,
        fusion_num_heads: int = 4,
        fusion_patch_size: int = 1024,
    ) -> None:
        super().__init__()
        self.backbone = build_model(backbone)
        self.decoupling = SubtractiveDecoupling(
            channels=backbone_out_channels,
            hidden=256,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
        )
        self.fusion = CrossStreamFusionAttention(
            channels=backbone_out_channels,
            num_heads=fusion_num_heads,
            patch_size=fusion_patch_size,
            enable_flash=True,
        )
        self.seg_head_v1 = nn.Linear(backbone_out_channels, num_classes)
        self.marg_head_v1 = nn.Linear(backbone_out_channels, 1)
        self.seg_head_v2 = nn.Linear(backbone_out_channels, num_classes)
        self.marg_head_v2 = nn.Linear(backbone_out_channels, 1)

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

        seg_logits_v1 = self.seg_head_v1(point_seg.feat)
        marg_logits_v1 = self.marg_head_v1(point_marg.feat)

        seg_feat_v2, marg_feat_v2 = self.fusion(
            point_seg.feat, point_marg.feat, point_seg,
        )

        seg_logits = self.seg_head_v2(seg_feat_v2)
        marg_logits = self.marg_head_v2(marg_feat_v2)

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
            seg_logits_v1=seg_logits_v1,
            marg_logits_v1=marg_logits_v1,
            alpha_mean=alpha_mean,
            alpha_std=alpha_std,
            alpha_abs_max=alpha_abs_max,
            w_fro=w_fro,
        )
