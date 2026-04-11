"""CR-SD: Subtractive Decoupling module.

A PointModule that takes a decoder feature and produces two residual-anchored
streams: a margin stream (``point_marg``) and a segmentation stream
(``point_seg``). The two streams share a LayerNorm, use independent MLPs, and
are coupled by subtracting a zero-initialised projection of the margin feature
(scaled by a per-channel learnable alpha) from the segmentation input.

This module is self-contained. It is not registered with ``MODELS`` — it is
used internally by ``DecoupledBFANetSegmentorV1``.
"""

import torch
import torch.nn as nn
from addict import Dict as AddictDict

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    MLP,
    Point,
    PointModule,
    PointSequential,
)


class SubtractiveDecoupling(PointModule):
    def __init__(self, channels=64, hidden=256, norm_layer=None, act_layer=None):
        super().__init__()
        self.norm = PointSequential(norm_layer(channels))
        self.mlp_marg = PointSequential(
            MLP(channels, hidden, channels, act_layer, drop=0.0)
        )
        self.mlp_seg = PointSequential(
            MLP(channels, hidden, channels, act_layer, drop=0.0)
        )
        self.proj_to_semantic = nn.Linear(channels, channels, bias=False)
        nn.init.zeros_(self.proj_to_semantic.weight)
        # Per-channel raw scalar; NO sigmoid.
        self.alpha = nn.Parameter(torch.full((channels,), 0.5))

    def forward(self, point):
        # 1. Cache the original decoder feature BEFORE LN for residual anchor.
        shortcut = point.feat

        # 2. Shared LayerNorm.
        point = self.norm(point)
        x_norm = point.feat

        # 3. Build a fresh Point for the margin stream so it owns its own
        #    sparse_conv_feat (independent of the seg stream).
        marg_dict = AddictDict()
        marg_dict.coord = point.coord
        marg_dict.grid_coord = point.grid_coord
        marg_dict.batch = point.batch
        marg_dict.offset = point.offset
        marg_dict.feat = x_norm
        for key in list(point.keys()):
            if key.startswith("serialized_"):
                marg_dict[key] = point[key]
        point_marg = Point(marg_dict)
        point_marg.sparsify()

        # 4. Margin MLP.
        point_marg = self.mlp_marg(point_marg)
        m = point_marg.feat

        # 5. Zero-initialised projection of margin back into semantic space.
        m_projected = self.proj_to_semantic(m)

        # 6. Subtractive input for the segmentation stream.
        seg_input = x_norm - self.alpha * m_projected

        # 7. Reuse the LN-ed point as point_seg. Critically, we must update
        #    sparse_conv_feat via replace_feature — spconv caches features on
        #    sparse_conv_feat, and a naked feat assignment is not seen by the
        #    next conv/MLP layer.
        point_seg = point
        point_seg.feat = seg_input
        point_seg.sparse_conv_feat = point_seg.sparse_conv_feat.replace_feature(
            seg_input
        )

        # 8. Segmentation MLP.
        point_seg = self.mlp_seg(point_seg)
        s = point_seg.feat

        # 9. Residual anchor onto BOTH streams, and keep sparse_conv_feat in sync.
        point_seg.feat = shortcut + s
        point_marg.feat = shortcut + m
        point_seg.sparse_conv_feat = point_seg.sparse_conv_feat.replace_feature(
            point_seg.feat
        )
        point_marg.sparse_conv_feat = point_marg.sparse_conv_feat.replace_feature(
            point_marg.feat
        )

        # 10. Attach seg stream to marg stream for retrieval by the segmentor.
        point_marg.point_seg = point_seg

        # 11. Return the margin point (seg point is reachable via .point_seg).
        return point_marg
