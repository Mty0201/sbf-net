"""Cross-stream fusion + dual-supervision semantic/boundary model (CR-M, g v4).

Architecture::

    backbone(PTv3) → (point, feat)
            ├─→ semantic_adapter → sem_feat   (N, C)   → semantic_head   → seg_logits_v1
            └─→ boundary_adapter → bnd_feat   (N, C)   → support_head    → support_pred_v1

    # g v4: cross-stream fusion-query patch attention
    (sem_feat, bnd_feat, point) → CrossStreamFusionAttention
        sem_feat_v2 = sem_feat + out_proj_sem(attn(q_fused, k_sem, v_sem))
        bnd_feat_v2 = bnd_feat + out_proj_bnd(attn(q_fused, k_bnd, v_bnd))

    sem_feat_v2 → semantic_head_v2 → seg_logits_v2
    bnd_feat_v2 → support_head_v2  → support_pred_v2

Key invariants at init:
    - ``out_proj_sem`` / ``out_proj_bnd`` are zero-init in CrossStreamFusionAttention,
      so ``sem_feat_v2 == sem_feat`` and ``bnd_feat_v2 == bnd_feat`` at step 0.
    - ``semantic_head_v2`` / ``support_head_v2`` are cloned from the v1 heads
      via ``load_state_dict`` right after construction, so their outputs match.
    - Consequently ``seg_logits_v2 == seg_logits_v1`` and
      ``support_pred_v2 == support_pred_v1`` at step 0 — training starts on
      the CR-L baseline before the g v4 projections learn anything non-zero.

Output keys:
    - ``seg_logits`` / ``support_pred`` are aliases of the v1 outputs so the
      existing trainer ``_build_loss_inputs`` branch for
      ``support_pred + edge`` continues to fire unchanged.
    - Explicit ``seg_logits_v1`` / ``support_pred_v1`` / ``seg_logits_v2`` /
      ``support_pred_v2`` are also provided for the dual-supervision loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import pointcept.models  # noqa: F401
from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point

from .gv4 import CrossStreamFusionAttention
from .heads import ResidualFeatureAdapter, SemanticHead, SupportHead


ADAPTER_TYPES = {
    "ResidualFeatureAdapter": ResidualFeatureAdapter,
}


@MODELS.register_module()
class BoundaryGatedSemanticModelV4(nn.Module):
    """CR-M dual-head model with CrossStreamFusionAttention and v1/v2 supervision."""

    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        gate_patch_size: int = 48,
        gate_num_heads: int = 4,
        enable_flash: bool = True,
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

        # v1 heads (= CR-L)
        self.semantic_head = SemanticHead(backbone_out_channels, num_classes)
        self.support_head = SupportHead(backbone_out_channels)

        # Cross-stream fusion attention (g v4)
        self.fusion = CrossStreamFusionAttention(
            channels=backbone_out_channels,
            patch_size=gate_patch_size,
            num_heads=gate_num_heads,
            enable_flash=enable_flash,
        )

        # v2 heads — independent instances, weights cloned from v1 at init.
        # These heads use only nn.Linear (no buffers), so load_state_dict is
        # a safe identity copy; see heads.py::SemanticHead / SupportHead.
        self.semantic_head_v2 = SemanticHead(backbone_out_channels, num_classes)
        self.support_head_v2 = SupportHead(backbone_out_channels)
        self.semantic_head_v2.load_state_dict(self.semantic_head.state_dict())
        self.support_head_v2.load_state_dict(self.support_head.state_dict())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_dict, return_point=False):
        backbone_output = self.backbone(Point(input_dict))
        point, feat = self._extract_feat(backbone_output)

        sem_feat = self.semantic_adapter(feat)
        bnd_feat = self.boundary_adapter(feat)

        # v1 predictions (= CR-L baseline)
        seg_logits_v1 = self.semantic_head(sem_feat)
        support_pred_v1 = self.support_head(bnd_feat)

        # g v4 cross-stream fusion attention
        sem_feat_v2, bnd_feat_v2 = self.fusion(sem_feat, bnd_feat, point)

        # v2 predictions
        seg_logits_v2 = self.semantic_head_v2(sem_feat_v2)
        support_pred_v2 = self.support_head_v2(bnd_feat_v2)

        output = dict(
            # Trainer-compat aliases (pointing at v1)
            seg_logits=seg_logits_v1,
            support_pred=support_pred_v1,
            # Explicit v1 / v2 outputs for the dual-supervision wrapper loss
            seg_logits_v1=seg_logits_v1,
            support_pred_v1=support_pred_v1,
            seg_logits_v2=seg_logits_v2,
            support_pred_v2=support_pred_v2,
        )
        if return_point:
            output["point"] = point
        return output
