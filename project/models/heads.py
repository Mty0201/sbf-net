"""Minimal task heads for semantic segmentation and boundary field supervision."""

import torch
import torch.nn as nn

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.utils.misc import offset2bincount


class ResidualFeatureAdapter(nn.Module):
    """Lightweight identity-friendly adapter for task-specific feature branches."""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        residual_scale=1.0,
        zero_init_last=True,
    ):
        super().__init__()
        hidden_channels = int(hidden_channels or in_channels)
        self.residual_scale = float(residual_scale)
        self.adapter = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
        )
        if zero_init_last:
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, feat):
        return feat + self.residual_scale * self.adapter(feat)


class SemanticHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.proj = (
            nn.Linear(in_channels, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, feat):
        return self.proj(feat)


class SupportHead(nn.Module):
    """Predict boundary support from shared features — no direction or distance."""

    def __init__(self, in_channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
        )
        self.support_head = nn.Linear(in_channels, 1)

    def forward(self, feat):
        feat = self.stem(feat)
        return self.support_head(feat)


class EdgeHead(nn.Module):
    """Predict boundary direction, distance, and support from shared features."""

    def __init__(self, in_channels, out_channels=5):
        super().__init__()
        if out_channels != 5:
            raise ValueError(
                "EdgeHead expects 5 output channels arranged as dir(3) + dist(1) + support(1)."
            )
        self.stem = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
        )
        self.support_head = nn.Linear(in_channels, 1)
        self.dist_head = nn.Linear(in_channels, 1)
        self.dir_head = nn.Linear(in_channels, 3)

    def forward(self, feat):
        feat = self.stem(feat)
        return dict(
            support_pred=self.support_head(feat),
            dist_pred=self.dist_head(feat),
            dir_pred=self.dir_head(feat),
        )


class SupportConditionedEdgeHead(nn.Module):
    """Stage-2 edge head with support-first conditioning for direction."""

    def __init__(self, in_channels, out_channels=5):
        super().__init__()
        if out_channels != 5:
            raise ValueError(
                "SupportConditionedEdgeHead expects 5 output channels arranged as dir(3) + dist(1) + support(1)."
            )
        self.support_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
        )
        self.support_head = nn.Linear(in_channels, 1)
        self.direction_private_tower = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
        )
        self.dir_head = nn.Linear(in_channels, 3)
        self.dist_head = nn.Linear(in_channels, 1)

    def forward(self, feat):
        support_feat = self.support_mlp(feat)
        dir_input = torch.cat([feat, support_feat], dim=1)
        dir_feat = self.direction_private_tower(dir_input)
        return dict(
            support_pred=self.support_head(support_feat),
            dist_pred=self.dist_head(support_feat),
            dir_pred=self.dir_head(dir_feat),
        )


class BoundaryGatingModule(nn.Module):
    """Module g (v3): boundary→semantic gating with patch self-attention.

    Uses boundary adapter features + PTv3-style serialized patch attention
    to produce a per-point residual gate that modulates semantic features.
    The attention lets each point see its spatial neighbors' boundary features,
    enabling the gate to reason about local semantic transition patterns
    (e.g. "my neighbors have different boundary characteristics → I'm near
    an edge → boost semantic features here").

    No dedicated loss — trained entirely by semantic loss gradients flowing
    back through the gate into boundary_feat and then into the backbone.

    Architecture:
        boundary_feat (N, C) → patch self-attention (PTv3 serialization)
        → FFN → output_head → gate (N, C) → sigmoid
        semantic_feat = semantic_feat * (1 + gate)   [residual gating]
    """

    def __init__(self, in_channels, patch_size=48, num_heads=4, enable_flash=True):
        super().__init__()
        C = in_channels
        assert C % num_heads == 0
        self.channels = C
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.scale = (C // num_heads) ** -0.5
        self.enable_flash = enable_flash

        # Self-attention on boundary_feat
        self.qkv = nn.Linear(C, C * 3)
        self.attn_proj = nn.Linear(C, C)

        if enable_flash:
            assert flash_attn is not None, "flash_attn required when enable_flash=True"
            self.rpe = None
        else:
            pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
            rpe_num = 2 * pos_bnd + 1
            self.rpe_pos_bnd = pos_bnd
            self.rpe_num = rpe_num
            self.rpe = nn.Parameter(torch.zeros(3 * rpe_num, num_heads))
            nn.init.trunc_normal_(self.rpe, std=0.02)

        self.norm1 = nn.LayerNorm(C)
        self.norm2 = nn.LayerNorm(C)
        self.ffn = nn.Sequential(
            nn.Linear(C, C * 4),
            nn.GELU(),
            nn.Linear(C * 4, C),
        )

        # Output head: per-channel gate logits
        self.output_head = nn.Sequential(
            nn.Linear(C, C // 2),
            nn.GELU(),
            nn.Linear(C // 2, C),
        )
        # Zero-init last layer so gate starts at sigmoid(0) = 0.5
        # → initial multiplier = 1.5, close to identity
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def _get_padding_and_inverse(self, offset):
        """Pad each batch element to a multiple of patch_size."""
        K = self.patch_size
        bincount = offset2bincount(offset)
        bincount_pad = (
            torch.div(bincount + K - 1, K, rounding_mode="trunc") * K
        )
        mask_pad = bincount > K
        bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad

        _offset = nn.functional.pad(offset, (1, 0))
        _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))

        pad = torch.arange(_offset_pad[-1], device=offset.device)
        unpad = torch.arange(_offset[-1], device=offset.device)
        cu_seqlens = []
        for i in range(len(offset)):
            unpad[_offset[i]: _offset[i + 1]] += _offset_pad[i] - _offset[i]
            if bincount[i] != bincount_pad[i]:
                pad[
                    _offset_pad[i + 1] - K + (bincount[i] % K): _offset_pad[i + 1]
                ] = pad[
                    _offset_pad[i + 1] - 2 * K + (bincount[i] % K): _offset_pad[i + 1] - K
                ]
            pad[_offset_pad[i]: _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
            cu_seqlens.append(
                torch.arange(
                    _offset_pad[i], _offset_pad[i + 1],
                    step=K, dtype=torch.int32, device=offset.device,
                )
            )
        cu_seqlens = nn.functional.pad(
            torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
        )
        return pad, unpad, cu_seqlens

    @torch.no_grad()
    def _get_rel_pos(self, grid_coord_ordered):
        """Compute pairwise RPE indices within each patch (non-flash path)."""
        K = self.patch_size
        gc = grid_coord_ordered.reshape(-1, K, 3)
        rel = gc.unsqueeze(2) - gc.unsqueeze(1)
        idx = (
            rel.clamp(-self.rpe_pos_bnd, self.rpe_pos_bnd)
            + self.rpe_pos_bnd
            + torch.arange(3, device=rel.device) * self.rpe_num
        )
        out = self.rpe.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        return out.permute(0, 3, 1, 2)

    def forward(self, boundary_feat, point):
        """
        Args:
            boundary_feat: (N, C) features from boundary adapter
            point: Point dict from backbone with serialized_order,
                   serialized_inverse, grid_coord, offset

        Returns:
            gate: (N, C) values in (0, 1) for residual gating
        """
        H = self.num_heads
        K = self.patch_size
        C = self.channels

        feat = boundary_feat

        pad, unpad, cu_seqlens = self._get_padding_and_inverse(point.offset)
        order = point.serialized_order[0][pad]
        inverse = unpad[point.serialized_inverse[0]]

        qkv = self.qkv(self.norm1(feat))[order]

        if self.enable_flash:
            attn_out = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(torch.bfloat16).reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=K,
                dropout_p=0.0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            attn_out = attn_out.to(qkv.dtype)
        else:
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H)
                .permute(2, 0, 3, 1, 4)
                .unbind(dim=0)
            )
            attn = (q * self.scale) @ k.transpose(-2, -1)
            grid_coord_ordered = point.grid_coord[order]
            attn = attn + self._get_rel_pos(grid_coord_ordered)
            attn = attn.float().softmax(dim=-1).to(qkv.dtype)
            attn_out = (attn @ v).transpose(1, 2).reshape(-1, C)

        attn_out = self.attn_proj(attn_out[inverse])
        feat = feat + attn_out
        feat = feat + self.ffn(self.norm2(feat))

        return torch.sigmoid(self.output_head(feat))


class BoundaryConsistencyModule(nn.Module):
    """Learnable module g: boundary offset derivation with tolerance.

    From semantic logits, derives a 3D boundary offset field. Unlike the
    original exact-regression approach, this version is trained with:
      - Cosine direction loss on valid points only (~2%): coarse directional
        alignment, tolerant of magnitude errors.
      - Local consistency loss within patches: same-side neighbors (same GT
        class) should predict similar offset directions.

    Gradients flow through seg_logits back to backbone, nudging semantic
    predictions to produce better boundary structure.

    Uses PTv3-style serialized patch self-attention instead of KNN.

    Architecture:
        softmax(seg_logits) + coord → linear projection → patch self-attention
        (with RPE from grid_coord) → MLP → output_head → offset (N, 3)
    """

    def __init__(self, num_classes, channels=64, patch_size=48, num_heads=4,
                 enable_flash=True):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.enable_flash = enable_flash

        # Input projection: softmax probs (num_classes) + coord (3) → channels
        self.input_proj = nn.Linear(num_classes + 3, channels)

        # Self-attention
        self.qkv = nn.Linear(channels, channels * 3)
        self.attn_proj = nn.Linear(channels, channels)

        if enable_flash:
            assert flash_attn is not None, "flash_attn required when enable_flash=True"
            self.rpe = None
        else:
            # RPE when flash attention is off (same design as PTv3)
            pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
            rpe_num = 2 * pos_bnd + 1
            self.rpe_pos_bnd = pos_bnd
            self.rpe_num = rpe_num
            self.rpe = nn.Parameter(torch.zeros(3 * rpe_num, num_heads))
            nn.init.trunc_normal_(self.rpe, std=0.02)

        # FFN (same ratio as PTv3: 4x)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

        # Output head: 3D boundary offset field
        self.output_head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 3),
        )

    def _get_padding_and_inverse(self, offset):
        """Pad each batch element's point count to a multiple of patch_size.

        Mirrors SerializedAttention.get_padding_and_inverse from PTv3.
        """
        K = self.patch_size
        bincount = offset2bincount(offset)
        bincount_pad = (
            torch.div(bincount + K - 1, K, rounding_mode="trunc") * K
        )
        # Only pad when count > patch_size
        mask_pad = bincount > K
        bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad

        _offset = nn.functional.pad(offset, (1, 0))
        _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))

        pad = torch.arange(_offset_pad[-1], device=offset.device)
        unpad = torch.arange(_offset[-1], device=offset.device)
        cu_seqlens = []
        for i in range(len(offset)):
            unpad[_offset[i]: _offset[i + 1]] += _offset_pad[i] - _offset[i]
            if bincount[i] != bincount_pad[i]:
                pad[
                    _offset_pad[i + 1] - K + (bincount[i] % K): _offset_pad[i + 1]
                ] = pad[
                    _offset_pad[i + 1] - 2 * K + (bincount[i] % K): _offset_pad[i + 1] - K
                ]
            pad[_offset_pad[i]: _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
            cu_seqlens.append(
                torch.arange(
                    _offset_pad[i], _offset_pad[i + 1],
                    step=K, dtype=torch.int32, device=offset.device,
                )
            )
        cu_seqlens = nn.functional.pad(
            torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
        )
        return pad, unpad, cu_seqlens

    @torch.no_grad()
    def _get_rel_pos(self, grid_coord_ordered):
        """Compute pairwise RPE indices within each patch (non-flash path)."""
        K = self.patch_size
        gc = grid_coord_ordered.reshape(-1, K, 3)  # (num_patches, K, 3)
        rel = gc.unsqueeze(2) - gc.unsqueeze(1)  # (num_patches, K, K, 3)
        idx = (
            rel.clamp(-self.rpe_pos_bnd, self.rpe_pos_bnd)
            + self.rpe_pos_bnd
            + torch.arange(3, device=rel.device) * self.rpe_num
        )
        out = self.rpe.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)  # (num_patches, K, K, H)
        return out.permute(0, 3, 1, 2)  # (num_patches, H, K, K)

    def forward(self, seg_logits, coord, point):
        """
        Args:
            seg_logits: (N, num_classes) raw logits from semantic head
            coord: (N, 3) point coordinates
            point: Point dict from backbone, must contain serialized_order,
                   serialized_inverse, grid_coord, offset

        Returns:
            offset_pred: (N, 3) predicted boundary offset field
        """
        H = self.num_heads
        K = self.patch_size
        C = self.channels

        # Build input features
        sem_prob = seg_logits.softmax(dim=-1)
        feat = self.input_proj(torch.cat([sem_prob, coord], dim=-1))  # (N, C)

        # Reuse backbone serialization — pick first order (index 0)
        pad, unpad, cu_seqlens = self._get_padding_and_inverse(point.offset)
        order = point.serialized_order[0][pad]
        inverse = unpad[point.serialized_inverse[0]]

        # Reorder features into serialized patches
        qkv = self.qkv(self.norm1(feat))[order]  # (N_pad, 3C)

        if self.enable_flash:
            attn_out = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(torch.bfloat16).reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=K,
                dropout_p=0.0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            attn_out = attn_out.to(qkv.dtype)
        else:
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H)
                .permute(2, 0, 3, 1, 4)
                .unbind(dim=0)
            )
            attn = (q * self.scale) @ k.transpose(-2, -1)
            # RPE from grid_coord
            grid_coord_ordered = point.grid_coord[order]
            attn = attn + self._get_rel_pos(grid_coord_ordered)
            attn = attn.float().softmax(dim=-1).to(qkv.dtype)
            attn_out = (attn @ v).transpose(1, 2).reshape(-1, C)

        # Unsort back to original order
        attn_out = self.attn_proj(attn_out[inverse])
        feat = feat + attn_out

        # FFN
        feat = feat + self.ffn(self.norm2(feat))

        return self.output_head(feat)  # (N, 1)
