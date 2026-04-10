"""Module g (v4): cross-stream fusion-query patch attention (CR-M).

BFANet-inspired cross-stream attention that respects the author's original
intent: a fused query built from both semantic and boundary streams attends
to the keys/values of each stream separately. Unlike the BFANet "ES
Attention" reference implementation, which degenerates because its
K dimension is 1 (softmax over a single element becomes the identity and
the fusion query does nothing), CR-M uses ``K = patch_size`` via PTv3-style
serialized patches so the softmax is non-degenerate and the fusion query
actually steers the attention weights.

Pipeline per point::

    q_sem, k_sem, v_sem = qkv_sem(sem_feat)
    q_bnd, k_bnd, v_bnd = qkv_bnd(bnd_feat)
    q_fused              = fusion_q(concat(q_sem, q_bnd))      # BFANet fusion query
    sem_attn             = attn(q_fused, k_sem, v_sem)         # patch self-attn on sem KV
    bnd_attn             = attn(q_fused, k_bnd, v_bnd)         # patch self-attn on bnd KV
    sem_feat_v2          = sem_feat + out_proj_sem(sem_attn)   # residual, zero-init
    bnd_feat_v2          = bnd_feat + out_proj_bnd(bnd_attn)

Initialization:
    - ``out_proj_sem`` / ``out_proj_bnd`` weight and bias are zero-init,
      so at step 0 the residual deltas are exactly zero and
      ``sem_feat_v2 == sem_feat``, ``bnd_feat_v2 == bnd_feat``.
    - Combined with cloned v2 heads this guarantees the CR-L baseline at
      step 0 before any gradient arrives.

The serialization / padding / relative-position helpers are intentionally
copy-pasted from ``BoundaryGatingModule`` in ``heads.py`` to keep CR-J and
CR-M fully decoupled — they can be consolidated once both experiments
have served their purpose.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.utils.misc import offset2bincount


class CrossStreamFusionAttention(nn.Module):
    """Cross-stream fusion-query patch attention for CR-M (g v4).

    Shared serialized patches of size ``patch_size`` are used for both
    streams so the cross-stream attention happens within well-defined
    neighborhoods. Each point produces a fused query from its semantic and
    boundary queries, then attends separately into the semantic and
    boundary patches' keys/values.
    """

    def __init__(
        self,
        channels: int,
        patch_size: int = 48,
        num_heads: int = 4,
        enable_flash: bool = True,
    ):
        super().__init__()
        assert channels % num_heads == 0, (
            "channels must be divisible by num_heads"
        )
        self.channels = channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.enable_flash = enable_flash

        # Pre-norm for each stream
        self.norm_sem = nn.LayerNorm(channels)
        self.norm_bnd = nn.LayerNorm(channels)

        # Per-stream QKV projections
        self.qkv_sem = nn.Linear(channels, channels * 3)
        self.qkv_bnd = nn.Linear(channels, channels * 3)

        # BFANet-style fusion query MLP: concat(q_sem, q_bnd) → fused query
        self.fusion_q = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

        # Residual output projections — zero-init so step-0 delta == 0
        self.out_proj_sem = nn.Linear(channels, channels)
        self.out_proj_bnd = nn.Linear(channels, channels)
        nn.init.zeros_(self.out_proj_sem.weight)
        nn.init.zeros_(self.out_proj_sem.bias)
        nn.init.zeros_(self.out_proj_bnd.weight)
        nn.init.zeros_(self.out_proj_bnd.bias)

        if enable_flash:
            assert flash_attn is not None, (
                "flash_attn required when enable_flash=True"
            )
            self.rpe = None
        else:
            pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
            rpe_num = 2 * pos_bnd + 1
            self.rpe_pos_bnd = pos_bnd
            self.rpe_num = rpe_num
            self.rpe = nn.Parameter(torch.zeros(3 * rpe_num, num_heads))
            nn.init.trunc_normal_(self.rpe, std=0.02)

    # ------------------------------------------------------------------
    # Serialization helpers (copy-paste of BoundaryGatingModule's helpers)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sem_feat: torch.Tensor,
        bnd_feat: torch.Tensor,
        point,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run cross-stream fusion attention.

        Args:
            sem_feat: (N, C) semantic-stream features.
            bnd_feat: (N, C) boundary-stream features.
            point:    backbone Point dict with ``offset``, ``serialized_order``,
                      ``serialized_inverse``, ``grid_coord``.

        Returns:
            (sem_feat_v2, bnd_feat_v2), each shaped (N, C). At init the
            residual deltas are zero so both outputs equal their inputs.
        """
        H = self.num_heads
        K = self.patch_size
        C = self.channels
        head_dim = C // H

        # Shared patch layout
        pad, unpad, cu_seqlens = self._get_padding_and_inverse(point.offset)
        order = point.serialized_order[0][pad]
        inverse = unpad[point.serialized_inverse[0]]

        # Pre-norm + QKV projections, then reorder into patches
        qkv_s = self.qkv_sem(self.norm_sem(sem_feat))[order]  # (N_pad, 3C)
        qkv_b = self.qkv_bnd(self.norm_bnd(bnd_feat))[order]  # (N_pad, 3C)

        # Split per stream → per-head, (N_pad, H, head_dim) each
        N_pad = qkv_s.shape[0]
        qkv_s = qkv_s.reshape(N_pad, 3, H, head_dim)
        qkv_b = qkv_b.reshape(N_pad, 3, H, head_dim)
        q_s, k_s, v_s = qkv_s.unbind(dim=1)
        q_b, k_b, v_b = qkv_b.unbind(dim=1)

        # BFANet fusion query: concat flat(q_s, q_b) → MLP → reshape to heads
        q_s_flat = q_s.reshape(N_pad, C)
        q_b_flat = q_b.reshape(N_pad, C)
        q_fused = self.fusion_q(
            torch.cat([q_s_flat, q_b_flat], dim=-1)
        )  # (N_pad, C)
        q_fused = q_fused.reshape(N_pad, H, head_dim)

        if self.enable_flash:
            # flash_attn_varlen_func expects (total, H, head_dim) with cu_seqlens
            q_fp = q_fused.to(torch.bfloat16)
            k_sp = k_s.to(torch.bfloat16)
            v_sp = v_s.to(torch.bfloat16)
            k_bp = k_b.to(torch.bfloat16)
            v_bp = v_b.to(torch.bfloat16)

            sem_attn = flash_attn.flash_attn_varlen_func(
                q_fp, k_sp, v_sp,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=K,
                max_seqlen_k=K,
                dropout_p=0.0,
                softmax_scale=self.scale,
            )  # (N_pad, H, head_dim)
            bnd_attn = flash_attn.flash_attn_varlen_func(
                q_fp, k_bp, v_bp,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=K,
                max_seqlen_k=K,
                dropout_p=0.0,
                softmax_scale=self.scale,
            )  # (N_pad, H, head_dim)

            sem_attn = sem_attn.reshape(N_pad, C).to(q_fused.dtype)
            bnd_attn = bnd_attn.reshape(N_pad, C).to(q_fused.dtype)
        else:
            # Non-flash path: explicit softmax with RPE (shared bias)
            num_patch = N_pad // K
            q_fp = q_fused.reshape(num_patch, K, H, head_dim).permute(0, 2, 1, 3)
            k_sp = k_s.reshape(num_patch, K, H, head_dim).permute(0, 2, 1, 3)
            v_sp = v_s.reshape(num_patch, K, H, head_dim).permute(0, 2, 1, 3)
            k_bp = k_b.reshape(num_patch, K, H, head_dim).permute(0, 2, 1, 3)
            v_bp = v_b.reshape(num_patch, K, H, head_dim).permute(0, 2, 1, 3)

            grid_coord_ordered = point.grid_coord[order]
            rpe_bias = self._get_rel_pos(grid_coord_ordered)  # (num_patch, H, K, K)

            logits_s = (q_fp * self.scale) @ k_sp.transpose(-2, -1) + rpe_bias
            attn_s = logits_s.float().softmax(dim=-1).to(q_fp.dtype)
            sem_attn = (attn_s @ v_sp).transpose(1, 2).reshape(N_pad, C)

            logits_b = (q_fp * self.scale) @ k_bp.transpose(-2, -1) + rpe_bias
            attn_b = logits_b.float().softmax(dim=-1).to(q_fp.dtype)
            bnd_attn = (attn_b @ v_bp).transpose(1, 2).reshape(N_pad, C)

        # Undo serialization, apply zero-init output projections, add residual
        sem_delta = self.out_proj_sem(sem_attn[inverse])
        bnd_delta = self.out_proj_bnd(bnd_attn[inverse])

        sem_feat_v2 = sem_feat + sem_delta
        bnd_feat_v2 = bnd_feat + bnd_delta
        return sem_feat_v2, bnd_feat_v2
