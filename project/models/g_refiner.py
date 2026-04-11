"""CR-SDE: GatedSegRefiner.

A feature-level refiner module that sits between ``SubtractiveDecoupling`` and
the two logit heads. It takes the already-decoupled ``point_seg`` (seg stream)
and ``point_marg`` (margin stream) and writes a zero-initialised residual back
onto ``point_seg.feat``:

    gate      = sigmoid(Linear(point_marg.feat))        # [N,1], step0 = 0.5
    attn_out  = SerializedAttention(LN(point_seg))      # seg-stream self-attn
    delta     = tanh(alpha_raw) * gate * out_proj(attn_out)
    point_seg.feat = point_seg.feat + delta

Step-0 identity is guaranteed by ``alpha_raw = 0`` alone: ``alpha_g =
tanh(alpha_raw) = 0`` makes ``delta = 0`` regardless of what ``gate`` and
``attn_out`` are. The tanh bounds per-channel scale to (-1, 1) so a
runaway optimizer cannot explode ``delta`` relative to the seg feature.

Other parameters are initialised so the gradient path through ``alpha_raw``
is ALIVE at step 0:

  * ``gate_proj`` is zero-init (weight + bias) → ``gate ≡ sigmoid(0) =
    0.5`` at step 0. A constant non-zero gate is enough for
    ``dL/d alpha_raw ∝ gate · attn_out`` to be non-zero.
  * ``out_proj`` uses small-scale normal init (std=0.02) so ``attn_out``
    is not identically zero at step 0. Without this, the step-0 gradient
    on ``alpha_raw`` would be ``gate · 0 = 0`` and the refiner would
    never start learning — a silent dead-branch failure.

This asymmetry (alpha_raw zero, out_proj non-zero) is the minimal init
that satisfies both "strict step-0 identity" AND "non-dead gradient at
step 0". See ``check_cr_sde_smoke.py`` for the formal verification.

This module is self-contained. It is NOT registered with ``MODELS`` — it is
used internally by ``DecoupledBFANetSegmentorGRef``. ``DecoupledBFANetSegmentorV1``
is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from addict import Dict as AddictDict

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    Point,
    PointModule,
    SerializedAttention,
)


class GatedSegRefiner(PointModule):
    def __init__(
        self,
        channels: int = 64,
        num_heads: int = 4,
        patch_size: int = 1024,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        enable_flash: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.attn = SerializedAttention(
            channels=channels,
            num_heads=num_heads,
            patch_size=patch_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=0,
            enable_rpe=False,
            enable_flash=enable_flash,
            upcast_attention=False,
            upcast_softmax=False,
        )
        # gate_proj: zero-init so gate is the constant 0.5 at step 0.
        self.gate_proj = nn.Linear(channels, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        # out_proj: small-scale non-zero init so attn_out is non-zero at
        # step 0 — this is what keeps the gradient path through alpha_raw
        # alive. Zero-initialising out_proj would kill alpha_raw's gradient
        # and the refiner would be a dead branch forever.
        self.out_proj = nn.Linear(channels, channels)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_proj.bias)
        # alpha_raw: zero-init. tanh(0)=0 gives delta=0 (step-0 identity),
        # and tanh'(0)=1 keeps gradient flow maximal at step 0.
        self.alpha_raw = nn.Parameter(torch.zeros(channels))

    @staticmethod
    def _clone_for_attention(point: Point, new_feat: torch.Tensor) -> Point:
        """Shallow clone of a Point carrying only what SerializedAttention reads.

        SerializedAttention reads ``feat``, ``offset``, ``serialized_order``,
        ``serialized_inverse`` and (when ``enable_rpe``) ``grid_coord``. It also
        caches ``pad_*`` / ``unpad_*`` / ``cu_seqlens_*`` keys on the point.
        Clone = fresh Point so those cache keys do not land on point_seg.
        """
        cloned = AddictDict()
        cloned.coord = point.coord
        cloned.grid_coord = point.grid_coord
        cloned.offset = point.offset
        cloned.batch = point.batch
        cloned.feat = new_feat
        for key in list(point.keys()):
            if key.startswith("serialized_"):
                cloned[key] = point[key]
        return Point(cloned)

    def forward(self, point_seg: Point, point_marg: Point) -> tuple[Point, dict]:
        gate = torch.sigmoid(self.gate_proj(point_marg.feat))  # [N,1]

        x_norm = self.norm(point_seg.feat)
        attn_point = self._clone_for_attention(point_seg, x_norm)
        attn_point = self.attn(attn_point)
        attn_out = self.out_proj(attn_point.feat)  # [N,C]

        alpha_g = torch.tanh(self.alpha_raw)  # [C], ∈ (-1, 1)
        delta = alpha_g * gate * attn_out  # broadcast: [C]*[N,1]*[N,C]

        point_seg.feat = point_seg.feat + delta
        if hasattr(point_seg, "sparse_conv_feat"):
            point_seg.sparse_conv_feat = point_seg.sparse_conv_feat.replace_feature(
                point_seg.feat
            )

        with torch.no_grad():
            diagnostics = dict(
                g_alpha_mean=alpha_g.mean().detach(),
                g_alpha_absmax=alpha_g.abs().max().detach(),
                g_gate_mean=gate.mean().detach(),
                g_gate_std=gate.std().detach(),
                g_delta_norm=delta.norm(dim=-1).mean().detach(),
            )
        return point_seg, diagnostics
