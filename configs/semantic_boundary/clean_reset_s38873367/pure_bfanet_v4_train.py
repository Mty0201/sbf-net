"""Clean-reset CR-P: full BFANet reproduction @ seed=38873367.

Part of workstream clean-reset-s38873367.

CR-P = CR-N loss + CR-M dual-stream model. The "one-shot full BFANet
reproduction" control:

  - Boundary detection: offline radius search at r = 0.06 m absolute
    (BFANet ScanNet reference), consumed via ``boundary_mask_r060.npy``
    per-chunk precomputed masks.
  - Loss design: PureBFANetLoss (hard-mask 10x semantic CE upweight,
    unweighted BCE with pos_weight=1, global Dice over the whole output),
    wrapped in dual-supervision over v1 (pre-attention) and v2
    (post-attention) outputs.
  - Dual-layer supervision weights: v1 = 0.5, v2 = 1.0 — BFANet-faithful,
    v2 is the primary refined prediction and v1 is an auxiliary
    regularizer preventing the backbone from degenerating.
  - Model: BoundaryGatedSemanticModelV4 — shared PTv3 backbone, v1 heads,
    CrossStreamFusionAttention (g v4, zero-init residual), v2 heads
    cloned from v1 at init so step 0 is numerically equivalent to a
    pure single-stream baseline.

This is the definitive test of "does faithful BFANet beat semantic-only
CR-A (0.7336) on our building-facade dataset?". Differences from CR-N
are architectural only (dual stream + g v4), differences from CR-M are
loss-only (PureBFANetLoss with new r=0.06 mask, plus 0.5:1.0 v1/v2
weighting).

Everything else (data, optimizer, scheduler, seed, epochs, batch sizes,
grid_size, mix_prob, amp, grad_accum) is byte-identical to CR-M and
CR-L for apples-to-apples comparison.
"""

from __future__ import annotations

import runpy
from pathlib import Path


_dir = Path(__file__).resolve().parent
repo_root = _dir.parents[2]

model = runpy.run_path(str(_dir / "clean_reset_gated_v4_model.py"))["model"]

data = runpy.run_path(str(_dir / "clean_reset_data.py"))["data"]
data["train_batch_size"] = 4
data["val_batch_size"] = 1

loss = dict(
    type="DualSupervisionPureBFANetLoss",
    aux_weight=0.3,
    boundary_ce_weight=10.0,
    boundary_threshold=0.5,
    pos_weight=1.0,
    dice_weight=1.0,
    dice_smooth=1.0,
    v1_weight=0.5,
    v2_weight=1.0,
)
evaluator = dict(type="RedesignedSupportFocusEvaluator")

optimizer = dict(
    type="AdamW",
    lr=0.006,
    weight_decay=0.05,
)

param_dicts = [dict(keyword="block", lr=0.0006)]

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

seed = 38873367
weight = None
resume = False
work_dir = str(repo_root / "outputs" / "clean_reset_s38873367" / "pure_bfanet_v4")

runtime = dict(
    log_freq=1,
    val_log_freq=1,
    save_freq=100,
    grad_accum_steps=6,
    mix_prob=0.8,
    enable_amp=True,
)

trainer = dict(
    total_epoch=2000,
    eval_epoch=100,
    num_workers=8,
    max_train_batches=None,
    max_val_batches=None,
)
