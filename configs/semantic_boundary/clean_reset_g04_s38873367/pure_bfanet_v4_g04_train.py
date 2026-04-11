"""Clean-reset CR-Q: CR-P architecture + user's validated grid=0.04 recipe.

CR-Q is the single-variable test of the "grid=0.06 rasterisation floor
caps the aux-head route at ~0.725" hypothesis. It is CR-P with exactly
three data-pipeline parameters changed, and batch_size / grad_accum
rebalanced so the effective batch matches CR-P for apples-to-apples
comparison:

  grid_size:         0.06  -> 0.04      (data)
  SphereCrop rate:   0.6   -> 0.4       (data)
  SphereCrop max:    40960 -> 80000     (data)
  train_batch_size:  4     -> 2         (memory — larger point clouds)
  grad_accum_steps:  6     -> 12        (keeps effective batch = 24)

Everything else — model architecture (BoundaryGatedSemanticModelV4),
loss (DualSupervisionPureBFANetLoss with v1_weight=v2_weight=1.0,
aux_weight=1.0, boundary_ce_weight=10.0, pos_weight=1.0, dice_weight=1.0),
boundary_mask source (boundary_mask_r060.npy — grid-independent, raw-point
KNN at r=0.06 m absolute), optimiser, scheduler, seed, total epochs — is
byte-identical to CR-P.

## Why this experiment is necessary

CR-P ep1-78 on grid=0.06 landed best val_mIoU 0.7241 @ ep45, statistically
identical to CR-L 0.7251 (Δ=0.001). Aux-head ceiling confirmed at ~0.725
across single-stream (CR-L) and dual-stream (CR-P) architectures under
the current grid. Post-hoc analysis found:

  - g v4 fusion is near-identity (v2 − v1 ≈ 0.003-0.007 across all epochs)
  - BFANet naive BCE does NOT collapse on our 7% positive rate
  - train_sem is 0.18-0.64 higher than CR-L while val is identical →
    train_sem not a reliable predictor under single-seed noise

Real-transform measurements on 5 chunks × 15 Monte-Carlo runs show that
at grid=0.06+rate=0.6+max=40960:
  - boundary voxels are 3.5 cm apart (just over half a grid diagonal)
  - only 6.75% of non-boundary voxels lie within 6 cm of a boundary voxel
  - the boundary band is effectively rasterised: b→nearest-nonb p50 = 6 cm
    = exactly 1 grid step, leaving no soft transition layer for attention
    to resolve

Widening the radius r was tested and rejected — at r=0.12, 33% of wall
and 29% of window voxels become "boundary", polluting class interiors
and inflating the 10× semantic CE upweight onto non-boundary geometry,
while `nonb<6cm` only rises from 6.46% to 7.72%. Narrowing grid is the
only remaining lever.

Switching to grid=0.04+rate=0.4+max=80000 (the user's validated
ScanNet/PTv3 recipe):
  - b→b p50 drops 3.47 → 2.54 cm (boundaries become spatially connected
    in grid-neighbour terms, so cross-stream attention K neighbours can
    actually group them)
  - nonb<6cm rises 6.75% → 10.28% (+52% more transition voxels per fixed
    physical distance — this is the signal aux head needs)
  - raw boundary ratio stays at 11-12% (no class pollution)

## Evidence files

  - /tmp/transform_probe.py         real pipeline trace on chunk 020103
  - /tmp/transform_probe2.py        post-transform geometry 3 chunks
  - /tmp/transform_probe3.py        user_04 recipe vs baseline 5 chunks
  - /tmp/widen_r_probe.py           widening r at grid=0.06 measurement
  - .planning/phases/05-boundary-proximity-cue-experiment/
        cr_q_grid_rasterization_analysis.md   full writeup

## Memory budget note

Per-batch point count goes from 40960 to 80000 (×1.95). PTv3 peak memory
is roughly linear in the point count for the serialised attention layers,
so halving batch_size (4 → 2) keeps peak memory close to CR-P's. If this
proves too tight on the real env, fall back to grad_accum_steps=24 and
train_batch_size=1 — the effective batch stays 24 but wall time goes up
~25%. DO NOT drop grad_accum below 12, or the effective batch falls
below CR-P and the comparison is contaminated by optimisation dynamics.
"""

from __future__ import annotations

import runpy
from pathlib import Path


_dir = Path(__file__).resolve().parent
repo_root = _dir.parents[2]

model = runpy.run_path(str(_dir / "clean_reset_gated_v4_model.py"))["model"]

data = runpy.run_path(str(_dir / "clean_reset_data_g04.py"))["data"]
data["train_batch_size"] = 2
data["val_batch_size"] = 1

loss = dict(
    type="DualSupervisionPureBFANetLoss",
    aux_weight=1.0,
    boundary_ce_weight=10.0,
    boundary_threshold=0.5,
    pos_weight=1.0,
    dice_weight=1.0,
    dice_smooth=1.0,
    v1_weight=1.0,
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
work_dir = str(repo_root / "outputs" / "clean_reset_g04_s38873367" / "pure_bfanet_v4_g04")

runtime = dict(
    log_freq=1,
    val_log_freq=1,
    save_freq=100,
    grad_accum_steps=12,
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
