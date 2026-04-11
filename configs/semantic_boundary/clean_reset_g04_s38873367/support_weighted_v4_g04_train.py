"""Clean-reset CR-V: CR-Q architecture/data + continuous support-weighted semantic CE.

CR-V is a single-variable delta from CR-Q (``pure_bfanet_v4_g04_train.py``):

    loss.type:   DualSupervisionPureBFANetLoss
              -> DualSupervisionSupportWeightedBFANetLoss
    work_dir:    .../pure_bfanet_v4_g04
              -> .../support_weighted_v4_g04

Everything else — model architecture (BoundaryGatedSemanticModelV4),
data recipe (grid=0.04, rate=0.4, max=80000), seed, epochs, batch size,
gradient accumulation, optimiser, scheduler, AMP — is byte-identical to
CR-Q. This isolates a single question: does replacing the binary
``1 + boundary_mask * 9`` semantic CE weight with a continuous
``1 + s_weight * 9`` weight (core=1.0, exp-decay across 6-12cm buffer,
0 outside) help on top of the CR-Q grid=0.04 baseline?

## Design

The corrected Phase 8 insight (2026-04-11, visual inspection of the
grid=0.06 vs grid=0.04 probe XYZ files) is that r=0.06 was never a band
— it was always two parallel 1-voxel scatter lines on each side of a
1D edge curve, jittering stochastically every epoch because
GridSample's train-mode random-representative selection picks a
different raw point per voxel per iteration. At grid=0.04 the scatter
lines thicken into continuous bands, but the ~12 cm wide "transition
region" still has no internal structure to regress against. Therefore:

  - Boundary branch uses the HARD binary ``boundary_mask_r060`` target,
    identical to CR-Q. There is nothing inside the band to regress to.
  - Continuous support retreats to a PURE semantic CE weighting device.
    Its only role is to extend the 10× → 1× CE weight footprint wider
    than the hard mask, so the semantic head spends more gradient on
    points near (but not on) real semantic edges.

``s_weight`` is precomputed offline by
``scripts/data/generate_support_weight.py`` into
``s_weight_r060_r120.npy`` next to each chunk's ``coord.npy``:

    s_weight = 1.0                               if d <= 0.06 m
               exp(-3 * (d - 0.06) / 0.06)       if 0.06 < d <= 0.12 m
               0.0                               otherwise

where d is the per-point distance to the nearest point of a different
semantic class. Sample values: d=0.06 → 1.00 → 10.0×, d=0.08 → 0.37 →
4.3×, d=0.10 → 0.14 → 2.3×, d=0.12 → 0.05 → 1.5×, d>0.12 → 0 → 1.0×.

## Success criterion

CR-V is the full Phase 8 answer to "does continuous boundary-proximity
weighting on the semantic CE beat the hard-mask binary weighting under
grid=0.04?". Decision rule vs CR-Q (or CR-R if run):
  - best(CR-V) > best(CR-Q) beyond noise floor (~0.007) → buffer shell
    contributes, CR-V is the v2.0 headline result
  - best(CR-V) ≈ best(CR-Q) → weighting shape does not matter, CR-R/Q
    is what to build on
  - best(CR-V) < best(CR-Q) → continuous weighting is actively harmful

## Memory budget

Identical to CR-Q: bs=2, grad_accum=12, eff=24, AMP on. Peak CUDA
expected ~3.7-4 GiB at bs=2 (CR-Q measured 3.7 GiB). If OOM, fall back
to grad_accum=24 / bs=1.
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
    type="DualSupervisionSupportWeightedBFANetLoss",
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
work_dir = str(repo_root / "outputs" / "clean_reset_g04_s38873367" / "support_weighted_v4_g04")

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
