"""Clean-reset CR-M: cross-stream fusion + dual supervision @ seed=38873367.

Part of workstream clean-reset-s38873367.
Model: BoundaryGatedSemanticModelV4 — dual v1/v2 heads with
CrossStreamFusionAttention (g v4). At init v2 matches v1 exactly, so
step 0 reproduces the CR-L baseline before any g v4 gradient flows.
Loss: DualSupervisionBoundaryBinaryLoss — wraps BoundaryBinaryLoss and
runs it on both v1 and v2 outputs at equal weight, with v1_/v2_ prefixed
log keys so training telemetry can compare the two branches directly.

All other training knobs (optimizer, scheduler, seed, grad_accum_steps,
mix_prob, total_epoch, eval_epoch, batch sizes) are byte-identical to
CR-L (``boundary_binary_train.py``) so any measured delta is attributable
to g v4 + dual supervision.
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
    type="DualSupervisionBoundaryBinaryLoss",
    aux_weight=0.3,
    boundary_ce_weight=10.0,
    sample_weight_scale=9.0,
    boundary_threshold=0.5,
    pos_weight=1.0,
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
work_dir = str(repo_root / "outputs" / "clean_reset_s38873367" / "boundary_gated_v4")

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
