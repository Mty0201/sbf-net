"""Clean-reset boundary binary (CR-L) @ seed=38873367.

Part of workstream clean-reset-s38873367.
Same dual-head model as CR-I (SharedBackboneSemanticSupportModel).
Loss: BoundaryBinaryLoss — BFANet-inspired binary boundary classification
enhanced with continuous support sample weighting.
  - Aux: binary BCE (support > 0.9 = positive), sample weight = 1 + support * 9
  - Semantic CE: GT support weighted (truncated > 0.5), same as CR-I
"""

from __future__ import annotations

import runpy
from pathlib import Path


_dir = Path(__file__).resolve().parent
repo_root = _dir.parents[2]

model = runpy.run_path(str(_dir / "clean_reset_support_model.py"))["model"]

data = runpy.run_path(str(_dir / "clean_reset_data.py"))["data"]
data["train_batch_size"] = 4
data["val_batch_size"] = 1

loss = dict(
    type="BoundaryBinaryLoss",
    aux_weight=0.3,
    boundary_ce_weight=10.0,
    sample_weight_scale=9.0,
    boundary_threshold=0.9,
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
work_dir = str(repo_root / "outputs" / "clean_reset_s38873367" / "boundary_binary")

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
