"""Clean-reset soft-weighted semantic (CR-O) @ seed=38873367.

Part of workstream clean-reset-s38873367.
Same semantic-only model as CR-A (no boundary head, no aux loss).
Loss: SoftWeightedSemanticLoss — fully continuous support-weighted CE + Lovasz.

Difference from CR-K (boundary_weighted_semantic_train.py):
  CR-K uses truncated soft band: weight = 1 if support <= 0.5, else
  1 + s*9. Transition zone (0 < s <= 0.5) is unweighted.
  CR-O removes the truncation: weight = 1 + s*9 across the full [0,1]
  range. Every point with non-zero support gets some weight increase.

Purpose: minimal smooth extension of CR-A. Tests whether continuous
boundary-proximity weighting alone moves val_mIoU above CR-A's
0.7336 within seed noise (which Phase 5 diagnosis showed is the
practical noise floor for v2.0 single-run results).
"""

from __future__ import annotations

import runpy
from pathlib import Path


_dir = Path(__file__).resolve().parent
repo_root = _dir.parents[2]

model = runpy.run_path(str(_dir / "clean_reset_semantic_model.py"))["model"]

data = runpy.run_path(str(_dir / "clean_reset_data.py"))["data"]
data["train_batch_size"] = 4
data["val_batch_size"] = 1

loss = dict(
    type="SoftWeightedSemanticLoss",
    boundary_ce_weight=10.0,
)
evaluator = dict(type="SemanticEvaluator")

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
work_dir = str(repo_root / "outputs" / "clean_reset_s38873367" / "soft_weighted_semantic")

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
