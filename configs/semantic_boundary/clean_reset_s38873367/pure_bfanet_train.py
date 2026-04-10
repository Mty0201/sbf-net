"""Clean-reset pure BFANet control (CR-N) @ seed=38873367.

Part of workstream clean-reset-s38873367.
Same dual-head model as CR-L (SharedBackboneSemanticSupportModel).
Loss: PureBFANetLoss — faithful BFANet reproduction.

Differences from CR-L (boundary_binary_train.py):
  - Semantic CE: hard-mask 10x upweight where `support > 0.5`, else 1.
    CR-L uses continuous sample_weight_scale=9 instead.
  - Boundary BCE: unweighted (no per-point sample_weight_scale).
  - Dice: **global** over the whole output, not local to `support>0`.

Everything else (model, data, optimizer, scheduler, seed, epochs, batch,
grid_size, mix_prob, amp, grad_accum) is identical to CR-L for apples-to-apples.
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
    type="PureBFANetLoss",
    aux_weight=1.0,
    boundary_ce_weight=10.0,
    boundary_threshold=0.5,
    pos_weight=1.0,
    dice_weight=1.0,
    dice_smooth=1.0,
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
work_dir = str(repo_root / "outputs" / "clean_reset_s38873367" / "pure_bfanet")

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
