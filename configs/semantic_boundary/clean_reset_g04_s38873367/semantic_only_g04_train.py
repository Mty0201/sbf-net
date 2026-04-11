"""Clean-reset CR-R: pure PTv3 grid=0.04 baseline anchor (Phase 8 slot #1).

Hyperparameters aligned with CR-Q/CR-V for apples-to-apples comparison
against DualSupervisionPureBFANetLoss (CR-Q) and
DualSupervisionSupportWeightedBFANetLoss (CR-V) at the same grid, batch
and effective batch. Loss: SemanticOnlyLoss (CE + Lovasz). Model:
SharedBackboneSemanticModel (no boundary branch, no support branch).

CR-R is the no-aux-head anchor the Phase 8 comparison hangs on: if it
lands at the same val_mIoU as CR-Q/CR-V, boundary / support supervision
contributes nothing at grid=0.04 and the aux-head route is closed.
If CR-Q/CR-V beat CR-R beyond the ~0.007 single-seed noise floor, that
delta is the real signal from boundary-proximity weighting (hard or
continuous, as tested by CR-Q vs CR-V).

Hyperparameters that differ from CR-A (the grid=0.06 semantic_only_train):
    data:              clean_reset_data.py       -> clean_reset_data_g04.py
    train_batch_size:  4                         -> 2
    grad_accum_steps:  6                         -> 12  (eff batch = 24)
    work_dir:          clean_reset_s38873367/... -> clean_reset_g04_s38873367/...

Everything else (optimizer lr=0.006 wd=0.05, block-keyword param_dict at
0.0006, OneCycleLR with max_lr=[0.006, 0.0006] pct_start=0.05,
seed=38873367, total_epoch=2000, eval_epoch=100, mix_prob=0.8, AMP on,
num_workers=8) is byte-identical to CR-A semantic_only_train.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path


_dir = Path(__file__).resolve().parent
repo_root = _dir.parents[2]

model = runpy.run_path(str(_dir / "clean_reset_semantic_model.py"))["model"]

data = runpy.run_path(str(_dir / "clean_reset_data_g04.py"))["data"]
data["train_batch_size"] = 2
data["val_batch_size"] = 1

loss = dict(type="SemanticOnlyLoss")
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
work_dir = str(repo_root / "outputs" / "clean_reset_g04_s38873367" / "semantic_only")

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
