"""Variant C ablation: SmoothL1+Tversky support, no focus term, 300 eval epochs.

Tests whether the cleaner SharedBackboneSemanticSupportModel architecture matches
the support-only baseline when using the same support loss formulation.
"""

from __future__ import annotations

import runpy
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]

model = runpy.run_path(
    str(repo_root / "configs" / "semantic_boundary" / "semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-model.py")
)["model"]

data = runpy.run_path(
    str(repo_root / "configs" / "bf" / "semseg-pt-v3m1-0-base-bf.py")
)["data"]
data["train_batch_size"] = 4
data["val_batch_size"] = 1

loss = dict(
    type="RedesignedSupportFocusLoss",
    support_reg_weight=1.0,
    support_cover_weight=0.2,
    support_tversky_alpha=0.3,
    support_tversky_beta=0.7,
    focus_mode="none",
)

evaluator = dict(
    type="RedesignedSupportFocusEvaluator",
    boundary_metric_threshold=0.2,
)

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

seed = 3407
weight = None
resume = False
work_dir = str(repo_root / "outputs" / "redesigned_variant_c_train")

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
