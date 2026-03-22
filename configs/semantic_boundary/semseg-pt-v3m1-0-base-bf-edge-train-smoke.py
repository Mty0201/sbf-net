"""Smoke-only training config for semantic boundary stage-1 checks."""

from __future__ import annotations

import runpy
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]

model = runpy.run_path(
    str(
        repo_root
        / "configs"
        / "semantic_boundary"
        / "semseg-pt-v3m1-0-base-bf-edge-model.py"
    )
)["model"]

data = runpy.run_path(
    str(repo_root / "configs" / "bf" / "semseg-pt-v3m1-0-base-bf.py")
)["data"]
data["train_batch_size"] = 1
data["val_batch_size"] = 1

loss = dict(
    type="SemanticBoundaryLoss",
    tau_dir=1e-3,
    support_weight=1.0,
    support_cover_weight=1.0,
    support_reg_weight=0.25,
    support_tversky_alpha=0.3,
    support_tversky_beta=0.7,
    dir_weight=1.0,
    dist_weight=1.0,
)
evaluator = dict(
    type="SemanticBoundaryEvaluator",
    tau_dir=1e-3,
    support_weight=1.0,
    support_cover_weight=1.0,
    support_reg_weight=0.25,
    support_tversky_alpha=0.3,
    support_tversky_beta=0.7,
    dir_weight=1.0,
    dist_weight=1.0,
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
    pct_start=0.5,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

weight = None
resume = False
work_dir = str(repo_root / "outputs" / "semantic_boundary_train_smoke")

runtime = dict(
    log_freq=1,
    val_log_freq=1,
    save_freq=None,
    grad_accum_steps=2,
    mix_prob=0.8,
    enable_amp=False,
)

trainer = dict(
    total_epoch=2,
    eval_epoch=1,
    num_workers=0,
    max_train_batches=1,
    max_val_batches=1,
    cpu_fallback_shell_backbone=True,
)
