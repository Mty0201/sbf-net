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

optimizer = dict(
    type="Adam",
    lr=1e-3,
    weight_decay=0.0,
)

scheduler = dict(
    type="OneCycleLR",
    max_lr=1e-3,
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
)

trainer = dict(
    total_epoch=2,
    eval_epoch=1,
    batch_size=1,
    num_workers=0,
    max_train_batches=1,
    max_val_batches=1,
    cpu_fallback_shell_backbone=True,
)
