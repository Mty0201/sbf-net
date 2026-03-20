"""Full semantic-only training config for trainer calibration."""

from __future__ import annotations

import runpy
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]

model = runpy.run_path(
    str(repo_root / "configs" / "semantic_boundary" / "semseg-pt-v3m1-0-base-bf-semantic-model.py")
)["model"]

data = runpy.run_path(
    str(repo_root / "configs" / "semantic_boundary" / "semseg-pt-v3m1-0-base-bf-semantic-data.py")
)["data"]
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

seed = 3407
weight = None
resume = False
work_dir = str(repo_root / "outputs" / "semantic_only_train")

runtime = dict(
    log_freq=1,
    val_log_freq=4,
    save_freq=100,
    grad_accum_steps=6,
)

trainer = dict(
    total_epoch=3000,
    eval_epoch=100,
    batch_size=2,
    num_workers=8,
    max_train_batches=None,
    max_val_batches=None,
    cpu_fallback_shell_backbone=False,
)
