"""Smoke-only config for Route A runtime check.

Verifies that:
  - RouteASemanticBoundaryLoss initialises correctly.
  - Training loop runs with coherence path active (when support_id present).
  - Training loop runs gracefully when support_id is absent (coherence = 0).
"""
# ======================================================================
# LEGACY CONFIG — belongs to the old experiment chain (Phases 5-11).
# Not the canonical baseline. See configs/semantic_boundary/clean_reset_s38873367/
# for the current clean-reset experiment configs.
# ======================================================================


from __future__ import annotations

import runpy
from pathlib import Path


repo_root = Path(__file__).resolve().parents[3]

model = runpy.run_path(
    str(
        repo_root
        / "configs"
        / "semantic_boundary" / "old"
        / "semseg-pt-v3m1-0-base-bf-edge-stage2-v2-model.py"
    )
)["model"]

data = runpy.run_path(
    str(
        repo_root
        / "configs"
        / "semantic_boundary" / "old"
        / "semseg-pt-v3m1-0-base-bf-edge-route-a-train.py"
    )
)["data"]
data["train_batch_size"] = 1
data["val_batch_size"] = 1

loss = dict(
    type="RouteASemanticBoundaryLoss",
    tau_dir=1e-3,
    dist_scale=0.08,
    support_weight=1.0,
    support_cover_weight=0.2,
    support_reg_weight=1.0,
    support_tversky_alpha=0.3,
    support_tversky_beta=0.7,
    dir_weight=1.0,
    dist_weight=0.0,
    coherence_weight=0.1,
    local_radius=0.30,
    max_points_per_basin=100,
)

evaluator = dict(
    type="SemanticBoundaryEvaluator",
    tau_dir=1e-3,
    dist_scale=0.08,
    support_weight=1.0,
    support_cover_weight=0.2,
    support_reg_weight=1.0,
    support_tversky_alpha=0.3,
    support_tversky_beta=0.7,
    dir_weight=1.0,
    dist_weight=0.0,
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
work_dir = str(repo_root / "outputs" / "semantic_boundary_route_a_train_smoke")

runtime = dict(
    log_freq=1,
    val_log_freq=1,
    save_freq=None,
    grad_accum_steps=2,
    mix_prob=0.0,
    enable_amp=False,
)

trainer = dict(
    total_epoch=2,
    eval_epoch=1,
    num_workers=0,
    max_train_batches=1,
    max_val_batches=1,
)
