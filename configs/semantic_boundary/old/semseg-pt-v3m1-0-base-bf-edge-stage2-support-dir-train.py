"""Stage-2 training config for support-conditioned direction re-entry."""
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
        / "semseg-pt-v3m1-0-base-bf-edge-stage2-support-dir-model.py"
    )
)["model"]

data = runpy.run_path(
    str(repo_root / "configs" / "bf" / "semseg-pt-v3m1-0-base-bf.py")
)["data"]
data["train_batch_size"] = 4
data["val_batch_size"] = 1

loss = dict(
    type="SemanticBoundaryLoss",
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
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

weight = None
resume = False
work_dir = str(repo_root / "outputs" / "semantic_boundary_stage2_support_dir_train")

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
