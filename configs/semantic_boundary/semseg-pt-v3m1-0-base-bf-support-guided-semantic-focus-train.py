"""Active-route training config for the support-guided semantic focus route.

This config is the Phase 7 active implementation route. It trains from scratch
with no legacy checkpoint loading. The model emits seg_logits + support_pred
only — no dir_pred or dist_pred. The loss and evaluator consume support_pred
+ edge without depending on legacy edge_pred.

Checkpoint compatibility note: SharedBackboneSemanticSupportModel has a different
head structure than SharedBackboneSemanticBoundaryModel (no EdgeHead, no dir/dist
heads). Loading a legacy boundary-model checkpoint with strict=True will fail.
This is intentional — the active route trains from scratch. If partial backbone
loading is needed in future, set weight to the checkpoint path and modify
_load_checkpoint_or_weight to use strict=False, but that is out of scope for
Phase 7.
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
    type="SupportGuidedSemanticFocusLoss",
    support_loss_weight=1.0,
    focus_loss_weight=1.0,
    focus_lambda=1.0,
    focus_gamma=1.0,
)
evaluator = dict(
    type="SupportGuidedSemanticFocusEvaluator",
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
work_dir = str(repo_root / "outputs" / "support_guided_semantic_focus_train")

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
