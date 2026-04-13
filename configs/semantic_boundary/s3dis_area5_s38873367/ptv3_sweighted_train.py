"""S3DIS exp 5: pure PTv3 + s_weight-weighted semantic CE (no aux head).

SharedBackboneSemanticModel + SoftWeightedSemanticLoss.
Every point's CE weight = 1 + s_weight * 9 (core boundary -> 10, background -> 1).
No boundary/support head, no aux loss — minimal smooth extension of semantic_only.
"""

from __future__ import annotations
import runpy
from pathlib import Path

_dir = Path(__file__).resolve().parent
repo_root = _dir.parents[2]

model = runpy.run_path(str(_dir / "s3dis_semantic_model.py"))["model"]
data = runpy.run_path(str(_dir / "s3dis_data.py"))["data"]
data["train_batch_size"] = 2
data["val_batch_size"] = 1

loss = dict(type="SoftWeightedSemanticLoss", boundary_ce_weight=10.0)
evaluator = dict(type="SemanticEvaluator")

optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
param_dicts = [dict(keyword="block", lr=0.0006)]
scheduler = dict(
    type="OneCycleLR", max_lr=[0.006, 0.0006],
    pct_start=0.05, anneal_strategy="cos",
    div_factor=10.0, final_div_factor=1000.0,
)

seed = 38873367
weight = None
resume = False
work_dir = str(repo_root / "outputs" / "s3dis_area5_s38873367" / "ptv3_sweighted")

runtime = dict(
    log_freq=1, val_log_freq=1, save_freq=100,
    grad_accum_steps=12, mix_prob=0.8, enable_amp=True,
)
trainer = dict(
    total_epoch=100, eval_epoch=100,
    num_workers=8, max_train_batches=None, max_val_batches=None,
)
