"""ZAHA exp 1: pure semantic PTv3 (CE + Lovasz). Baseline anchor."""

from __future__ import annotations
import runpy
from pathlib import Path

_dir = Path(__file__).resolve().parent
repo_root = _dir.parents[2]

model = runpy.run_path(str(_dir / "zaha_semantic_model.py"))["model"]
data = runpy.run_path(str(_dir / "zaha_data.py"))["data"]
data["train_batch_size"] = 2
data["val_batch_size"] = 1

loss = dict(type="SemanticOnlyLoss")
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
work_dir = str(repo_root / "outputs" / "zaha_lofg2_s38873367" / "semantic_only")

runtime = dict(
    log_freq=1, val_log_freq=1, save_freq=100,
    grad_accum_steps=6, mix_prob=0.8, enable_amp=True,
)
trainer = dict(
    total_epoch=4000, eval_epoch=40,
    num_workers=8, max_train_batches=None, max_val_batches=None,
)
