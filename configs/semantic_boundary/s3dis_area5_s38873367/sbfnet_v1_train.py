"""S3DIS exp 3: SBF-net v1 — decoupler + s_weight CE weighting, v1 only.

DecoupledBFANetSegmentorV1 (PTv3 backbone + SubtractiveDecoupling + seg/marg heads).
CRSDLoss: s_weight continuous 10x CE upweight + multiclass Lovasz on semantics,
BCE + global Dice on the margin head supervised by boundary_mask.
No post-fusion v2 branch — this is the single-supervision decoupled model.
"""

from __future__ import annotations
import runpy
from pathlib import Path

_dir = Path(__file__).resolve().parent
repo_root = _dir.parents[2]

model = runpy.run_path(str(_dir / "s3dis_decoupled_v1_model.py"))["model"]
data = runpy.run_path(str(_dir / "s3dis_data.py"))["data"]
data["train_batch_size"] = 2
data["val_batch_size"] = 1

loss = dict(
    type="CRSDLoss",
    aux_weight=1.0,
    boundary_ce_weight=10.0,
    dice_weight=1.0,
    dice_smooth=1.0,
)
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
work_dir = str(repo_root / "outputs" / "s3dis_area5_s38873367" / "sbfnet_v1")

runtime = dict(
    log_freq=1, val_log_freq=1, save_freq=100,
    grad_accum_steps=12, mix_prob=0.8, enable_amp=True,
)
trainer = dict(
    total_epoch=100, eval_epoch=100,
    num_workers=8, max_train_batches=None, max_val_batches=None,
)
