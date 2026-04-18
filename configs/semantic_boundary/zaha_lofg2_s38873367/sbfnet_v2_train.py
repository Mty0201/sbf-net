"""ZAHA exp 4: SBF-net v2 — full CR-SDE (decoupler + g v4 fusion + dual supervision).

DecoupledBFANetSegmentorGRef: PTv3 backbone + SubtractiveDecoupling +
CrossStreamFusionAttention (g v4) + v1 (pre-fusion) / v2 (post-fusion) dual heads.
CRSDLoss: s_weight continuous 10x CE upweight + multiclass Lovasz on semantics,
BCE + global Dice on margin heads. Loss is applied to BOTH v1 and v2 branches
when seg_logits_v1 / marg_logits_v1 are present.
"""

from __future__ import annotations
import runpy
from pathlib import Path

_dir = Path(__file__).resolve().parent
repo_root = _dir.parents[2]

model = runpy.run_path(str(_dir / "zaha_decoupled_gref_model.py"))["model"]
data = runpy.run_path(str(_dir / "zaha_data.py"))["data"]
data["train_batch_size"] = 2
data["val_batch_size"] = 1

loss = dict(
    type="CRSDLoss",
    aux_weight=1.0,
    boundary_ce_weight=10.0,
    dice_weight=1.0,
    dice_smooth=1.0,
)
evaluator = dict(type="ZAHASupportFocusEvaluator")

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
work_dir = str(repo_root / "outputs" / "zaha_lofg2_s38873367" / "sbfnet_v2")

runtime = dict(
    log_freq=1, val_log_freq=1, save_freq=100,
    grad_accum_steps=6, mix_prob=0.8, enable_amp=True,
)
trainer = dict(
    total_epoch=2000, eval_epoch=100,
    num_workers=8, max_train_batches=None, max_val_batches=None,
)
