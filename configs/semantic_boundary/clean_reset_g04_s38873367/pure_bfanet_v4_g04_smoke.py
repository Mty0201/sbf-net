"""CR-Q smoke: 2 epoch × 20 train batches, new WSL env validation."""

from __future__ import annotations

import runpy
from pathlib import Path

_dir = Path(__file__).resolve().parent
repo_root = _dir.parents[2]

_base = runpy.run_path(str(_dir / "pure_bfanet_v4_g04_train.py"))

model = _base["model"]
data = _base["data"]
loss = _base["loss"]
evaluator = _base["evaluator"]
optimizer = _base["optimizer"]
param_dicts = _base["param_dicts"]
scheduler = _base["scheduler"]
seed = _base["seed"]
weight = None
resume = False

work_dir = str(repo_root / "outputs" / "clean_reset_g04_s38873367" / "pure_bfanet_v4_g04_smoke")

runtime = dict(
    log_freq=1,
    val_log_freq=1,
    save_freq=100,
    grad_accum_steps=12,
    mix_prob=0.8,
    enable_amp=True,
)

trainer = dict(
    total_epoch=2,
    eval_epoch=1,
    num_workers=4,
    max_train_batches=20,
    max_val_batches=5,
)
