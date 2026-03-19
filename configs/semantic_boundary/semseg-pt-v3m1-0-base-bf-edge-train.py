"""Full stage-1 training config for semantic boundary training."""

from __future__ import annotations

import runpy
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]

model = runpy.run_path(
    str(repo_root / "configs" / "semantic_boundary" / "semseg-pt-v3m1-0-base-bf-edge-model.py")
)["model"]

data = runpy.run_path(
    str(repo_root / "configs" / "bf" / "semseg-pt-v3m1-0-base-bf.py")
)["data"]

optimizer = dict(
    type="Adam",
    lr=1e-3,
    weight_decay=0.0,
)

trainer = dict(
    epochs=300,
    batch_size=1,
    num_workers=4,
    max_train_batches=None,
    max_val_batches=None,
    cpu_fallback_shell_backbone=False,
    work_dir=str(repo_root / "outputs" / "semantic_boundary_train"),
)
