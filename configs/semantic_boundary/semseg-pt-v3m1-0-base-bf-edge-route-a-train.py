"""Route A training config: Stage-2 v2 model + local within-basin direction coherence.

This config is isolated from the main training config.  It differs from the
Stage-2 v2 train config in two ways only:
  1. Loss type -> RouteASemanticBoundaryLoss (adds basin coherence term).
  2. Dataset transform/collect extended to carry support_id through the
     transform chain and into the batch.

When edge_support_id.npy is absent from a sample directory, BFDataset will
silently omit support_id from data_dict.  The loss then falls back to
coherence_weight=0 behaviour automatically (no crash, no coherence term).
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]

model = runpy.run_path(
    str(
        repo_root
        / "configs"
        / "semantic_boundary"
        / "semseg-pt-v3m1-0-base-bf-edge-stage2-v2-model.py"
    )
)["model"]

# ------------------------------------------------------------------
# Data config: inherit base transforms, extend for support_id.
# ------------------------------------------------------------------
dataset_type = "BFDataset"
data_root = os.environ.get("SBF_DATA_ROOT")
if data_root is None:
    raise RuntimeError(
        "SBF_DATA_ROOT is required; set it to the edge dataset root."
    )

data = dict(
    num_classes=8,
    ignore_index=-1,
    train_batch_size=4,
    val_batch_size=1,
    names=[
        "balustrade",
        "balcony",
        "advboard",
        "wall",
        "eave",
        "column",
        "window",
        "clutter",
    ],
    train=dict(
        type=dataset_type,
        split=("training",),
        data_root=data_root,
        transform=[
            dict(type="InjectIndexValidKeys", keys=("edge", "support_id")),
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.06,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=40960, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "edge", "support_id"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="validation",
        data_root=data_root,
        transform=[
            dict(type="InjectIndexValidKeys", keys=("edge", "support_id")),
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.06,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "origin_segment",
                    "inverse",
                    "edge",
                    "support_id",
                ),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
)

# ------------------------------------------------------------------
# Loss: RouteASemanticBoundaryLoss with basin coherence.
# ------------------------------------------------------------------
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
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

weight = None
resume = False
work_dir = str(repo_root / "outputs" / "semantic_boundary_route_a_train")

runtime = dict(
    log_freq=1,
    val_log_freq=1,
    save_freq=100,
    grad_accum_steps=6,
    mix_prob=0.0,
    enable_amp=True,
)

trainer = dict(
    total_epoch=2000,
    eval_epoch=100,
    num_workers=8,
    max_train_batches=None,
    max_val_batches=None,
)
