"""Stage-2 v2 model config with post-backbone branch split and semantic protection."""
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
        / "semseg-pt-v3m1-0-base-bf-edge-model.py"
    )
)["model"]

model["semantic_adapter_cfg"] = dict(
    type="ResidualFeatureAdapter",
    hidden_channels=64,
    residual_scale=1.0,
    zero_init_last=True,
)
model["boundary_adapter_cfg"] = dict(
    type="ResidualFeatureAdapter",
    hidden_channels=64,
    residual_scale=1.0,
    zero_init_last=True,
)
model["edge_head_cfg"] = dict(type="SupportConditionedEdgeHead")
