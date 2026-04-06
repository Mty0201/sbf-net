"""Stage-2 model config with support-conditioned direction head."""
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

model["edge_head_cfg"] = dict(type="SupportConditionedEdgeHead")
