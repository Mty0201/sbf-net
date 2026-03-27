"""Stage-2 model config with support-conditioned direction head."""

from __future__ import annotations

import runpy
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]

model = runpy.run_path(
    str(
        repo_root
        / "configs"
        / "semantic_boundary"
        / "semseg-pt-v3m1-0-base-bf-edge-model.py"
    )
)["model"]

model["edge_head_cfg"] = dict(type="SupportConditionedEdgeHead")
