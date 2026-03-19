"""Forward smoke test for the shared-backbone semantic-boundary model."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch
import torch.nn as nn


def bootstrap_paths() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    pointcept_root = repo_root.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(pointcept_root))
    return repo_root


def main():
    repo_root = bootstrap_paths()

    import project.models  # noqa: F401
    from pointcept.models import build_model

    cfg_path = (
        repo_root
        / "configs"
        / "semantic_boundary"
        / "semseg-pt-v3m1-0-base-bf-edge-model.py"
    )
    cfg = runpy.run_path(str(cfg_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device).eval()
    print("env: ptv3")
    print(f"config: {cfg_path}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"device: {device}")
    print(f"model_type: {type(model).__name__}")
    print(f"backbone_type: {type(model.backbone).__name__}")

    num_points = 2048
    pseudo_input = dict(
        coord=torch.randn(num_points, 3, dtype=torch.float32, device=device),
        grid_coord=torch.randint(0, 64, (num_points, 3), dtype=torch.int64, device=device),
        feat=torch.randn(num_points, 6, dtype=torch.float32, device=device),
        offset=torch.tensor([num_points], dtype=torch.int64, device=device),
    )

    smoke_mode = "actual_backbone_cuda"
    if not torch.cuda.is_available():
        smoke_mode = "shell_only_no_cuda"

        class IdentityPointBackbone(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.proj = nn.Linear(in_channels, out_channels)

            def forward(self, point):
                point.feat = self.proj(point.feat)
                return point

        model.backbone = IdentityPointBackbone(
            in_channels=pseudo_input["feat"].shape[1],
            out_channels=model.semantic_head.proj.in_features,
        ).to(device)

    with torch.no_grad():
        output = model(pseudo_input)

    print(f"smoke_mode: {smoke_mode}")
    print("output_keys:", sorted(output.keys()))
    for key in sorted(output.keys()):
        value = output[key]
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        dtype = str(value.dtype) if hasattr(value, "dtype") else type(value).__name__
        print(f"{key}: shape={shape}, dtype={dtype}")


if __name__ == "__main__":
    main()
