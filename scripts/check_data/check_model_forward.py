"""Forward smoke test for the shared-backbone semantic-boundary model."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import torch


def bootstrap_paths() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    pointcept_root_env = os.environ.get("POINTCEPT_ROOT")
    if pointcept_root_env is None:
        raise RuntimeError(
            "POINTCEPT_ROOT is required; implicit parent-directory fallback has been removed."
        )
    pointcept_root = Path(pointcept_root_env).resolve()
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

    with torch.no_grad():
        output = model(pseudo_input)

    print("path_mode: actual_backbone_only")
    print("output_keys:", sorted(output.keys()))
    for key in sorted(output.keys()):
        value = output[key]
        shape = tuple(value.shape) if hasattr(value, "shape") else None
        dtype = str(value.dtype) if hasattr(value, "dtype") else type(value).__name__
        print(f"{key}: shape={shape}, dtype={dtype}")


if __name__ == "__main__":
    main()
