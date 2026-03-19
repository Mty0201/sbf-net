"""Minimal validation-step smoke test on a single real sample."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


GRID_SIZE = 0.04
MAX_POINTS = 8192


def bootstrap_paths() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    pointcept_root = repo_root.parent
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(pointcept_root))
    return repo_root


def load_real_sample(sample_dir: Path):
    coord = np.load(sample_dir / "coord.npy").astype(np.float32)
    color = np.load(sample_dir / "color.npy").astype(np.float32)
    normal = np.load(sample_dir / "normal.npy").astype(np.float32)
    segment = np.load(sample_dir / "segment.npy").reshape(-1).astype(np.int64)
    edge = np.load(sample_dir / "edge.npy").astype(np.float32)

    original_n = coord.shape[0]
    used_n = min(original_n, MAX_POINTS)

    coord = coord[:used_n]
    color = color[:used_n] / 255.0
    normal = normal[:used_n]
    segment = segment[:used_n]
    edge = edge[:used_n]

    min_coord = coord.min(axis=0, keepdims=True)
    grid_coord = np.floor((coord - min_coord) / GRID_SIZE).astype(np.int64)
    feat = np.concatenate([color, normal], axis=1).astype(np.float32)

    return dict(
        sample_path=str(sample_dir),
        original_n=original_n,
        used_n=used_n,
        coord=torch.from_numpy(coord),
        grid_coord=torch.from_numpy(grid_coord),
        feat=torch.from_numpy(feat),
        offset=torch.tensor([used_n], dtype=torch.int64),
        segment=torch.from_numpy(segment),
        edge=torch.from_numpy(edge),
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def has_nan_dict(data: dict[str, torch.Tensor]) -> bool:
    for value in data.values():
        if isinstance(value, torch.Tensor) and torch.isnan(value).any():
            return True
    return False


def main():
    repo_root = bootstrap_paths()

    import project.models  # noqa: F401
    from project.evaluator import SemanticBoundaryEvaluator
    from project.losses import SemanticBoundaryLoss
    from pointcept.models import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_dir = repo_root / "samples" / "training" / "020101"
    model_cfg = runpy.run_path(
        str(
            repo_root
            / "configs"
            / "semantic_boundary"
            / "semseg-pt-v3m1-0-base-bf-edge-model.py"
        )
    )["model"]

    model = build_model(model_cfg).to(device).eval()
    loss_fn = SemanticBoundaryLoss().to(device)
    evaluator = SemanticBoundaryEvaluator()

    batch = move_batch_to_device(load_real_sample(sample_dir), device)

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
            in_channels=batch["feat"].shape[1],
            out_channels=model.semantic_head.proj.in_features,
        ).to(device)

    forward_input = {
        "coord": batch["coord"],
        "grid_coord": batch["grid_coord"],
        "feat": batch["feat"],
        "offset": batch["offset"],
    }

    with torch.no_grad():
        output = model(forward_input)
        loss_dict = loss_fn(
            seg_logits=output["seg_logits"],
            edge_pred=output["edge_pred"],
            segment=batch["segment"],
            edge=batch["edge"],
        )
        metric_dict = evaluator(
            seg_logits=output["seg_logits"],
            edge_pred=output["edge_pred"],
            segment=batch["segment"],
            edge=batch["edge"],
        )

    print("env: ptv3")
    print(f"device: {device}")
    print(f"smoke_mode: {smoke_mode}")
    print(f"sample_path: {batch['sample_path']}")
    print(f"original_N: {batch['original_n']}")
    print(f"used_N: {batch['used_n']}")
    print(f"seg_logits_shape: {tuple(output['seg_logits'].shape)}")
    print(f"edge_pred_shape: {tuple(output['edge_pred'].shape)}")

    for key in ["loss", "loss_semantic", "loss_mask", "loss_vec", "loss_strength"]:
        print(f"{key}: {float(loss_dict[key].detach().cpu()):.6f}")

    for key in [
        "val_mIoU",
        "val_mAcc",
        "val_allAcc",
        "val_loss_mask",
        "val_loss_vec",
        "val_loss_strength",
        "mask_precision",
        "mask_recall",
        "mask_f1",
        "vec_error_masked",
        "strength_error_masked",
    ]:
        print(f"{key}: {float(metric_dict[key].detach().cpu()):.6f}")

    print(f"loss_has_nan: {has_nan_dict(loss_dict)}")
    print(f"metric_has_nan: {has_nan_dict(metric_dict)}")


if __name__ == "__main__":
    main()
