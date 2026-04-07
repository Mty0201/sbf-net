"""Minimal loss smoke tests for semantic boundary supervision."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def bootstrap_paths() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    sys.path.insert(0, str(repo_root))
    return repo_root


def describe_losses(tag: str, loss_dict: dict[str, torch.Tensor]):
    print(f"[{tag}]")
    for key, value in loss_dict.items():
        scalar = float(value.detach().cpu())
        is_nan = bool(torch.isnan(value).any().item())
        print(f"  {key}: {scalar:.6f}, is_nan={is_nan}")


def make_pseudo_case(num_points: int, num_classes: int, all_zero_valid: bool = False):
    torch.manual_seed(0)
    seg_logits = torch.randn(num_points, num_classes, dtype=torch.float32)
    edge_pred = torch.randn(num_points, 5, dtype=torch.float32)
    segment = torch.randint(0, num_classes, (num_points,), dtype=torch.long)
    edge = torch.zeros(num_points, 5, dtype=torch.float32)
    edge[:, 0:3] = torch.randn(num_points, 3, dtype=torch.float32) * 0.04

    if all_zero_valid:
        edge[:, 3] = 0.0
        edge[:, 4] = 0.0
    else:
        valid_count = max(1, num_points // 3)
        edge[:, 3] = 0.0
        edge[:, 4] = 0.0
        edge[:valid_count, 3] = 1.0
        edge[:valid_count, 4] = 1.0
    return seg_logits, edge_pred, segment, edge


def run_real_sample_case(loss_fn, sample_dir: Path, num_classes: int):
    segment = np.load(sample_dir / "segment.npy").reshape(-1)
    edge = np.load(sample_dir / "edge.npy").astype(np.float32)
    num_points = segment.shape[0]

    seg_logits = torch.randn(num_points, num_classes, dtype=torch.float32)
    edge_pred = torch.randn(num_points, 5, dtype=torch.float32)
    segment_t = torch.from_numpy(segment).long()
    edge_t = torch.from_numpy(edge).float()

    print(f"[real_sample] path={sample_dir}")
    print(f"  N={num_points}")
    loss_dict = loss_fn(seg_logits, edge_pred, segment_t, edge_t)
    describe_losses("real_sample_losses", loss_dict)


def main():
    repo_root = bootstrap_paths()
    from project.losses import SemanticBoundaryLoss

    loss_fn = SemanticBoundaryLoss()
    num_classes = 8

    seg_logits, edge_pred, segment, edge = make_pseudo_case(
        num_points=16, num_classes=num_classes, all_zero_valid=False
    )
    describe_losses(
        "pseudo_mixed_valid",
        loss_fn(seg_logits, edge_pred, segment, edge),
    )

    seg_logits, edge_pred, segment, edge = make_pseudo_case(
        num_points=16, num_classes=num_classes, all_zero_valid=True
    )
    describe_losses(
        "pseudo_all_zero_valid",
        loss_fn(seg_logits, edge_pred, segment, edge),
    )

    run_real_sample_case(
        loss_fn,
        repo_root / "samples" / "training" / "020101",
        num_classes=num_classes,
    )
    run_real_sample_case(
        loss_fn,
        repo_root / "samples" / "validation" / "010101",
        num_classes=num_classes,
    )


if __name__ == "__main__":
    main()
