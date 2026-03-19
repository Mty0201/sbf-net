"""Smoke test for stage-1 validation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def bootstrap_paths() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    sys.path.insert(0, str(repo_root))
    return repo_root


def main():
    repo_root = bootstrap_paths()
    from project.evaluator.semantic_boundary_evaluator import SemanticBoundaryEvaluator

    sample_dir = repo_root / "samples" / "training" / "020101"
    segment = np.load(sample_dir / "segment.npy").reshape(-1).astype(np.int64)
    edge = np.load(sample_dir / "edge.npy").astype(np.float32)

    num_points = segment.shape[0]
    num_classes = 8

    torch.manual_seed(0)
    seg_logits = torch.randn(num_points, num_classes, dtype=torch.float32)
    edge_pred = torch.randn(num_points, 5, dtype=torch.float32)
    segment_t = torch.from_numpy(segment).long()
    edge_t = torch.from_numpy(edge).float()

    evaluator = SemanticBoundaryEvaluator()
    metrics = evaluator(
        seg_logits=seg_logits,
        edge_pred=edge_pred,
        segment=segment_t,
        edge=edge_t,
    )

    print(f"sample_path: {sample_dir}")
    print(f"N: {num_points}")
    for key in metrics.keys():
        value = metrics[key]
        scalar = float(value.detach().cpu())
        is_nan = bool(torch.isnan(value).any().item())
        print(f"{key}: {scalar:.6f}, is_nan={is_nan}")


if __name__ == "__main__":
    main()
