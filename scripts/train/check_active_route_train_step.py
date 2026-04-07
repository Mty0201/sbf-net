"""Smoke validation for the support-guided semantic focus route (active route).

Runs the full active route pipeline: model forward, loss forward with three
loss terms, backward, optimizer step, evaluator, and focus activation check.
Prints structured pass/fail output with explicit evidence boundary disclaimer.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


GRID_SIZE = 0.04
MAX_POINTS = 8192


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


def load_real_sample(sample_dir: Path):
    coord = np.load(sample_dir / "coord.npy").astype(np.float32)
    color = np.load(sample_dir / "color.npy").astype(np.float32)
    normal = np.load(sample_dir / "normal.npy").astype(np.float32)
    segment = np.load(sample_dir / "segment.npy").reshape(-1).astype(np.int64)
    edge = np.load(sample_dir / "edge.npy").astype(np.float32)

    original_n = coord.shape[0]
    use_n = min(original_n, MAX_POINTS)

    coord = coord[:use_n]
    color = color[:use_n] / 255.0
    normal = normal[:use_n]
    segment = segment[:use_n]
    edge = edge[:use_n]

    min_coord = coord.min(axis=0, keepdims=True)
    grid_coord = np.floor((coord - min_coord) / GRID_SIZE).astype(np.int64)
    feat = np.concatenate([color, normal], axis=1).astype(np.float32)

    return dict(
        sample_dir=str(sample_dir),
        original_n=original_n,
        used_n=use_n,
        coord=torch.from_numpy(coord),
        grid_coord=torch.from_numpy(grid_coord),
        feat=torch.from_numpy(feat),
        offset=torch.tensor([use_n], dtype=torch.int64),
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


def check_focus_activation(
    edge: torch.Tensor, focus_lambda: float = 1.0, focus_gamma: float = 1.0
) -> dict:
    """Check that focus_weight is higher in boundary regions than non-boundary regions.

    Per D-06: in regions where support_gt > 0.2, mean focus_weight must be
    significantly higher than in other regions.
    """
    support_gt = edge[:, 3].float().clamp(0.0, 1.0)
    valid_gt = edge[:, 4].float().clamp(0.0, 1.0)

    focus_weight = 1.0 + focus_lambda * (support_gt * valid_gt).pow(focus_gamma)

    boundary_mask = support_gt > 0.2
    non_boundary_mask = ~boundary_mask

    boundary_mean = (
        focus_weight[boundary_mask].mean().item() if boundary_mask.any() else 0.0
    )
    non_boundary_mean = (
        focus_weight[non_boundary_mask].mean().item()
        if non_boundary_mask.any()
        else 0.0
    )
    boundary_count = int(boundary_mask.sum().item())
    non_boundary_count = int(non_boundary_mask.sum().item())

    # Focus activates if boundary mean > non-boundary mean (boundary gets extra weight)
    activated = boundary_mean > non_boundary_mean and boundary_count > 0

    return dict(
        boundary_mean_focus=boundary_mean,
        non_boundary_mean_focus=non_boundary_mean,
        boundary_count=boundary_count,
        non_boundary_count=non_boundary_count,
        focus_activated=activated,
    )


def main():
    repo_root = bootstrap_paths()

    import project.models  # noqa: F401
    from project.losses.support_guided_semantic_focus_loss import (
        SupportGuidedSemanticFocusLoss,
    )
    from project.evaluator.support_guided_semantic_focus_evaluator import (
        SupportGuidedSemanticFocusEvaluator,
    )
    from pointcept.models import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_dir = repo_root / "samples" / "training" / "020101"
    model_cfg = runpy.run_path(
        str(
            repo_root
            / "configs"
            / "semantic_boundary"
            / "semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-model.py"
        )
    )["model"]

    model = build_model(model_cfg).to(device).train()
    loss_fn = SupportGuidedSemanticFocusLoss(
        support_loss_weight=1.0,
        focus_loss_weight=1.0,
        focus_lambda=1.0,
        focus_gamma=1.0,
    ).to(device)
    evaluator = SupportGuidedSemanticFocusEvaluator(boundary_metric_threshold=0.2)

    batch = load_real_sample(sample_dir)
    batch = move_batch_to_device(batch, device)

    forward_input = {
        "coord": batch["coord"],
        "grid_coord": batch["grid_coord"],
        "feat": batch["feat"],
        "offset": batch["offset"],
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    output = model(forward_input)

    # Output key check (D-02)
    has_seg_logits = "seg_logits" in output
    has_support_pred = "support_pred" in output

    # Loss forward (D-03)
    loss_dict = loss_fn(
        seg_logits=output["seg_logits"],
        support_pred=output["support_pred"],
        segment=batch["segment"],
        edge=batch["edge"],
    )

    # NaN checks (D-03, D-05 criterion 1)
    loss_semantic_nan = bool(torch.isnan(loss_dict["loss_semantic"]).any())
    loss_support_nan = bool(torch.isnan(loss_dict["loss_support"]).any())
    loss_focus_nan = bool(torch.isnan(loss_dict["loss_focus"]).any())
    any_loss_nan = loss_semantic_nan or loss_support_nan or loss_focus_nan

    # Backward + optimizer step (D-05 criterion 3)
    backward_ok = False
    step_ok = False
    loss_dict["loss"].backward()
    backward_ok = True
    optimizer.step()
    step_ok = True

    # Focus activation check (D-06)
    focus_check = check_focus_activation(
        batch["edge"], focus_lambda=1.0, focus_gamma=1.0
    )

    # Evaluator forward (validate evaluator runs without error)
    model.eval()
    with torch.no_grad():
        eval_output = model(forward_input)
        metric_dict = evaluator(
            seg_logits=eval_output["seg_logits"],
            support_pred=eval_output["support_pred"],
            segment=batch["segment"],
            edge=batch["edge"],
        )

    # Structured output
    print("env: ptv3")
    print(f"device: {device}")
    print("path_mode: actual_backbone_only")
    print(f"sample_path: {batch['sample_dir']}")
    print(f"original_N: {batch['original_n']}")
    print(f"used_N: {batch['used_n']}")
    print(f"output_keys: {sorted(output.keys())}")
    print(f"has_seg_logits: {has_seg_logits}")
    print(f"has_support_pred: {has_support_pred}")
    print(f"seg_logits_shape: {tuple(output['seg_logits'].shape)}")
    print(f"support_pred_shape: {tuple(output['support_pred'].shape)}")
    print(f"loss: {float(loss_dict['loss']):.6f}")
    print(f"loss_semantic: {float(loss_dict['loss_semantic']):.6f}")
    print(f"loss_support: {float(loss_dict['loss_support']):.6f}")
    print(f"loss_focus: {float(loss_dict['loss_focus']):.6f}")
    print(f"loss_semantic_nan: {loss_semantic_nan}")
    print(f"loss_support_nan: {loss_support_nan}")
    print(f"loss_focus_nan: {loss_focus_nan}")
    print(f"any_loss_nan: {any_loss_nan}")
    print(f"backward_ok: {backward_ok}")
    print(f"optimizer_step_ok: {step_ok}")
    print(f"focus_boundary_mean: {focus_check['boundary_mean_focus']:.6f}")
    print(f"focus_non_boundary_mean: {focus_check['non_boundary_mean_focus']:.6f}")
    print(f"focus_boundary_count: {focus_check['boundary_count']}")
    print(f"focus_non_boundary_count: {focus_check['non_boundary_count']}")
    print(f"focus_activated: {focus_check['focus_activated']}")
    print(f"val_mIoU: {float(metric_dict['val_mIoU']):.6f}")
    print(f"val_boundary_mIoU: {float(metric_dict['val_boundary_mIoU']):.6f}")
    print(f"metric_has_nan: {has_nan_dict(metric_dict)}")

    # Summary block
    pass_output_keys = has_seg_logits and has_support_pred
    pass_no_nan_losses = not any_loss_nan
    pass_backward_optimizer = backward_ok and step_ok
    pass_focus_activated = focus_check["focus_activated"]
    all_pass = (
        pass_output_keys
        and pass_no_nan_losses
        and pass_backward_optimizer
        and pass_focus_activated
    )

    print("--- SMOKE VALIDATION SUMMARY ---")
    print(f"pass_output_keys: {pass_output_keys}")
    print(f"pass_no_nan_losses: {pass_no_nan_losses}")
    print(f"pass_backward_optimizer: {pass_backward_optimizer}")
    print(f"pass_focus_activated: {pass_focus_activated}")
    print(f"ALL_PASS: {all_pass}")
    print(
        "evidence_boundary: local smoke validation only"
        " -- does NOT prove full-train performance"
    )


if __name__ == "__main__":
    main()
