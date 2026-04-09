"""Smoke validation for clean-reset CR-A and CR-B pipelines.

Verifies that both configs can:
1. Construct the trainer (model, loss, evaluator)
2. Load a dataset batch with correct shapes
3. Forward + loss + backward + optimizer step
4. Run evaluator without error

Uses a real sample from samples/ and bypasses the full DataLoader to test
the model/loss/evaluator wiring directly.
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
        coord=torch.from_numpy(coord),
        grid_coord=torch.from_numpy(grid_coord),
        feat=torch.from_numpy(feat),
        offset=torch.tensor([use_n], dtype=torch.int64),
        segment=torch.from_numpy(segment),
        edge=torch.from_numpy(edge),
    )


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def check_pipeline(name: str, model_cfg_path: Path, loss_cfg: dict, evaluator_cfg: dict,
                    repo_root: Path, device: torch.device):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    import project.models  # noqa: F401
    from project.losses import build_loss
    from project.evaluator import build_evaluator
    from pointcept.models import build_model

    model_cfg = runpy.run_path(str(model_cfg_path))["model"]

    # Step 1: Build model, loss, evaluator
    print("[1] Building model/loss/evaluator...")
    model = build_model(model_cfg).to(device).train()
    loss_fn = build_loss(loss_cfg).to(device)
    evaluator = build_evaluator(evaluator_cfg)
    print("    OK: all constructed without error")

    # Step 2: Load sample and check shapes
    print("[2] Loading sample...")
    sample_dir = repo_root / "samples" / "training" / "020101"
    batch = load_real_sample(sample_dir)
    batch = move_to_device(batch, device)
    n_points = batch["coord"].shape[0]
    print(f"    points: {n_points}")
    print(f"    edge shape: {tuple(batch['edge'].shape)}")
    print(f"    segment shape: {tuple(batch['segment'].shape)}")
    assert batch["edge"].shape[0] == n_points, "edge/coord shape mismatch!"
    assert batch["segment"].shape[0] == n_points, "segment/coord shape mismatch!"
    print("    OK: all shapes consistent")

    # Step 3: Forward
    print("[3] Model forward...")
    forward_input = {
        "coord": batch["coord"],
        "grid_coord": batch["grid_coord"],
        "feat": batch["feat"],
        "offset": batch["offset"],
    }
    output = model(forward_input)
    output_keys = sorted(output.keys())
    print(f"    output keys: {output_keys}")
    print(f"    seg_logits shape: {tuple(output['seg_logits'].shape)}")
    if "support_pred" in output:
        print(f"    support_pred shape: {tuple(output['support_pred'].shape)}")
    if "offset_pred" in output:
        print(f"    offset_pred shape: {tuple(output['offset_pred'].shape)}")

    # Step 4: Loss
    print("[4] Loss forward...")
    loss_kwargs = dict(seg_logits=output["seg_logits"], segment=batch["segment"])
    if "support_pred" in output and "edge" in batch:
        loss_kwargs["support_pred"] = output["support_pred"]
        loss_kwargs["edge"] = batch["edge"]
        if "offset_pred" in output:
            loss_kwargs["offset_pred"] = output["offset_pred"]
    elif "offset_pred" in output and "edge" in batch:
        loss_kwargs["offset_pred"] = output["offset_pred"]
        loss_kwargs["edge"] = batch["edge"]
    loss_dict = loss_fn(**loss_kwargs)
    loss_val = float(loss_dict["loss"].detach().cpu())
    print(f"    loss: {loss_val:.6f}")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1 and k != "loss":
            print(f"    {k}: {float(v.detach().cpu()):.6f}")
    assert not torch.isnan(loss_dict["loss"]), "loss is NaN!"
    print("    OK: no NaN")

    # Step 5: Backward + optimizer step
    print("[5] Backward + optimizer step...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss_dict["loss"].backward()
    optimizer.step()
    print("    OK: backward + step completed")

    # Step 6: Evaluator
    print("[6] Evaluator...")
    model.eval()
    with torch.no_grad():
        eval_output = model(forward_input)
        eval_kwargs = dict(seg_logits=eval_output["seg_logits"], segment=batch["segment"])
        if "support_pred" in eval_output and "edge" in batch:
            eval_kwargs["support_pred"] = eval_output["support_pred"]
            eval_kwargs["edge"] = batch["edge"]
            if "offset_pred" in eval_output:
                eval_kwargs["offset_pred"] = eval_output["offset_pred"]
        elif "offset_pred" in eval_output and "edge" in batch:
            eval_kwargs["offset_pred"] = eval_output["offset_pred"]
            eval_kwargs["edge"] = batch["edge"]
        metric_dict = evaluator(**eval_kwargs)
    print(f"    val_mIoU: {float(metric_dict['val_mIoU']):.6f}")
    if "val_boundary_mIoU" in metric_dict:
        print(f"    val_boundary_mIoU: {float(metric_dict['val_boundary_mIoU']):.6f}")
    if "support_reg_error" in metric_dict:
        print(f"    support_reg_error: {float(metric_dict['support_reg_error']):.6f}")
    if "support_cover" in metric_dict:
        print(f"    support_cover: {float(metric_dict['support_cover']):.6f}")
    print("    OK: evaluator completed")

    print(f"\n  PASS: {name}")
    return True


def main():
    repo_root = bootstrap_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_dir = repo_root / "configs" / "semantic_boundary" / "clean_reset_s38873367"

    pipelines = [
        (
            "CR-A (semantic-only)",
            config_dir / "clean_reset_semantic_model.py",
            dict(type="SemanticOnlyLoss"),
            dict(type="SemanticEvaluator"),
        ),
        (
            "CR-B (support-only)",
            config_dir / "clean_reset_support_model.py",
            dict(
                type="RedesignedSupportFocusLoss",
                support_reg_weight=1.0,
                support_cover_weight=0.20,
                support_tversky_alpha=0.3,
                support_tversky_beta=0.7,
                focus_mode="none",
            ),
            dict(type="RedesignedSupportFocusEvaluator"),
        ),
        (
            "CR-C (proximity-cue)",
            config_dir / "clean_reset_support_model.py",
            dict(type="BoundaryProximityCueLoss", aux_weight=0.3),
            dict(type="RedesignedSupportFocusEvaluator"),
        ),
        (
            "CR-D (serial-derivation)",
            config_dir / "clean_reset_serial_derivation_model.py",
            dict(type="SerialDerivationLoss", aux_weight=0.3, offset_weight=1.0),
            dict(type="RedesignedSupportFocusEvaluator"),
        ),
        (
            "CR-E (serial-derivation-only)",
            config_dir / "clean_reset_serial_derivation_only_model.py",
            dict(type="SerialDerivationOnlyLoss", offset_weight=1.0),
            dict(type="SemanticEvaluator"),
        ),
        (
            "CR-F (unweighted-boundary-cue)",
            config_dir / "clean_reset_support_model.py",
            dict(type="UnweightedBoundaryCueLoss", aux_weight=0.3),
            dict(type="RedesignedSupportFocusEvaluator"),
        ),
        (
            "CR-G (soft-boundary)",
            config_dir / "clean_reset_support_model.py",
            dict(type="SoftBoundaryLoss", aux_weight=0.3),
            dict(type="RedesignedSupportFocusEvaluator"),
        ),
    ]

    results = {}
    for name, model_path, loss_cfg, eval_cfg in pipelines:
        try:
            ok = check_pipeline(name, model_path, loss_cfg, eval_cfg, repo_root, device)
            results[name] = ok
        except Exception as e:
            print(f"\n  FAIL: {name} — {type(e).__name__}: {e}")
            results[name] = False

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
        if not ok:
            all_pass = False
    print(f"\n  ALL_PASS: {all_pass}")
    print("  evidence_boundary: local smoke validation only"
          " -- does NOT prove full-train performance")


if __name__ == "__main__":
    main()
