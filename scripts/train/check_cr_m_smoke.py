"""Smoke validation for CR-M (cross-stream fusion + dual supervision).

Covers plan 05-02 smoke sub-tasks:
    6.1  Step-0 equivalence     — v1 and v2 predictions must match at init
    6.2  Gradient flow          — g v4 parameters get non-zero gradients after one backward
    6.3  Loss wrapper keys      — DualSupervisionBoundaryBinaryLoss emits v1_/v2_ keys
    6.4-lite  Trainer wiring    — SemanticBoundaryTrainer._build_loss_inputs forwards v2 keys
    6.5  CR-L no-regression     — run the CR-L loss on the same SharedBackboneSemanticSupportModel
                                  and confirm it still returns without error (additive trainer
                                  change must not break CR-L)

Usage::

    POINTCEPT_ROOT=/home/mty0201/Pointcept \
    python scripts/train/check_cr_m_smoke.py
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
            "POINTCEPT_ROOT is required; pass it via the environment."
        )
    pointcept_root = Path(pointcept_root_env).resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(pointcept_root))
    return repo_root


def load_real_sample(sample_dir: Path) -> dict:
    coord = np.load(sample_dir / "coord.npy").astype(np.float32)
    color = np.load(sample_dir / "color.npy").astype(np.float32)
    normal = np.load(sample_dir / "normal.npy").astype(np.float32)
    segment = np.load(sample_dir / "segment.npy").reshape(-1).astype(np.int64)
    edge = np.load(sample_dir / "edge.npy").astype(np.float32)

    use_n = min(coord.shape[0], MAX_POINTS)
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


def _forward_input(batch: dict) -> dict:
    return {
        "coord": batch["coord"],
        "grid_coord": batch["grid_coord"],
        "feat": batch["feat"],
        "offset": batch["offset"],
    }


def check_step0_equivalence(model, batch: dict) -> None:
    print("[6.1] Step-0 equivalence (v1 == v2)")
    model.eval()
    with torch.no_grad():
        output = model(_forward_input(batch))
    v1_seg = output["seg_logits_v1"]
    v2_seg = output["seg_logits_v2"]
    v1_sup = output["support_pred_v1"]
    v2_sup = output["support_pred_v2"]
    max_seg_diff = (v1_seg - v2_seg).abs().max().item()
    max_sup_diff = (v1_sup - v2_sup).abs().max().item()
    print(f"       max |seg_v1 - seg_v2| = {max_seg_diff:.3e}")
    print(f"       max |sup_v1 - sup_v2| = {max_sup_diff:.3e}")
    assert torch.allclose(v1_seg, v2_seg, atol=1e-5), (
        f"seg_logits_v1 vs v2 mismatch (max {max_seg_diff:.3e}) — "
        "zero-init or head clone broken."
    )
    assert torch.allclose(v1_sup, v2_sup, atol=1e-5), (
        f"support_pred_v1 vs v2 mismatch (max {max_sup_diff:.3e}) — "
        "zero-init or head clone broken."
    )
    # seg_logits alias must point at v1
    assert torch.allclose(output["seg_logits"], v1_seg), (
        "seg_logits alias must equal seg_logits_v1"
    )
    assert torch.allclose(output["support_pred"], v1_sup), (
        "support_pred alias must equal support_pred_v1"
    )
    print("       OK")


def check_gradient_flow(model, batch: dict) -> None:
    """g v4 gradient flow — two-step check.

    Zero-init of ``out_proj_sem/out_proj_bnd`` creates a chicken-and-egg at
    step 0: the output projection receives gradient on its own weight
    (``d_loss/d_W = sem_attn.T @ d_loss/d_sem_delta``) and bias, but because
    ``W == 0`` the upstream gradient ``d_loss/d_sem_attn = W.T @ d_loss/d_sem_delta``
    is also zero, so ``qkv_sem`` / ``fusion_q`` see no gradient on the first
    backward. This is mathematically correct, not a bug.

    After one optimizer step the output projections become non-zero, and
    from the second backward onward the full g v4 path receives gradient.
    This check verifies *both* stages.
    """
    print("[6.2] Gradient flow through g v4 (two-step check)")
    model.train()

    def grad_norm(p):
        return float(p.grad.detach().float().norm()) if p.grad is not None else 0.0

    # --- Step 1: only out_proj + v2 head should see gradient ---
    model.zero_grad(set_to_none=True)
    output = model(_forward_input(batch))
    loss = output["seg_logits_v2"].float().mean() + output["support_pred_v2"].float().mean()
    loss.backward()

    out_proj_s = grad_norm(model.fusion.out_proj_sem.weight)
    out_proj_b = grad_norm(model.fusion.out_proj_bnd.weight)
    v2_head = grad_norm(model.semantic_head_v2.proj.weight)
    v1_head = grad_norm(model.semantic_head.proj.weight)
    qkv_sem_n = grad_norm(model.fusion.qkv_sem.weight)

    print(f"       step1 |grad out_proj_sem|  = {out_proj_s:.3e}  (expect >0)")
    print(f"       step1 |grad out_proj_bnd|  = {out_proj_b:.3e}  (expect >0)")
    print(f"       step1 |grad v2 sem head|   = {v2_head:.3e}  (expect >0)")
    print(f"       step1 |grad v1 sem head|   = {v1_head:.3e}  (expect 0)")
    print(f"       step1 |grad qkv_sem|       = {qkv_sem_n:.3e}  (expect 0 — W_out == 0)")

    assert out_proj_s > 0, "out_proj_sem got no gradient — v2 path broken"
    assert out_proj_b > 0, "out_proj_bnd got no gradient — v2 path broken"
    assert v2_head > 0, "v2 semantic head got no gradient"
    assert v1_head == 0.0, (
        "v1 semantic head received gradient from v2-only loss — aliasing leak?"
    )
    assert qkv_sem_n == 0.0, (
        "qkv_sem got gradient at step 0, but out_proj weights are zero-init — "
        "unexpected path leakage?"
    )

    # --- One optimizer step so the output projections become non-zero ---
    optimizer = torch.optim.SGD(
        [model.fusion.out_proj_sem.weight,
         model.fusion.out_proj_sem.bias,
         model.fusion.out_proj_bnd.weight,
         model.fusion.out_proj_bnd.bias],
        lr=1.0,
    )
    optimizer.step()

    # --- Step 2: full path should now be gradient-live ---
    model.zero_grad(set_to_none=True)
    output = model(_forward_input(batch))
    loss = output["seg_logits_v2"].float().mean() + output["support_pred_v2"].float().mean()
    loss.backward()

    qkv_sem_n = grad_norm(model.fusion.qkv_sem.weight)
    qkv_bnd_n = grad_norm(model.fusion.qkv_bnd.weight)
    fusion_q0 = grad_norm(model.fusion.fusion_q[0].weight)
    fusion_q2 = grad_norm(model.fusion.fusion_q[2].weight)

    print(f"       step2 |grad qkv_sem|       = {qkv_sem_n:.3e}  (expect >0)")
    print(f"       step2 |grad qkv_bnd|       = {qkv_bnd_n:.3e}  (expect >0)")
    print(f"       step2 |grad fusion_q[0]|   = {fusion_q0:.3e}  (expect >0)")
    print(f"       step2 |grad fusion_q[2]|   = {fusion_q2:.3e}  (expect >0)")

    assert qkv_sem_n > 0, "qkv_sem still has no gradient after out_proj update"
    assert qkv_bnd_n > 0, "qkv_bnd still has no gradient after out_proj update"
    assert fusion_q0 > 0, "fusion_q[0] still has no gradient"
    assert fusion_q2 > 0, "fusion_q[2] still has no gradient"

    print("       OK")


def check_loss_wrapper_keys(device: torch.device) -> None:
    print("[6.3] DualSupervisionBoundaryBinaryLoss key prefixing")
    from project.losses import DualSupervisionBoundaryBinaryLoss

    loss_fn = DualSupervisionBoundaryBinaryLoss().to(device)

    N, C, K = 200, 8, 5
    torch.manual_seed(0)
    seg_logits = torch.randn(N, C, device=device)
    support_pred = torch.randn(N, 1, device=device)
    seg_logits_v2 = torch.randn(N, C, device=device)
    support_pred_v2 = torch.randn(N, 1, device=device)
    segment = torch.randint(0, C, (N,), device=device)
    edge = torch.rand(N, K, device=device)

    out = loss_fn(
        seg_logits=seg_logits,
        support_pred=support_pred,
        segment=segment,
        edge=edge,
        seg_logits_v2=seg_logits_v2,
        support_pred_v2=support_pred_v2,
    )

    required = {
        "loss",
        "v1_loss",
        "v2_loss",
        "v1_loss_semantic",
        "v1_loss_bce",
        "v1_loss_dice",
        "v1_dice_score",
        "v2_loss_semantic",
        "v2_loss_bce",
        "v2_loss_dice",
        "v2_dice_score",
    }
    missing = required - set(out.keys())
    assert not missing, f"Missing keys: {sorted(missing)}"

    total_check = out["loss"]
    summed = out["v1_loss"] + out["v2_loss"]
    diff = (total_check - summed).abs().item()
    print(f"       |loss - (v1_loss + v2_loss)| = {diff:.3e}")
    assert diff < 1e-5, "loss must equal v1_loss + v2_loss"

    print("       OK")

    # Also check that missing v2 inputs raises a helpful error.
    raised = False
    try:
        loss_fn(
            seg_logits=seg_logits,
            support_pred=support_pred,
            segment=segment,
            edge=edge,
        )
    except RuntimeError as e:
        raised = True
        assert "seg_logits_v2" in str(e)
    assert raised, (
        "DualSupervisionBoundaryBinaryLoss should raise when v2 inputs are missing"
    )


def check_trainer_wiring(model, batch: dict, device: torch.device) -> None:
    print("[6.4-lite] Trainer _build_loss_inputs forwards v2 keys")
    from project.trainer.trainer import SemanticBoundaryTrainer
    from project.losses import DualSupervisionBoundaryBinaryLoss

    model.eval()
    with torch.no_grad():
        output = model(_forward_input(batch))

    kwargs = SemanticBoundaryTrainer._build_loss_inputs(output, batch)
    for needed in ("seg_logits", "support_pred", "seg_logits_v2", "support_pred_v2", "edge"):
        assert needed in kwargs, f"_build_loss_inputs missing {needed}"

    loss_fn = DualSupervisionBoundaryBinaryLoss().to(device)
    out = loss_fn(**kwargs)
    print(f"       loss      = {float(out['loss']):.6f}")
    print(f"       v1_loss   = {float(out['v1_loss']):.6f}")
    print(f"       v2_loss   = {float(out['v2_loss']):.6f}")
    print(f"       v1_dice   = {float(out['v1_dice_score']):.6f}")
    print(f"       v2_dice   = {float(out['v2_dice_score']):.6f}")
    # Sanity: computed total must equal v1+v2 (independent of step-0 equivalence,
    # which is already checked against a fresh model in 6.1).
    summed = float(out["v1_loss"] + out["v2_loss"])
    assert abs(float(out["loss"]) - summed) < 1e-4, "loss != v1_loss + v2_loss"
    assert not torch.isnan(out["loss"]), "loss is NaN"
    print("       OK")


def check_cr_l_regression(repo_root: Path, batch: dict, device: torch.device) -> None:
    print("[6.5] CR-L no-regression (BoundaryBinaryLoss on SharedBackboneSemanticSupportModel)")
    import project.models  # noqa: F401
    from pointcept.models import build_model
    from project.losses import build_loss
    from project.trainer.trainer import SemanticBoundaryTrainer

    model_cfg_path = (
        repo_root
        / "configs"
        / "semantic_boundary"
        / "clean_reset_s38873367"
        / "clean_reset_support_model.py"
    )
    model_cfg = runpy.run_path(str(model_cfg_path))["model"]
    model = build_model(model_cfg).to(device).train()

    loss_fn = build_loss(
        dict(
            type="BoundaryBinaryLoss",
            aux_weight=0.3,
            boundary_ce_weight=10.0,
            sample_weight_scale=9.0,
            boundary_threshold=0.5,
            pos_weight=1.0,
        )
    ).to(device)

    output = model(_forward_input(batch))
    kwargs = SemanticBoundaryTrainer._build_loss_inputs(output, batch)
    # seg_logits_v2 / support_pred_v2 should NOT be present (no v2 output)
    assert "seg_logits_v2" not in kwargs
    assert "support_pred_v2" not in kwargs

    out = loss_fn(**kwargs)
    total = float(out["loss"])
    print(f"       CR-L loss = {total:.6f}")
    assert total == total, "CR-L loss is NaN"  # nan check
    # Backward also works
    total_tensor = out["loss"]
    total_tensor.backward()
    print("       OK")


def main():
    repo_root = bootstrap_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import project.models  # noqa: F401
    from pointcept.models import build_model

    model_cfg_path = (
        repo_root
        / "configs"
        / "semantic_boundary"
        / "clean_reset_s38873367"
        / "clean_reset_gated_v4_model.py"
    )
    model_cfg = runpy.run_path(str(model_cfg_path))["model"]
    print(f"Building model: {model_cfg['type']}")
    model = build_model(model_cfg).to(device)

    # Parameter count sanity check
    total_params = sum(p.numel() for p in model.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    print(f"Total params:  {total_params:,}")
    print(f"Fusion params: {fusion_params:,}  ({100*fusion_params/total_params:.2f}%)")

    sample_dir = repo_root / "samples" / "training" / "020101"
    batch = load_real_sample(sample_dir)
    batch = move_to_device(batch, device)
    print(f"Sample points: {batch['coord'].shape[0]}")

    check_step0_equivalence(model, batch)
    check_gradient_flow(model, batch)
    check_loss_wrapper_keys(device)
    check_trainer_wiring(model, batch, device)
    check_cr_l_regression(repo_root, batch, device)

    print("\nALL CR-M SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
