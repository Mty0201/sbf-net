"""Smoke validation for CR-V (CR-Q architecture/data + continuous support-weighted semantic CE).

CR-V is a single-variable delta from CR-Q:
    loss.type:   DualSupervisionPureBFANetLoss  -> DualSupervisionSupportWeightedBFANetLoss
    work_dir:    .../pure_bfanet_v4_g04         -> .../support_weighted_v4_g04

Everything else (model, data, optimizer, scheduler, seed, epochs, batch, AMP)
is byte-identical to CR-Q. The only new field flowing through the pipeline
is `s_weight`, a per-point float32 weight in [0, 1] precomputed at
`s_weight_r060_r120.npy` alongside each chunk.

This smoke test verifies:
    1. Real dataloader emits `s_weight` with correct shape/dtype/range.
    2. Invariants vs boundary_mask: core subset (s>=1-eps) agrees with
       boundary_mask>0.5 set; mean s_weight > mean boundary_mask (buffer
       contributes).
    3. Batch collate preserves s_weight across variable-N samples.
    4. Model forward produces all 4 outputs (seg_logits_v1/v2, support_pred_v1/v2).
    5. New loss class accepts s_weight via _build_loss_inputs forwarding.
    6. Step-0 v1_loss == v2_loss exact (dual-stream identity init).
    7. Backward produces grads on all heads + fusion.out_proj.
    8. CR-V effective CE weight per batch is not >1.5× CR-Q (risk flag from plan).
    9. CUDA peak memory at bs=2 is reasonable (<6 GiB).

Usage:

    POINTCEPT_ROOT=/home/mty0201/Pointcept \
    SBF_DATA_ROOT=/home/mty0201/data/BF_edge_chunk_npy \
    python scripts/train/check_cr_v_smoke.py
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def bootstrap_paths() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    pointcept_root_env = os.environ.get("POINTCEPT_ROOT")
    if pointcept_root_env is None:
        raise RuntimeError("POINTCEPT_ROOT is required")
    pointcept_root = Path(pointcept_root_env).resolve()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(pointcept_root))
    return repo_root


def main() -> None:
    repo_root = bootstrap_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if os.environ.get("SBF_DATA_ROOT") is None:
        raise RuntimeError("SBF_DATA_ROOT is required")
    print(f"SBF_DATA_ROOT: {os.environ['SBF_DATA_ROOT']}")

    import project.datasets  # noqa: F401
    import project.models  # noqa: F401
    import project.transforms  # noqa: F401
    from pointcept.datasets import build_dataset
    from pointcept.datasets.utils import point_collate_fn
    from pointcept.models import build_model
    from project.losses import build_loss
    from project.trainer.trainer import SemanticBoundaryTrainer

    cfg_path = (
        repo_root
        / "configs"
        / "semantic_boundary"
        / "clean_reset_g04_s38873367"
        / "support_weighted_v4_g04_train.py"
    )
    print(f"Config: {cfg_path}")
    cfg = runpy.run_path(str(cfg_path))

    # --- [1] Sanity check CR-V config is CR-Q-shaped with the expected delta ---
    print("\n[1] CR-V config parameters")
    train_tf = cfg["data"]["train"]["transform"]
    train_grid = next(t for t in train_tf if t.get("type") == "GridSample")
    sphere_rate = next(
        t for t in train_tf
        if t.get("type") == "SphereCrop" and "sample_rate" in t
    )
    sphere_max = next(
        t for t in train_tf
        if t.get("type") == "SphereCrop" and "point_max" in t
    )
    assert train_grid["grid_size"] == 0.04, f"grid={train_grid['grid_size']}"
    assert sphere_rate["sample_rate"] == 0.4, f"rate={sphere_rate['sample_rate']}"
    assert sphere_max["point_max"] == 80000, f"max={sphere_max['point_max']}"
    assert cfg["data"]["train_batch_size"] == 2
    assert cfg["runtime"]["grad_accum_steps"] == 12
    eff = cfg["data"]["train_batch_size"] * cfg["runtime"]["grad_accum_steps"]
    assert eff == 24, f"effective batch = {eff}, expected 24"
    assert cfg["loss"]["type"] == "DualSupervisionSupportWeightedBFANetLoss", (
        f"loss.type = {cfg['loss']['type']}"
    )
    print(f"    grid=0.04 rate=0.4 max=80000 bs=2 accum=12 eff=24")
    print(f"    loss.type = {cfg['loss']['type']}")

    # --- [2] Build real dataset and pull one transformed sample ---
    print("\n[2] Build train dataset (real BF_edge_chunk_npy)")
    train_dataset = build_dataset(cfg["data"]["train"])
    print(f"    dataset size: {len(train_dataset)}")

    print("\n[3] Pull one sample through transform pipeline")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    sample = train_dataset[0]
    print(f"    keys: {sorted(sample.keys())}")
    for k in ("coord", "grid_coord", "segment", "edge", "boundary_mask", "s_weight", "feat"):
        assert k in sample, f"missing key {k}"
    n_pts = sample["coord"].shape[0]
    print(f"    N after transform: {n_pts}")
    assert n_pts <= 80000, f"post-SphereCrop N={n_pts} exceeds point_max"

    # --- s_weight field properties ---
    sw = sample["s_weight"]
    if isinstance(sw, torch.Tensor):
        sw_np = sw.detach().cpu().numpy()
    else:
        sw_np = np.asarray(sw)
    sw_np = sw_np.reshape(-1).astype(np.float32)
    assert sw_np.shape[0] == n_pts, f"s_weight N={sw_np.shape[0]} != coord N={n_pts}"
    assert sw_np.min() >= 0.0 - 1e-5 and sw_np.max() <= 1.0 + 1e-5, (
        f"s_weight out of [0,1]: [{sw_np.min()}, {sw_np.max()}]"
    )
    print(f"    s_weight range: [{sw_np.min():.4f}, {sw_np.max():.4f}]")

    bmask = sample["boundary_mask"]
    if isinstance(bmask, torch.Tensor):
        bmask_np = bmask.detach().cpu().numpy()
    else:
        bmask_np = np.asarray(bmask)
    bmask_np = bmask_np.reshape(-1).astype(np.float32)
    core_sw = sw_np >= 1.0 - 1e-4
    mask_pos = bmask_np > 0.5
    # Invariant: core ⊆ boundary_mask (allowing the tiny "d=exact-6cm" edge case)
    core_not_in_mask = int((core_sw & ~mask_pos).sum())
    mask_not_in_core = int((mask_pos & ~core_sw).sum())
    print(f"    core (s>=1): {int(core_sw.sum())} ({core_sw.mean()*100:.2f}%)")
    print(f"    boundary_mask:  {int(mask_pos.sum())} ({mask_pos.mean()*100:.2f}%)")
    print(f"    core ∧ ¬bmask (edge-case, <=few): {core_not_in_mask}")
    print(f"    bmask ∧ ¬core (must be 0):        {mask_not_in_core}")
    # Post-GridSample the tiny edge-case count can grow; relax to a small fraction
    assert core_not_in_mask <= max(5, int(0.001 * n_pts)), "core vs bmask drift too large"
    assert mask_not_in_core == 0, "boundary_mask has points outside core — invariant broken"
    # Invariant: mean s > mean bmask (buffer contributes mass)
    print(f"    mean s_weight: {sw_np.mean():.4f}  vs mean boundary_mask: {bmask_np.mean():.4f}")
    assert sw_np.mean() > bmask_np.mean(), "buffer not contributing mass"

    # --- [4] Collate a tiny batch (size 2 to match CR-V train_batch_size) ---
    print("\n[4] Collate batch of size 2")
    s0 = train_dataset[0]
    s1 = train_dataset[1]
    batch = point_collate_fn([s0, s1], mix_prob=0.0)
    total_pts = batch["coord"].shape[0]
    print(f"    batch total N: {total_pts}")
    print(f"    batch offset: {batch['offset'].tolist()}")
    assert "s_weight" in batch, "s_weight dropped during collate"
    assert "boundary_mask" in batch, "boundary_mask dropped during collate"

    bsw = batch["s_weight"]
    if isinstance(bsw, torch.Tensor):
        bsw_np = bsw.detach().cpu().numpy().reshape(-1).astype(np.float32)
    else:
        bsw_np = np.asarray(bsw).reshape(-1).astype(np.float32)
    print(f"    batch s_weight range: [{bsw_np.min():.4f}, {bsw_np.max():.4f}]")
    print(f"    batch s_weight mean:  {bsw_np.mean():.4f}")
    print(f"    batch core_frac:      {(bsw_np >= 1-1e-4).mean()*100:.2f}%")

    bm_np = (
        batch["boundary_mask"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        if isinstance(batch["boundary_mask"], torch.Tensor)
        else np.asarray(batch["boundary_mask"]).reshape(-1).astype(np.float32)
    )

    # --- CR-V vs CR-Q effective CE weight comparison ---
    # Risk from plan: if CR-V mean weight > 1.5× CR-Q mean weight, flag.
    boundary_ce_weight = 10.0
    w_crq = 1.0 + bm_np * (boundary_ce_weight - 1.0)
    w_crv = 1.0 + bsw_np * (boundary_ce_weight - 1.0)
    print(f"    CR-Q mean ce_weight (w/ bmask): {w_crq.mean():.4f}")
    print(f"    CR-V mean ce_weight (w/ s_w):   {w_crv.mean():.4f}")
    print(f"    ratio CR-V / CR-Q:              {w_crv.mean() / w_crq.mean():.4f}")
    assert w_crv.mean() / w_crq.mean() < 1.5, (
        f"CR-V mean ce_weight inflation {w_crv.mean() / w_crq.mean():.2f}× "
        f"exceeds 1.5× plan risk threshold"
    )

    # Move batch to device
    def _to_dev(v):
        return v.to(device) if isinstance(v, torch.Tensor) else v
    batch = {k: _to_dev(v) for k, v in batch.items()}

    # --- [5] Build model + loss, forward, loss, backward, optimizer step ---
    print("\n[5] Build CR-V model + loss and run 1 optimizer step")
    model = build_model(cfg["model"]).to(device).train()
    total_params = sum(p.numel() for p in model.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    print(f"    model params: {total_params:,} "
          f"(fusion {fusion_params:,}, {100*fusion_params/total_params:.2f}%)")

    loss_fn = build_loss(cfg["loss"]).to(device)
    print(f"    loss class: {loss_fn.__class__.__name__}")

    fwd_in = {
        "coord": batch["coord"],
        "grid_coord": batch["grid_coord"],
        "feat": batch["feat"],
        "offset": batch["offset"],
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("    forward...")
    output = model(fwd_in)
    for key in ("seg_logits", "seg_logits_v1", "seg_logits_v2",
                "support_pred", "support_pred_v1", "support_pred_v2"):
        assert key in output, f"model output missing {key}"
    print(f"    seg_logits shape: {tuple(output['seg_logits'].shape)}")
    print(f"    support_pred shape: {tuple(output['support_pred'].shape)}")

    print("    loss...")
    kwargs = SemanticBoundaryTrainer._build_loss_inputs(output, batch)
    for k in ("seg_logits", "support_pred", "seg_logits_v2", "support_pred_v2",
              "segment", "edge", "boundary_mask", "s_weight"):
        assert k in kwargs, f"trainer _build_loss_inputs did not forward {k}"
    loss_dict = loss_fn(**kwargs)
    loss_val = float(loss_dict["loss"].detach().float().cpu())
    print(f"    loss: {loss_val:.6f}")
    assert not np.isnan(loss_val), "loss is NaN"
    assert not np.isinf(loss_val), "loss is inf"

    # Step-0 dual-stream identity: v1_loss must equal v2_loss exactly
    if "v1_loss" in loss_dict and "v2_loss" in loss_dict:
        v1 = float(loss_dict["v1_loss"].detach().float().cpu())
        v2 = float(loss_dict["v2_loss"].detach().float().cpu())
        print(f"    v1_loss: {v1:.6f}   v2_loss: {v2:.6f}   |v1-v2|: {abs(v1-v2):.3e}")
        assert abs(v1 - v2) < 1e-5, f"step-0 v1 != v2 ({v1} vs {v2})"

    for k in ("v1_loss_semantic", "v1_loss_bce", "v1_loss_dice", "v2_loss_semantic"):
        if k in loss_dict:
            print(f"    {k}: {float(loss_dict[k].detach().float().cpu()):.6f}")

    print("    backward + optimizer step...")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss_dict["loss"].backward()

    def gn(p):
        return float(p.grad.detach().float().norm()) if p.grad is not None else 0.0
    sem_v1 = gn(model.semantic_head.proj.weight)
    sem_v2 = gn(model.semantic_head_v2.proj.weight)
    sup_v1 = gn(model.support_head.support_head.weight)
    sup_v2 = gn(model.support_head_v2.support_head.weight)
    fusion_out = gn(model.fusion.out_proj_sem.weight)
    print(f"    |grad sem_head_v1|     = {sem_v1:.3e}")
    print(f"    |grad sem_head_v2|     = {sem_v2:.3e}")
    print(f"    |grad sup_head_v1|     = {sup_v1:.3e}")
    print(f"    |grad sup_head_v2|     = {sup_v2:.3e}")
    print(f"    |grad fusion.out_proj| = {fusion_out:.3e}")
    assert sem_v1 > 0, "v1 semantic head got no gradient"
    assert sem_v2 > 0, "v2 semantic head got no gradient"
    assert sup_v1 > 0, "v1 support head got no gradient"
    assert sup_v2 > 0, "v2 support head got no gradient"
    assert fusion_out > 0, "fusion.out_proj got no gradient"

    optimizer.step()
    print("    optimizer step OK")

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\n    peak CUDA memory (bs=2, 1 step): {peak_mb:.0f} MiB")
        assert peak_mb < 6 * 1024, f"peak memory {peak_mb:.0f} MiB exceeds 6 GiB budget"

    print("\nALL CR-V SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
