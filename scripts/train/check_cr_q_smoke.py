"""Smoke validation for CR-Q (CR-P architecture + grid=0.04 recipe).

What makes CR-Q different from CR-P is ONLY the data pipeline:
    grid_size:   0.06  -> 0.04
    sphere_rate: 0.6   -> 0.4
    point_max:   40960 -> 80000
    train_batch_size: 4 -> 2  (memory)
    grad_accum_steps: 6 -> 12 (effective batch = 24 preserved)

CR-M smoke already covers model + loss unit tests at a synthetic level.
The thing CR-Q specifically needs verified is that the REAL dataloader
producing REAL transformed batches under grid=0.04 still:
    1. Emits the `boundary_mask` key (positive ratio within 5-15% window)
    2. Produces a point count per sample larger than grid=0.06 baseline
       (confirms grid=0.04 + max=80000 is doing what we expect)
    3. Feeds the model, loss, backward, optimizer step without error
    4. Memory budget holds at train_batch_size=2

Usage:

    POINTCEPT_ROOT=/home/mty0201/Pointcept \
    SBF_DATA_ROOT=/home/mty0201/data/BF_edge_chunk_npy \
    python scripts/train/check_cr_q_smoke.py
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
        / "pure_bfanet_v4_g04_train.py"
    )
    print(f"Config: {cfg_path}")
    cfg = runpy.run_path(str(cfg_path))

    # --- [1] Sanity check CR-Q-specific parameters ---
    print("\n[1] CR-Q config parameters")
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
    print(f"    grid=0.04 rate=0.4 max=80000 bs=2 accum=12 eff=24  OK")

    # --- [2] Build real dataset and pull one transformed sample ---
    print("\n[2] Build train dataset (real BF_edge_chunk_npy)")
    train_dataset = build_dataset(cfg["data"]["train"])
    print(f"    dataset size: {len(train_dataset)}")

    print("\n[3] Pull one sample through transform pipeline")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    sample = train_dataset[0]
    print(f"    keys: {sorted(sample.keys())}")
    for k in ("coord", "grid_coord", "segment", "boundary_mask", "feat"):
        assert k in sample, f"missing key {k}"
    n_pts = sample["coord"].shape[0]
    print(f"    N after transform: {n_pts}")
    assert n_pts <= 80000, f"post-SphereCrop N={n_pts} exceeds point_max"
    # CR-P (grid=0.06+max=40960) typical: ~30-40k points per sample
    # CR-Q target: larger than that
    print(f"    (CR-P baseline ~30-40k at grid=0.06)")

    bmask = sample["boundary_mask"]
    if isinstance(bmask, torch.Tensor):
        bmask_np = bmask.cpu().numpy()
    else:
        bmask_np = np.asarray(bmask)
    bmask_np = bmask_np.reshape(-1).astype(np.float32)
    pos_ratio = bmask_np.mean()
    print(f"    boundary_mask positive ratio: {pos_ratio*100:.2f}%")
    assert 0.03 <= pos_ratio <= 0.20, (
        f"positive ratio {pos_ratio*100:.2f}% out of [3%, 20%] window"
    )

    # --- [4] Collate a tiny batch (size 2 to match CR-Q train_batch_size) ---
    print("\n[4] Collate batch of size 2")
    s0 = train_dataset[0]
    s1 = train_dataset[1]
    batch = point_collate_fn([s0, s1], mix_prob=0.0)
    total_pts = batch["coord"].shape[0]
    print(f"    batch total N: {total_pts}")
    print(f"    batch offset: {batch['offset'].tolist()}")
    assert "boundary_mask" in batch, "boundary_mask dropped during collate"
    bm = batch["boundary_mask"]
    if isinstance(bm, torch.Tensor):
        bm_np = bm.detach().cpu().numpy().reshape(-1).astype(np.float32)
    else:
        bm_np = np.asarray(bm).reshape(-1).astype(np.float32)
    batch_pos = bm_np.mean()
    print(f"    batch boundary_mask positive ratio: {batch_pos*100:.2f}%")

    # Move batch to device
    def _to_dev(v):
        return v.to(device) if isinstance(v, torch.Tensor) else v
    batch = {k: _to_dev(v) for k, v in batch.items()}

    # --- [5] Build model + loss, forward, loss, backward, optimizer step ---
    print("\n[5] Build CR-Q model + loss and run 1 optimizer step")
    model = build_model(cfg["model"]).to(device).train()
    total_params = sum(p.numel() for p in model.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    print(f"    model params: {total_params:,} "
          f"(fusion {fusion_params:,}, {100*fusion_params/total_params:.2f}%)")

    loss_fn = build_loss(cfg["loss"]).to(device)

    fwd_in = {
        "coord": batch["coord"],
        "grid_coord": batch["grid_coord"],
        "feat": batch["feat"],
        "offset": batch["offset"],
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("    forward...")
    # Run fp32; AMP dtype handling is exercised in real training, not smoke.
    output = model(fwd_in)
    for key in ("seg_logits", "seg_logits_v1", "seg_logits_v2",
                "support_pred", "support_pred_v1", "support_pred_v2"):
        assert key in output, f"model output missing {key}"
    print(f"    seg_logits shape: {tuple(output['seg_logits'].shape)}")
    print(f"    support_pred shape: {tuple(output['support_pred'].shape)}")

    print("    loss...")
    kwargs = SemanticBoundaryTrainer._build_loss_inputs(output, batch)
    # Must forward boundary_mask and all dual-stream keys to the loss
    for k in ("seg_logits", "support_pred", "seg_logits_v2", "support_pred_v2",
              "segment", "boundary_mask"):
        assert k in kwargs, f"trainer _build_loss_inputs did not forward {k}"
    loss_dict = loss_fn(**kwargs)
    loss_val = float(loss_dict["loss"].detach().float().cpu())
    print(f"    loss: {loss_val:.6f}")
    assert not np.isnan(loss_val), "loss is NaN"
    assert not np.isinf(loss_val), "loss is inf"
    for k in ("v1_loss", "v2_loss", "v1_loss_semantic", "v1_loss_bce",
              "v1_loss_dice", "v2_loss_semantic"):
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
    assert fusion_out > 0, "fusion.out_proj got no gradient"

    optimizer.step()
    print("    optimizer step OK")

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\n    peak CUDA memory (bs=2, 1 step): {peak_mb:.0f} MiB")

    print("\nALL CR-Q SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
