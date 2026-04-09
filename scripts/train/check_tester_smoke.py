"""Smoke test for SemanticBoundaryTester.

Creates a tiny stub model, saves a fake checkpoint, and runs fragment-based
test inference on the sample validation data to verify the full pipeline:
  config loading -> dataset (test_mode=True) -> fragment TTA -> tester loop -> metrics
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def bootstrap_paths(pointcept_root_arg: str | None = None) -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    pointcept_root_input = pointcept_root_arg or os.environ.get("POINTCEPT_ROOT")
    if pointcept_root_input is None:
        raise RuntimeError("POINTCEPT_ROOT or --pointcept-root is required.")
    pointcept_root = Path(pointcept_root_input).resolve()
    if not pointcept_root.exists():
        raise FileNotFoundError(f"Pointcept root not found: {pointcept_root}")
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(pointcept_root))
    return repo_root, pointcept_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointcept-root", default=None)
    args = parser.parse_args()

    repo_root, pointcept_root = bootstrap_paths(args.pointcept_root)

    import torch
    import torch.nn as nn
    from pointcept.models.builder import MODELS

    # ── 1. Register a tiny stub model ──────────────────────────────────
    @MODELS.register_module("SmokeTestModel")
    class SmokeTestModel(nn.Module):
        """2-layer MLP that outputs seg_logits. No real backbone."""

        def __init__(self, in_channels=6, num_classes=8):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes),
            )

        def forward(self, input_dict):
            feat = input_dict["feat"]
            return dict(seg_logits=self.mlp(feat))

    # ── 2. Build config pointing at samples/ ───────────────────────────
    samples_root = str(repo_root / "samples")

    cfg = dict(
        model=dict(type="SmokeTestModel", in_channels=6, num_classes=8),
        data=dict(
            num_classes=8,
            ignore_index=-1,
            names=[
                "balustrade", "balcony", "advboard", "wall",
                "eave", "column", "window", "clutter",
            ],
            test=dict(
                type="BFDataset",
                split="validation",
                data_root=samples_root,
                transform=[
                    dict(type="InjectIndexValidKeys", keys=("edge",)),
                    dict(type="CenterShift", apply_z=True),
                    dict(type="NormalizeColor"),
                ],
                test_mode=True,
                test_cfg=dict(
                    voxelize=dict(
                        type="GridSample",
                        grid_size=0.06,
                        hash_type="fnv",
                        mode="test",
                        return_grid_coord=True,
                    ),
                    crop=None,
                    post_transform=[
                        dict(type="CenterShift", apply_z=False),
                        dict(type="ToTensor"),
                        dict(
                            type="Collect",
                            keys=("coord", "grid_coord", "index"),
                            feat_keys=("color", "normal"),
                        ),
                    ],
                    aug_transform=[
                        [
                            dict(
                                type="RandomRotateTargetAngle",
                                angle=[0],
                                axis="z",
                                center=[0, 0, 0],
                                p=1,
                            )
                        ],
                    ],
                ),
            ),
        ),
        weight=None,  # filled below
        work_dir=None,  # filled below
        test_num_workers=0,
    )

    # ── 3. Save a fake checkpoint ──────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="sbf_tester_smoke_")
    cfg["work_dir"] = tmp_dir

    stub_model = SmokeTestModel(in_channels=6, num_classes=8)
    ckpt_path = os.path.join(tmp_dir, "smoke_model.pth")
    torch.save(
        dict(epoch=0, model_state_dict=stub_model.state_dict()),
        ckpt_path,
    )
    cfg["weight"] = ckpt_path

    # ── 4. Run the tester ──────────────────────────────────────────────
    from project.tester import SemanticBoundaryTester

    print("=" * 60)
    print("SMOKE TEST: SemanticBoundaryTester")
    print(f"  work_dir:  {tmp_dir}")
    print(f"  weight:    {ckpt_path}")
    print(f"  data_root: {samples_root}")
    print("=" * 60)

    tester = SemanticBoundaryTester(cfg)
    result = tester.test()

    # ── 5. Validate result ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SMOKE TEST RESULT")
    print(f"  mIoU:   {result['mIoU']:.4f}")
    print(f"  mAcc:   {result['mAcc']:.4f}")
    print(f"  allAcc: {result['allAcc']:.4f}")
    print(f"  iou_class shape: {result['iou_class'].shape}")
    print(f"  scenes tested: {len(result['record'])}")

    assert result["iou_class"].shape == (8,), "iou_class should have 8 entries"
    assert len(result["record"]) > 0, "should have tested at least 1 scene"
    assert 0.0 <= result["mIoU"] <= 1.0, "mIoU out of range"
    assert 0.0 <= result["mAcc"] <= 1.0, "mAcc out of range"
    assert 0.0 <= result["allAcc"] <= 1.0, "allAcc out of range"

    pred_files = [f for f in os.listdir(os.path.join(tmp_dir, "result")) if f.endswith("_pred.npy")]
    assert len(pred_files) > 0, "no prediction files saved"
    print(f"  pred files: {pred_files}")

    print("\nSMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
