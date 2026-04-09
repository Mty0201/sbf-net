"""Smoke test: verify training automatically triggers test after completion.

Runs a 1-epoch, 1-batch training using a tiny stub model on sample data,
then verifies the trainer calls SemanticBoundaryTester with model_best.pth
and produces result/ predictions.
"""

from __future__ import annotations

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
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(pointcept_root))
    return repo_root, pointcept_root


def main():
    repo_root, _ = bootstrap_paths()

    import torch
    import torch.nn as nn
    from pointcept.models.builder import MODELS

    # ── 1. Tiny stub model ─────────────────────────────────────────────
    @MODELS.register_module("TrainTestSmokeModel")
    class TrainTestSmokeModel(nn.Module):
        def __init__(self, in_channels=6, num_classes=8):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes),
            )

        def forward(self, input_dict):
            return dict(seg_logits=self.mlp(input_dict["feat"]))

    # ── 2. Config: 1 epoch, 1 batch, sample data ──────────────────────
    samples_root = str(repo_root / "samples")
    tmp_dir = tempfile.mkdtemp(prefix="sbf_train_test_smoke_")

    cfg = dict(
        model=dict(type="TrainTestSmokeModel", in_channels=6, num_classes=8),
        data=dict(
            num_classes=8,
            ignore_index=-1,
            names=[
                "balustrade", "balcony", "advboard", "wall",
                "eave", "column", "window", "clutter",
            ],
            train_batch_size=1,
            val_batch_size=1,
            train=dict(
                type="BFDataset",
                split="training",
                data_root=samples_root,
                transform=[
                    dict(type="InjectIndexValidKeys", keys=("edge",)),
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="GridSample",
                        grid_size=0.06,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", point_max=4096, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "edge"),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
            ),
            val=dict(
                type="BFDataset",
                split="validation",
                data_root=samples_root,
                transform=[
                    dict(type="InjectIndexValidKeys", keys=("edge",)),
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="GridSample",
                        grid_size=0.06,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "edge"),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
            ),
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
        loss=dict(type="SemanticOnlyLoss"),
        evaluator=dict(type="SemanticEvaluator"),
        optimizer=dict(type="AdamW", lr=0.001, weight_decay=0.0),
        scheduler=None,
        param_dicts=None,
        seed=42,
        weight=None,
        resume=False,
        work_dir=tmp_dir,
        runtime=dict(
            log_freq=1,
            val_log_freq=1,
            grad_accum_steps=1,
            mix_prob=0.0,
            enable_amp=False,
        ),
        trainer=dict(
            total_epoch=1,
            eval_epoch=1,
            num_workers=0,
            max_train_batches=1,
            max_val_batches=1,
        ),
    )

    # ── 3. Run trainer (train -> val -> test) ──────────────────────────
    from project.trainer import SemanticBoundaryTrainer

    print("=" * 60)
    print("SMOKE TEST: train -> val -> test integration")
    print(f"  work_dir: {tmp_dir}")
    print("=" * 60)

    trainer = SemanticBoundaryTrainer(cfg)
    trainer.run()

    # ── 4. Verify test output ──────────────────────────────────────────
    result_dir = os.path.join(tmp_dir, "result")
    assert os.path.isdir(result_dir), f"result/ directory not created at {result_dir}"

    pred_files = [f for f in os.listdir(result_dir) if f.endswith("_pred.npy")]
    assert len(pred_files) > 0, "no prediction files in result/"

    best_ckpt = os.path.join(tmp_dir, "model", "model_best.pth")
    assert os.path.isfile(best_ckpt), "model_best.pth not saved"

    print("\n" + "=" * 60)
    print("SMOKE TEST RESULT")
    print(f"  model_best.pth: exists")
    print(f"  result/ pred files: {pred_files}")
    print("\nSMOKE TEST PASSED: train -> val -> test integration works")
    print("=" * 60)


if __name__ == "__main__":
    main()
