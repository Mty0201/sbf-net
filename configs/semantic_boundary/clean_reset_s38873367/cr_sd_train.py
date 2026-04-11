"""CR-SD (grid=0.06): Subtractive Decoupling segmentor on g06 hyperparams.

Mirror of clean_reset_g04_s38873367/cr_sd_train.py, re-based onto the g06
workstream so the cheaper grid=0.06 pipeline can also run CR-SD. Model is
identical: DecoupledBFANetSegmentorV1 = PT-v3m1 backbone + SubtractiveDecoupling
module + seg_head (8 classes) + marg_head (binary boundary). Loss is computed
inside the segmentor — CE + multiclass Lovasz on semantics, BCE + binary Lovasz
on the margin head supervised by boundary_mask. V1 supervision only.

Training hyperparameters (optimiser, scheduler, seed, batch, grad_accum,
total_epoch, mix_prob, AMP) are byte-identical to the g06 semantic-only
baseline (semantic_only_train.py). Only `model`, `work_dir`, and docstring
differ from that baseline. Data pipeline is clean_reset_data.py (grid=0.06);
it already injects boundary_mask via InjectIndexValidKeys and carries it
through Collect. The `edge` key it also carries is unused by CR-SD and harmless.
"""

from __future__ import annotations

from pathlib import Path
import runpy


_dir = Path(__file__).resolve().parent
repo_root = _dir.parents[2]

model = dict(
    type="DecoupledBFANetSegmentorV1",
    num_classes=8,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
)

data = runpy.run_path(str(_dir / "clean_reset_data.py"))["data"]
data["train_batch_size"] = 4
data["val_batch_size"] = 1

loss = dict(type="CRSDLoss")
evaluator = dict(type="SemanticEvaluator")

optimizer = dict(
    type="AdamW",
    lr=0.006,
    weight_decay=0.05,
)

param_dicts = [dict(keyword="block", lr=0.0006)]

scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

seed = 38873367
weight = None
resume = False
work_dir = str(repo_root / "outputs" / "clean_reset_s38873367" / "cr_sd")

runtime = dict(
    log_freq=1,
    val_log_freq=1,
    save_freq=100,
    grad_accum_steps=3,
    mix_prob=0.8,
    enable_amp=True,
)

trainer = dict(
    total_epoch=2000,
    eval_epoch=100,
    num_workers=8,
    max_train_batches=None,
    max_val_batches=None,
)
