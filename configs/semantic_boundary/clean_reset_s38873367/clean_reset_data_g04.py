"""CR-Q data config: grid=0.04 + sphere_rate=0.4 + point_max=80000.

Derived from clean_reset_data.py. The ONLY differences are the three
downsampling parameters, applied consistently to train/val/test:

  grid_size:   0.06 -> 0.04   (train/val/test)
  sphere_rate: 0.6  -> 0.4    (train only)
  point_max:   40960 -> 80000 (train only)

Rationale: real training transform measurements on BF_edge_chunk_npy
(see .planning/phases/05-boundary-proximity-cue-experiment/cr_q_grid_rasterization_analysis.md)
show that grid=0.06 quantises the boundary band to 6 cm and leaves only
6.75% of non-boundary voxels within 6 cm of a boundary voxel, while
grid=0.04 + rate=0.4 + max=80000 raises that to 10.28% without inflating
the raw boundary ratio or polluting class interiors. Widening r is
rejected as an alternative — it scales wall coverage from 15% to 47%
at r=0.12 and does not meaningfully increase the non-boundary transition
layer density.

Intended consumer: pure_bfanet_v4_g04_train.py (CR-Q). Reuses the same
boundary_mask_r060.npy masks — mask generation is raw-point-cloud based
and grid-independent.
"""

import os

dataset_type = "BFDataset"
data_root = os.environ.get("SBF_DATA_ROOT")
if data_root is None:
    raise RuntimeError(
        "SBF_DATA_ROOT is required; implicit samples fallback has been removed."
    )

data = dict(
    num_classes=8,
    ignore_index=-1,
    names=[
        "balustrade",
        "balcony",
        "advboard",
        "wall",
        "eave",
        "column",
        "window",
        "clutter",
    ],
    train=dict(
        type=dataset_type,
        split=("training",),
        data_root=data_root,
        transform=[
            dict(type="InjectIndexValidKeys", keys=("edge", "boundary_mask", "s_weight")),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.4, mode="random"),
            dict(type="SphereCrop", point_max=80000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "edge", "boundary_mask", "s_weight"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="validation",
        data_root=data_root,
        transform=[
            dict(type="InjectIndexValidKeys", keys=("edge", "boundary_mask", "s_weight")),
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.04,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse", "edge", "boundary_mask", "s_weight"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="validation",
        data_root=data_root,
        transform=[
            dict(type="InjectIndexValidKeys", keys=("edge", "boundary_mask", "s_weight")),
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.04,
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
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
    ),
)
