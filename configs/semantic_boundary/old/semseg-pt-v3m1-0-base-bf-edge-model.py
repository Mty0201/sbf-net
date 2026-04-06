"""
Minimal model-only config for shared PTv3 backbone with semantic and dir/dist/support heads.

The edge output follows edge.npy = [dir_x, dir_y, dir_z, edge_dist, edge_support, edge_valid].
This config is intended for forward smoke testing only.
"""
# ======================================================================
# LEGACY CONFIG — belongs to the old experiment chain (Phases 5-11).
# Not the canonical baseline. See configs/semantic_boundary/clean_reset_s38873367/
# for the current clean-reset experiment configs.
# ======================================================================


model = dict(
    type="SharedBackboneSemanticBoundaryModel",
    num_classes=8,
    backbone_out_channels=64,
    edge_out_channels=5,
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
