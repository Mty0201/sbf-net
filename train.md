# Training

This document describes the current project-local training behavior of SBF-Net.  
本文档描述当前 SBF-Net 项目内训练系统的运行方式。

Current edge supervision uses:
当前 edge 监督使用：

- `edge.npy[:, 0:3]` -> `edge_dir`
- `edge.npy[:, 3]` -> `edge_dist`
- `edge.npy[:, 4]` -> `edge_support`
- `edge.npy[:, 5]` -> `edge_valid`

`edge_valid` is only a supervision validity domain. It is not a predicted mask target. `edge_dir` is the core geometric supervision, `edge_dist` is the explicit snapping step length, and `edge_support` no longer acts as a distance substitute. Legacy names `edge_strength` and `edge_mask` are kept only as dataset-side compatibility aliases.

Current support semantics:

- `edge_support` is treated as a coarse boundary-neighborhood proposal field
- its main goal is to cover the boundary snapping region rather than reconstruct an exact outer contour
- the valid-radius band itself is human-defined, so support coverage recall matters more than exact shape matching

## 1. Training Entry

Current project-local training entry:

```text
scripts/train/train.py
```

This script builds and runs the independent project trainer. It does not use Pointcept's trainer as the runtime entry.

The current default entry still points to the full dual-task config. Semantic-only calibration uses the same entry script with a different config.

## 2. Config Variants

Current stage provides two training configs.
当前阶段提供双任务配置和 semantic-only 校准配置。

### Smoke Config

Path:

```text
configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train-smoke.py
```

Purpose:

- minimal smoke verification
- Pointcept-style displayed epoch organization on a tiny run
- gradient accumulation enabled
- one displayed epoch
- one validation batch only
- CPU fallback allowed for local checks

### Full Train Config

Path:

```text
configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py
```

Purpose:

- stage-1 full training
- full train loader
- full validation loader
- Pointcept-style `total_epoch / eval_epoch / train_loop` organization
- gradient accumulation enabled
- scheduler enabled
- resume and weight loading supported by the training entry
- no smoke batch limits
- CPU fallback disabled by default
- best checkpoint selected by `val_mIoU`

### Semantic-Only Baseline Configs

Additional semantic-only configs are also provided for framework calibration against the original Pointcept PTv3 semantic segmentation path.
semantic-only 不是主任务替代方案，而是训练框架校准基线。

- smoke:
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-train-smoke.py`
- full:
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-train.py`

Purpose of semantic-only:

- isolate the effect of the project-local trainer/runtime from the added boundary branch
- compare SBF-Net semantic-only behavior against the original Pointcept PTv3 semantic segmentation training path
- inspect convergence trend, per-class semantic metrics, and logging behavior under the same project-local runtime

## 3. Smoke Training Command

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train-smoke.py \
  --pointcept-root /path/to/Pointcept
```

Before running, set:

```bash
export SBF_DATA_ROOT=/path/to/BF_edge_chunk_npy
```

## 4. Full Training Command

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py \
  --pointcept-root /path/to/Pointcept
```

Before running, set:

```bash
export SBF_DATA_ROOT=/path/to/BF_edge_chunk_npy
```

## 5. Semantic-Only Calibration Commands

Smoke:

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-train-smoke.py \
  --pointcept-root /path/to/Pointcept
```

Full:

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-train.py \
  --pointcept-root /path/to/Pointcept
```

## 6. Runtime Behavior

Current full dual-task and semantic-only training both use:

- `OneCycleLR`
- `grad_accum_steps`
- `mix_prob`
- AMP on CUDA full runs
- `total_epoch`
- `eval_epoch`
- derived `train_loop`
- `data.train_batch_size`
- `data.val_batch_size`

The effective batch size is:

```text
effective batch size = data.train_batch_size * grad_accum_steps
```

The trainer prints iteration-level logs by default. Typical train logs look like:

```text
Train: [epoch/max_epoch][iter/max_iter] Data ... Batch ... Remain ... loss ... Accum x/y Lr ...
```

Typical validation logs include:

```text
Val/Test: [iter/max_iter] ...
Val result: mIoU/mAcc/allAcc ...
Class_x Result: iou/accuracy ...
```

## 7. Per-Epoch Output

Each displayed epoch currently prints:

Train:

- `loss`
- `loss_semantic`
- `loss_edge`
- `loss_support`
- `loss_support_cover`
- `loss_support_reg`
- `loss_dir`
- `loss_dist`
- `valid_ratio`
- `support_positive_ratio`
- `dir_valid_ratio`
- `dist_gt_valid_mean`
- `dir_cosine`
- `dist_error`

Legacy aliases `loss_mask` and `loss_strength` may still be returned internally for compatibility, but they are no longer the main printed keys.

The semantic branch is now aligned between `semantic-only` and `dual-task`:

- semantic loss = `CrossEntropy + Lovasz`
- optimizer = `AdamW`
- block parameters use a lower LR group

Current train logs intentionally keep semantic loss aggregated as:

- `loss_semantic`

They do not separately print `loss_ce` or `loss_lovasz`.

Validation:

- `val_mIoU`
- `val_mAcc`
- `val_allAcc`
- `val_loss_edge`
- `val_loss_support`
- `val_loss_dir`
- `val_loss_dist`
- `val_loss_support_reg`
- `val_loss_support_cover`
- `valid_ratio`
- `support_positive_ratio`
- `dir_valid_ratio`
- `dist_gt_valid_mean`
- `dir_cosine`
- `dist_error`
- `support_cover`
- `support_error`

Legacy aliases such as `val_loss_mask`, `val_loss_strength`, `mask_f1`, and `strength_error_masked` are retained only as compatibility keys.

Runtime-related fields currently used by the trainer:

- `grad_accum_steps`
- `mix_prob`
- `enable_amp`
- `total_epoch`
- `eval_epoch`
- derived `train_loop`
- scheduler configuration
- `resume`
- `weight`
- `log_freq`
- `save_freq`

## 8. Checkpoint Output

Current trainer writes:

- `model_last.pth`
- `model_best.pth`

Default output directories:

- smoke:
  `outputs/semantic_boundary_train_smoke/model/`
- full:
  `outputs/semantic_boundary_train/model/`

Semantic-only output directories:

- smoke:
  `outputs/semantic_only_train_smoke/model/`
- full:
  `outputs/semantic_only_train/model/`

## 9. Best Checkpoint Rule

Current stage keeps the best checkpoint policy intentionally simple:

- `model_best.pth` is selected only by `val_mIoU`
- edge metrics are recorded during validation
- edge metrics do not participate in stage-1 best checkpoint selection

This rule is shared by both:

- dual-task training
- semantic-only calibration training

## 10. Training Artifacts

Current training artifacts are limited to:

- console logs
- `train.log`
- `model_last.pth`
- `model_best.pth`

Resume training:

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py \
  --pointcept-root /path/to/Pointcept \
  --resume
```

Load weight only:

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py \
  --pointcept-root /path/to/Pointcept \
  --weight /path/to/checkpoint.pth
```

## 11. Semantic-Only vs Dual-Task

Current task split:

- semantic-only:
  semantic segmentation only
- dual-task:
  semantic segmentation + boundary field prediction

semantic-only should be used for trainer/runtime calibration, not as the final target task.

Current dual-task edge semantics:

- model predicts `support + vec`
- `support` is the primary continuous boundary support supervision
- `vec` is the offset from a point to its nearest boundary support projection point
- `valid` only clips supervision to numerically meaningful points
- current trainer still prints several legacy field names for compatibility, but they should be interpreted as support-related aliases rather than mask-task outputs

When comparing SBF-Net semantic-only against the original Pointcept PTv3 semantic segmentation path, do not rely only on absolute final numbers. At the current stage, you should compare:

- convergence trend
- per-class semantic performance
- logging behavior
- checkpoint behavior
- runtime organization

## 12. What Gets Printed During Training

Each displayed epoch currently prints:

- current epoch
- total epoch / eval epoch / train loop summary
- optimizer steps per epoch
- current and average data time
- current and average batch time
- global remaining training time
- current accumulation state
- train losses
- validation semantic metrics
- validation edge metrics
- current `val_mIoU`
- best `val_mIoU`
- checkpoint output paths

Compatibility note:

- current trainer log strings still contain legacy names such as `loss_mask`, `loss_strength`, `val_loss_mask`, `mask_f1`, and `strength_error_masked`
- these are compatibility aliases kept to avoid changing the trainer loop
- they currently correspond to support-related values, not to a predicted mask branch

Current stage does not include:

- test pipeline outputs
- prediction export
- visualization export
- advanced logging backends
