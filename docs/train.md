# Training

## 1. Training Entry

Current project-local training entry:

```text
scripts/train/train.py
```

This script builds and runs the independent project trainer. It does not use Pointcept's trainer as the runtime entry.

## 2. Config Variants

Current stage provides two training configs.

### Smoke Config

Path:

```text
configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train-smoke.py
```

Purpose:

- minimal smoke verification
- one epoch only
- one train batch only
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
- no smoke batch limits
- CPU fallback disabled by default

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

## 5. Per-Epoch Output

Each epoch currently prints:

Train:

- `loss`
- `loss_semantic`
- `loss_mask`
- `loss_vec`
- `loss_strength`

Validation:

- `val_mIoU`
- `val_mAcc`
- `val_allAcc`
- `val_loss_mask`
- `val_loss_vec`
- `val_loss_strength`
- `mask_precision`
- `mask_recall`
- `mask_f1`
- `vec_error_masked`
- `strength_error_masked`

## 6. Checkpoint Output

Current trainer writes:

- `model_last.pth`
- `model_best.pth`

Default output directories:

- smoke:
  `outputs/semantic_boundary_train_smoke/`
- full:
  `outputs/semantic_boundary_train/`

## 7. Best Checkpoint Rule

Current stage keeps the best checkpoint policy intentionally simple:

- `model_best.pth` is selected only by `val_mIoU`
- edge metrics are recorded during validation
- edge metrics do not participate in stage-1 best checkpoint selection

## 8. Training Artifacts

Current training artifacts are limited to:

- console logs
- `model_last.pth`
- `model_best.pth`

## 9. What Gets Printed During Training

Each epoch currently prints:

- current epoch
- train losses
- validation semantic metrics
- validation edge metrics
- current `val_mIoU`
- best `val_mIoU`
- checkpoint output paths

Current stage does not include:

- test pipeline outputs
- prediction export
- visualization export
- advanced logging backends
