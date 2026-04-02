# Training

This file is a runtime reference, not the repository's workflow-control entry. Default planning and execution start with GSD and `.planning/`. For authoritative current-mainline status, experiment constraints, and no-fallback run rules, read [docs/canonical/README.md](docs/canonical/README.md) and [docs/canonical/sbf_training_guardrails.md](docs/canonical/sbf_training_guardrails.md).

## 1. Canonical Entrypoint

The canonical training entrypoint is:

```text
scripts/train/train.py
```

The canonical runtime trainer is `project.trainer.SemanticBoundaryTrainer`.

This repository does not use a Pointcept trainer as the runtime entry.

## 2. Required Runtime Inputs

Set both inputs before any smoke or longer run:

```bash
export POINTCEPT_ROOT=/path/to/Pointcept
export SBF_DATA_ROOT=/path/to/BF_edge_chunk_npy
```

Missing values should fail explicitly. Do not rely on implicit Pointcept-path or dataset-root fallbacks.

## 3. Current Config Roles

- Stable main config:
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`
- Current verification focus:
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py`
- Current smoke verification config:
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py`
- Semantic-only calibration smoke:
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-train-smoke.py`
- Semantic-only calibration full:
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-train.py`

The current validation center is the axis-side train path and its smoke config. Do not rewrite that state as if axis-side smoke or full-train verification were already complete.

## 4. Command Pattern

Use the exact command prefix:

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py
```

Current axis-side smoke example:

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py \
  --pointcept-root "${POINTCEPT_ROOT}"
```

Current axis-side verification/full-train example:

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py \
  --pointcept-root "${POINTCEPT_ROOT}"
```

## 5. Runtime Rules That Still Matter

- `val_mIoU` remains the best-checkpoint selection criterion
- no implicit `POINTCEPT_ROOT` fallback
- no implicit `SBF_DATA_ROOT` fallback
- no CPU PTv3 fallback language
- no assumption that a `pointcept` environment without `flash_attn` is sufficient for PTv3 initialization

If one of these conditions is not satisfied, the correct behavior is an explicit failure rather than a compatibility shortcut.

## 6. Where To Look Next

- [docs/canonical/sbf_training_guardrails.md](docs/canonical/sbf_training_guardrails.md): canonical guardrails and invalid-run patterns
- [docs/canonical/sbf_facts.md](docs/canonical/sbf_facts.md): current Stage-2 and active-mainline facts
- [install.md](install.md): environment setup reference
