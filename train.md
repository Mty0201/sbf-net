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

- stable runtime entry config:
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`
- Historical reference configs:
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py`
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py`
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-train-smoke.py`
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-train.py`
- support-only reference baseline:
  `support-only (reg=1, cover=0.2) = 74.6`
- support-shape side evidence:
  weaker than the support-only baseline and not the semantic-first mainline
- Phase 6 semantic-first candidate route:
  `support-guided semantic focus route` documented in `docs/canonical/sbf_semantic_first_contract.md`

The stable runtime entry remains unchanged. The axis-side and semantic-only configs remain auditable historical/reference paths, not the preferred current mainline. The current semantic-first design target is support-centric and support-only-first; do not rewrite that evidence as if support-shape or the candidate replacement route were already implemented or fully validated.

## 4. Command Pattern

Use the exact command prefix:

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py
```

Historical axis-side smoke example:

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py \
  --pointcept-root "${POINTCEPT_ROOT}"
```

Historical axis-side verification/full-train example:

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

- [docs/canonical/sbf_semantic_first_contract.md](docs/canonical/sbf_semantic_first_contract.md): support-only-first candidate route contract for Phase 6
- [docs/canonical/sbf_semantic_first_route.md](docs/canonical/sbf_semantic_first_route.md): support-only baseline and candidate-route definition
- [docs/canonical/sbf_training_guardrails.md](docs/canonical/sbf_training_guardrails.md): canonical guardrails and invalid-run patterns
- [docs/canonical/sbf_facts.md](docs/canonical/sbf_facts.md): current Stage-2 and active-mainline facts
- [install.md](install.md): environment setup reference
