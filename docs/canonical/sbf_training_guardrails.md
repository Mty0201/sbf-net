# SBF Training Guardrails

## Purpose

This document is the Phase 1 canonical source for running and reviewing SBF training safely.
It preserves the minimum runtime rules maintainers need to identify the correct entrypoint,
choose the right config, and reject invalid runs instead of leaning on workflow memory files.

## Required Runtime Inputs

Both runtime inputs are mandatory before any smoke or full-train launch:

- `POINTCEPT_ROOT`: path to the external Pointcept checkout used by this repository.
- `SBF_DATA_ROOT`: path to the BF dataset root with the expected split layout.

Missing values must fail explicitly instead of using implicit parent-directory or environment
fallbacks. The training path must not guess a nearby Pointcept checkout, and it must not invent
an `SBF_DATA_ROOT` default when the dataset root is unset.

## Canonical Training Entrypoint

The canonical training entrypoint is `scripts/train/train.py`.

The canonical runtime trainer is `project.trainer.SemanticBoundaryTrainer`.

This repository does not use a Pointcept trainer as the runtime entry. `scripts/train/train.py`
bootstraps the local repo plus the external Pointcept checkout, loads the Python config, and
then instantiates `SemanticBoundaryTrainer` directly.

## Config Roles

Use the configs with these roles in mind:

- **Stable runtime entry config** (unchanged by Phase 7):
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`
  This remains the stable canonical runtime entry config for the repository.

- **Strongest reference baseline** (support-only, the comparison target):
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-support-only-train.py`
  `support-only (reg=1, cover=0.2) = 74.6`
  This is the comparison target the active route should beat. It is not the active implementation route.

- **Active implementation route** (Phase 7, support-guided semantic focus):
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train.py`
  Uses `SharedBackboneSemanticSupportModel`, `SupportGuidedSemanticFocusLoss`, `SupportGuidedSemanticFocusEvaluator`.
  Trains from scratch (`weight = None`, `resume = False`). Locally smoke-validated (Phase 8). Full-train validation pending.
  Do not change `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py` — it is the stable entry, not the active route.

- **Historical reference configs** (evidence only):
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py`
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py`
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-train.py`
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-semantic-train-smoke.py`

- **Side evidence** (weaker than baseline):
  `support-shape` — weaker than the support-only baseline and not the semantic-first mainline.

The runtime entrypoint and fail-fast rules stay fixed. The active implementation route is the support-guided semantic focus route. Do not rewrite the stable entry config, do not elevate support-shape to the mainline, and do not claim local smoke/full-train validation that Phase 8 has not yet produced.

## Guardrails That Must Not Be Relaxed

- `val_mIoU` remains the best-checkpoint selection criterion.
- Missing `POINTCEPT_ROOT` or `SBF_DATA_ROOT` must fail fast instead of silently falling back.
- CPU PTv3 fallback language is banned. If the runtime or host environment cannot initialize the
  intended PTv3 path, the run should fail visibly.
- Phase 1 is not rewriting `README.md`, `train.md`, or `install.md` yet. This document is the
  canonical training guardrail source for this migration phase.

## Canonical Command Patterns

Use the exact command prefix:

```bash
conda run --no-capture-output -n ptv3 python scripts/train/train.py
```

Historical axis-side smoke example:

```bash
export POINTCEPT_ROOT=/path/to/Pointcept
export SBF_DATA_ROOT=/path/to/BF_edge_chunk_npy
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py \
  --pointcept-root "${POINTCEPT_ROOT}"
```

Historical axis-side verification/full-train example:

```bash
export POINTCEPT_ROOT=/path/to/Pointcept
export SBF_DATA_ROOT=/path/to/BF_edge_chunk_npy
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py \
  --pointcept-root "${POINTCEPT_ROOT}"
```

## Active Route Command Pattern

```bash
export POINTCEPT_ROOT=/path/to/Pointcept
export SBF_DATA_ROOT=/path/to/BF_edge_chunk_npy
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train.py \
  --pointcept-root "${POINTCEPT_ROOT}"
```

This config trains from scratch with no checkpoint loading. No Pointcept changes required.

## Smoke Versus Full-Train Use

The smoke path exists to validate wiring, environment setup, and trainer startup before an
expensive run.

- smoke success does not equal full-train validation.

Use `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py` for a
minimal historical axis-side smoke startup check, then use
`configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py` for the matching
historical longer-run reference when you need to audit that route. The active implementation route is the support-guided semantic focus route (see `## Active Route Command Pattern` above).

## Invalid Run Patterns

The following warnings must remain visible to maintainers:

- no implicit POINTCEPT_ROOT fallback
- no implicit SBF_DATA_ROOT fallback
- no CPU PTv3 fallback
- no assumption that the `pointcept` environment without `flash_attn` is sufficient for PTv3
  initialization

If one of these conditions is not satisfied, the correct behavior is an explicit failure rather
than a compatibility shortcut.

## Evidence Boundary

Phase 8 local validation confirms the active route runs correctly:
- Model forward produces `seg_logits` and `support_pred`
- All three loss terms (`loss_semantic`, `loss_support`, `loss_focus`) are non-NaN
- Backward pass and optimizer step succeed
- Focus weighting activates in boundary regions

**This is NOT full-train validation.** No performance claims are valid until the active route config is run in a full-training environment and compared against the support-only baseline (val_mIoU = 74.6). Do not cite local smoke results as evidence of improved semantic performance.

Validation script: `scripts/train/check_active_route_train_step.py`
Experiment handoff: `docs/canonical/sbf_validation_and_experiment_handoff.md`
