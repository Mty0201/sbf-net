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

- Stable main config: `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`
  This remains the stable canonical full-train config for the repository.
- Current verification focus: `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py`
  This is the current Stage-2 `axis + side + support` verification target.
- Current smoke verification config:
  `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py`
  Use this to validate the current `axis-side` wiring before spending time on a longer run.

The current validation center is the `axis-side` train path and its smoke config. Do not rewrite
that focus as if smoke or full-train verification has already been completed.

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

Current axis-side smoke example:

```bash
export POINTCEPT_ROOT=/path/to/Pointcept
export SBF_DATA_ROOT=/path/to/BF_edge_chunk_npy
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py \
  --pointcept-root "${POINTCEPT_ROOT}"
```

Current axis-side verification/full-train example:

```bash
export POINTCEPT_ROOT=/path/to/Pointcept
export SBF_DATA_ROOT=/path/to/BF_edge_chunk_npy
conda run --no-capture-output -n ptv3 python scripts/train/train.py \
  --config configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py \
  --pointcept-root "${POINTCEPT_ROOT}"
```

## Smoke Versus Full-Train Use

The smoke path exists to validate wiring, environment setup, and trainer startup before an
expensive run.

- smoke success does not equal full-train validation.

Use `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py` for a
minimal axis-side smoke startup check, then use
`configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train.py` for the current
verification target when the environment is ready for the longer run.

## Invalid Run Patterns

The following warnings must remain visible to maintainers:

- no implicit POINTCEPT_ROOT fallback
- no implicit SBF_DATA_ROOT fallback
- no CPU PTv3 fallback
- no assumption that the `pointcept` environment without `flash_attn` is sufficient for PTv3
  initialization

If one of these conditions is not satisfied, the correct behavior is an explicit failure rather
than a compatibility shortcut.
