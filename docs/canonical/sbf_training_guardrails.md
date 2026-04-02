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
