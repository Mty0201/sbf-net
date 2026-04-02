---
phase: 05-mainline-analysis-pivot
plan: 02
subsystem: docs
tags: [runtime, docs, guardrails, semantic-first]
requires:
  - phase: 05-mainline-analysis-pivot
    provides: semantic-first current-direction wording across maintainer docs
provides:
  - Runtime wrapper wording aligned to the semantic-first pivot
  - Canonical guardrails that keep historical config roles auditable without treating them as the active route
affects: [phase-05-mainline-analysis-pivot, runtime-guidance, canonical-docs]
key-files:
  created:
    - .planning/phases/05-mainline-analysis-pivot/05-02-SUMMARY.md
  modified:
    - train.md
    - docs/canonical/sbf_training_guardrails.md
key-decisions:
  - "The stable runtime entrypoint and fail-fast rules remain unchanged while config-role wording shifts to stable entry versus historical reference versus pending replacement route."
  - "Axis-side smoke/full-train commands remain documented only as historical/runtime-reference commands."
requirements-completed: [MAIN-01]
duration: 8m
completed: 2026-04-02
---

# Phase 5 Plan 02: Runtime Guidance Pivot Summary

Updated runtime-facing guidance so the stable entrypoint remains unchanged while the older axis-side path is documented as historical/reference evidence and the semantic-first route stays marked as pending later milestone phases.

## Accomplishments

- Rewrote `train.md` config-role wording to use stable runtime entry, historical reference configs, and a pending semantic-first replacement route.
- Updated `docs/canonical/sbf_training_guardrails.md` so command examples and config roles preserve auditable history without calling axis-side the active validation center.
- Kept the canonical training entrypoint, environment requirements, and no-fallback runtime rules intact.

## Task Commits

1. **Task 1: Rewrite the training wrapper config roles for the semantic-first pivot** - `62bfc54` (docs)
2. **Task 2: Update canonical training guardrails to match the new runtime wording boundaries** - `62bfc54` (docs)

## Self-Check: PASSED

- Verified runtime docs now contain `Stable runtime entry config`, `Historical reference configs`, and `Replacement semantic-first route`.
- Verified runtime docs no longer contain `Current verification focus` or `current validation center`.
