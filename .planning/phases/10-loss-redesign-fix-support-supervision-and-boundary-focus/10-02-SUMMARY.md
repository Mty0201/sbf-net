---
phase: 10-loss-redesign-fix-support-supervision-and-boundary-focus
plan: 02
subsystem: training-config
tags: [training-config, ablation, lovasz, support-loss, variant-c, variant-a]

# Dependency graph
requires:
  - phase: 10-01
    provides: RedesignedSupportFocusLoss class and RedesignedSupportFocusEvaluator
provides:
  - Variant C ablation training config (SmoothL1+Tversky, no focus)
  - Variant A boundary-focus training config (adds Lovasz-on-boundary)
affects: [full-training, experiment-validation]

# Tech tracking
tech-stack:
  added: []
  patterns: [config-variant-naming, conditional-experiment-gating]

key-files:
  created:
    - configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-redesigned-variant-c-train.py
    - configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-redesigned-variant-a-train.py
  modified: []

key-decisions:
  - "Both configs inherit model/optimizer/scheduler/data from active route verbatim"
  - "Variant A is conditional on Variant C achieving val_mIoU >= 0.74 (D-08)"

patterns-established:
  - "Config variant pattern: identical base with only loss dict and work_dir differing"

requirements-completed: [LOSS-05, LOSS-06, LOSS-07]

# Metrics
duration: 1min
completed: 2026-04-03
---

# Phase 10 Plan 02: Training Config Variants Summary

**Two training configs: Variant C ablation (SmoothL1+Tversky, no focus) and Variant A (adds Lovasz-on-boundary focus_weight=0.5)**

## Performance

- **Duration:** 1 min
- **Started:** 2026-04-03T14:48:52Z
- **Completed:** 2026-04-03T14:50:07Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created Variant C config matching support-only baseline loss parameters with SharedBackboneSemanticSupportModel
- Created Variant A config adding Lovasz-on-boundary focus with weight 0.5 and boundary_threshold 0.1
- Both configs verified to load via runpy without errors

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Variant C ablation training config** - `77564c1` (feat)
2. **Task 2: Create Variant A boundary-focus training config** - `7cc10f5` (feat)

## Files Created/Modified
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-redesigned-variant-c-train.py` - Ablation baseline: SmoothL1+Tversky support, focus_mode=none, 300 eval epochs
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-redesigned-variant-a-train.py` - Boundary focus: adds Lovasz with focus_weight=0.5, boundary_threshold=0.1

## Decisions Made
None - followed plan as specified. Both configs exactly match the plan's prescribed parameters.

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both configs are ready for full training runs
- Variant C should run first to establish whether the cleaner architecture matches baseline (target val_mIoU >= 0.74)
- Variant A only proceeds if Variant C succeeds (D-08 condition)

## Self-Check: PASSED

All files and commits verified.

---
*Phase: 10-loss-redesign-fix-support-supervision-and-boundary-focus*
*Completed: 2026-04-03*
