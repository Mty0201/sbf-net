---
phase: 11-boundary-metrics-fix-and-focus-tuning
plan: 02
subsystem: config
tags: [lovasz, focus-weight, training-config, variant-a2]

# Dependency graph
requires:
  - phase: 10-loss-redesign-fix-support-supervision-and-boundary-focus
    provides: RedesignedSupportFocusLoss with focus_mode=lovasz and Variant A config template
provides:
  - Variant A2 training config with focus_weight=0.15 and 300 eval epochs
affects: [training-experiment, boundary-focus-tuning]

# Tech tracking
tech-stack:
  added: []
  patterns: [config-variant-inheritance]

key-files:
  created:
    - configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-redesigned-variant-a2-train.py
  modified: []

key-decisions:
  - "focus_weight=0.15 keeps Lovasz gradient ~1.07x semantic CE at convergence (subordinate, not competing)"
  - "300 eval epochs via total_epoch=6000 and eval_epoch=300 matches baseline training duration"
  - "train_loop=20 data passes per eval epoch preserved from Variant A"

patterns-established:
  - "Variant config inheritance: copy Variant A, modify only loss params, trainer epochs, and work_dir"

requirements-completed: [METRIC-04, METRIC-05]

# Metrics
duration: 1min
completed: 2026-04-04
---

# Phase 11 Plan 02: Variant A2 Training Config Summary

**Variant A2 config with focus_weight=0.15 (down from 0.5) and 300 eval epochs to test subordinate Lovasz focus without epoch-count confound**

## Performance

- **Duration:** 1 min
- **Started:** 2026-04-04T13:49:18Z
- **Completed:** 2026-04-04T13:50:10Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created Variant A2 training config that reduces focus_weight from 0.5 to 0.15 to keep Lovasz gradient subordinate to semantic CE
- Extended training to 300 eval epochs (total_epoch=6000, eval_epoch=300) matching the support-only baseline duration
- All other parameters (model, optimizer, scheduler, data, support loss weights) identical to Variant A

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Variant A2 training config** - `86cf652` (feat)

## Files Created/Modified
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-redesigned-variant-a2-train.py` - Variant A2 training config with tuned focus_weight and extended epochs

## Decisions Made
- focus_weight=0.15 chosen per D-06: at convergence (semantic CE ~0.14), Lovasz gradient contribution is ~1.07x semantic gradient on boundary points, making it a mild boost rather than a competing signal
- eval_epoch=300 (not 100) per D-08: with train_loop=20 preserved, total_epoch=6000 yields 300 eval epochs matching the baseline's training duration
- boundary_threshold=0.1 kept identical to Variant A per D-07

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Variant A2 config is ready for full training on the training server
- Monitor val_boundary_mIoU throughout training (enabled by Plan 01 boundary metric fix)
- Success criteria per D-10: val_mIoU >= 0.74, val_boundary_mIoU > Variant C, no class regresses > 3 pp

## Self-Check: PASSED

- Config file: FOUND
- SUMMARY.md: FOUND
- Commit 86cf652: FOUND

---
*Phase: 11-boundary-metrics-fix-and-focus-tuning*
*Completed: 2026-04-04*
