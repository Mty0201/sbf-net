---
phase: 10-loss-redesign-fix-support-supervision-and-boundary-focus
plan: 01
subsystem: losses
tags: [smoothl1, tversky, lovasz, support-supervision, boundary-focus]

# Dependency graph
requires:
  - phase: 07-support-guided-semantic-focus-route-implementation
    provides: SharedBackboneSemanticSupportModel, SupportHead, trainer wiring
  - phase: 09-phase-7-full-training-results-analysis-and-tuning
    provides: BCE saturation diagnosis, focus term redundancy analysis
provides:
  - RedesignedSupportFocusLoss class with SmoothL1+Tversky support and optional Lovasz focus
  - RedesignedSupportFocusEvaluator class with SmoothL1 regression error metric
  - Registry dispatch for both in build_loss and build_evaluator
affects: [10-02 config creation, future training experiments]

# Tech tracking
tech-stack:
  added: []
  patterns: [SmoothL1+Tversky for continuous support regression, optional Lovasz-on-boundary focus via focus_mode flag]

key-files:
  created:
    - project/losses/redesigned_support_focus_loss.py
    - project/evaluator/redesigned_support_focus_evaluator.py
  modified:
    - project/losses/__init__.py
    - project/evaluator/__init__.py

key-decisions:
  - "Single loss class with focus_mode flag (none/lovasz) instead of two separate classes"
  - "Evaluator reports support_reg_error replacing support_bce for consistency with new loss"

patterns-established:
  - "focus_mode parameter pattern: configurable loss behavior via string mode flag"

requirements-completed: [LOSS-01, LOSS-02, LOSS-03, LOSS-04]

# Metrics
duration: 4min
completed: 2026-04-03
---

# Phase 10 Plan 01: Loss and Evaluator Redesign Summary

**SmoothL1+Tversky support loss replacing saturated BCE, with optional Lovasz-on-boundary focus mode for Variant A/C ablation**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-03T14:42:33Z
- **Completed:** 2026-04-03T14:46:32Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created RedesignedSupportFocusLoss with SmoothL1+Tversky support supervision (fixing BCE entropy-floor saturation) and optional Lovasz-on-boundary focus (fixing 93% redundant CE)
- Created RedesignedSupportFocusEvaluator that reports support_reg_error instead of support_bce
- Registered both in their respective __init__.py files with build_loss/build_evaluator dispatch

## Task Commits

Each task was committed atomically:

1. **Task 1: Create RedesignedSupportFocusLoss class** - `8dbd82b` (feat)
2. **Task 2: Create RedesignedSupportFocusEvaluator and register both** - `3318ad4` (feat)

## Files Created/Modified
- `project/losses/redesigned_support_focus_loss.py` - New loss class with SmoothL1+Tversky support and optional Lovasz focus
- `project/evaluator/redesigned_support_focus_evaluator.py` - New evaluator with SmoothL1 regression error metric
- `project/losses/__init__.py` - Added import and build_loss dispatch for RedesignedSupportFocusLoss
- `project/evaluator/__init__.py` - Added import and build_evaluator dispatch for RedesignedSupportFocusEvaluator

## Decisions Made
- Used single class with `focus_mode` parameter (`"none"` for Variant C, `"lovasz"` for Variant A) rather than two separate classes -- simpler to maintain and configure
- Evaluator reports `support_reg_error` (SmoothL1-based) replacing `support_bce` for metric consistency with the new loss design

## Deviations from Plan

None - plan executed exactly as written.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None - all functionality is fully wired.

## Next Phase Readiness
- Loss and evaluator are registered and ready for config creation (Plan 02)
- Plan 02 will create training configs that reference RedesignedSupportFocusLoss and RedesignedSupportFocusEvaluator

---
*Phase: 10-loss-redesign-fix-support-supervision-and-boundary-focus*
*Completed: 2026-04-03*
