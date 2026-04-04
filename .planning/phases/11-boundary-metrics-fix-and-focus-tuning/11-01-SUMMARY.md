---
phase: 11-boundary-metrics-fix-and-focus-tuning
plan: 01
subsystem: training
tags: [trainer, logging, metrics, boundary-miou, support-reg-error, log-parser]

requires:
  - phase: 10-loss-redesign
    provides: RedesignedSupportFocusLoss with SmoothL1+Tversky support and evaluator returning support_reg_error keys
provides:
  - Trainer metric dispatch and val batch logging for RedesignedSupportFocusLoss
  - Log parser redesigned run type detection and REDESIGNED_COLUMNS CSV output
affects: [11-02, training-runs, experiment-analysis]

tech-stack:
  added: []
  patterns: [loss-type-based metric dispatch in trainer validate(), run-type detection in log parser]

key-files:
  created: []
  modified:
    - project/trainer/trainer.py
    - scripts/analysis/parse_train_log.py

key-decisions:
  - "support_reg_error key used instead of support_bce to match redesigned evaluator return dict"
  - "Val batch log branch dispatches on 'support_reg_error in metric_meters' before 'val_boundary_mIoU' to distinguish redesigned from old Phase 7 loss"
  - "Train result log detects redesigned loss by presence of loss_support_reg in train_metrics"
  - "Log parser detects redesigned runs by loss_support_reg= appearing before loss_focus= check in Train result lines"

patterns-established:
  - "Metric dispatch ordering: specific loss types before generic fallback"
  - "Log parser run-type detection: most specific indicator checked first"

requirements-completed: [METRIC-01, METRIC-02, METRIC-03]

duration: 3min
completed: 2026-04-04
---

# Phase 11 Plan 01: Boundary Metrics Fix and Log Parser Update Summary

**Trainer registers and logs val_boundary_mIoU, support_reg_error, support_cover for RedesignedSupportFocusLoss; log parser detects redesigned runs and produces CSV with boundary+support columns**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-04T13:44:09Z
- **Completed:** 2026-04-04T13:47:23Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Trainer validate() registers all 8 boundary+support metrics for RedesignedSupportFocusLoss (val_boundary_mIoU, val_boundary_mAcc, boundary_point_ratio, support_reg_error, support_cover, valid_ratio, support_positive_ratio, val_loss_semantic)
- Val batch log format emits support_reg_error instead of support_bce for redesigned loss, enabling parseable boundary metric extraction
- Train result log distinguishes redesigned loss (with loss_support_reg/loss_support_cover) from old Phase 7 focus loss
- Log parser auto-detects redesigned run type and outputs REDESIGNED_COLUMNS CSV with all boundary and split support train loss keys

## Task Commits

Each task was committed atomically:

1. **Task 1: Add RedesignedSupportFocusLoss metric dispatch to trainer validate()** - `c49ba17` (feat)
2. **Task 2: Add redesigned run type to parse_train_log.py** - `5ee673e` (feat)

## Files Created/Modified
- `project/trainer/trainer.py` - Added RedesignedSupportFocusLoss metric dispatch, val batch log format with support_reg_error, and redesigned Train result log branch
- `scripts/analysis/parse_train_log.py` - Added REDESIGNED_COLUMNS, RE_SUPPORT_REG_ERROR regex, redesigned run type detection, and redesigned Train result backfill

## Decisions Made
- Used `support_reg_error` as the metric key (not `support_bce`) to match the redesigned evaluator return dict keys
- Val batch logging dispatches on `support_reg_error in metric_meters` before `val_boundary_mIoU in metric_meters` to distinguish redesigned from old Phase 7 loss type
- Log parser checks `loss_support_reg=` before `loss_focus=` in Train result lines to distinguish redesigned from old focus runs
- Both RE_SUPPORT_BCE and RE_SUPPORT_REG_ERROR are in the Val/Test batch regex list safely since only one matches per run type

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Trainer and log parser are ready for RedesignedSupportFocusLoss training runs
- Plan 11-02 can proceed with focus tuning knowing boundary metrics will be logged and parseable
- Old loss type branches (SupportGuidedSemanticFocusLoss, SemanticBoundaryLoss) remain unchanged

---
*Phase: 11-boundary-metrics-fix-and-focus-tuning*
*Completed: 2026-04-04*
