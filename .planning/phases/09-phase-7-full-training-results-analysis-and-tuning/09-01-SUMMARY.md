---
phase: 09-phase-7-full-training-results-analysis-and-tuning
plan: 01
subsystem: analysis
tags: [parsing, csv, metrics, training-logs]

requires:
  - phase: 08-phase-7-implementation-validation
    provides: completed training runs with train.log files
provides:
  - Log-to-CSV parsing for both focus (active route) and support_only (baseline) runs
  - metrics_epoch.csv and per_class_iou.csv for convergence and per-class analysis
affects: [09-02 analysis report]

tech-stack:
  added: []
  patterns: [line-by-line log parsing with auto-detection, regex-based metric extraction]

key-files:
  created:
    - scripts/analysis/parse_train_log.py
  modified: []

key-decisions:
  - "Train result backfill pattern: Val result appears before Train result in log, so rows are created at Val result time and train metrics backfilled when Train result appears"
  - "Boundary metrics from last Val/Test batch line used as epoch average (AverageMeter running avg equals epoch avg at final batch)"

patterns-established:
  - "Log parsing auto-detects run type from loss_focus= vs loss_edge= in Train result lines"

requirements-completed: [ANALYSIS-01, ANALYSIS-02]

duration: 3min
completed: 2026-04-03
---

# Phase 09 Plan 01: Log Parsing Script Summary

**Line-by-line train.log parser producing metrics_epoch.csv and per_class_iou.csv for both focus (active route, 100 epochs) and support_only (baseline, 300 epochs) runs**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-03T12:52:19Z
- **Completed:** 2026-04-03T12:55:37Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created `scripts/analysis/parse_train_log.py` with auto-detection of focus vs support_only run type
- Active route: 100 epoch rows with val_mIoU, boundary metrics, and train losses; best val_mIoU 0.7265 at epoch 61
- Support-only baseline: 300 epoch rows with all loss components; best val_mIoU 0.7457 at epoch 252
- Per-class IoU CSV with 8 classes (balustrade, balcony, advboard, wall, eave, column, window, clutter) for both runs

## Task Commits

Each task was committed atomically:

1. **Task 1: Create log parsing script for both run types** - `48faa85` (feat)

## Files Created/Modified
- `scripts/analysis/parse_train_log.py` - CLI tool that parses train.log into metrics_epoch.csv and per_class_iou.csv

## Decisions Made
- Train result lines appear AFTER Val result in the log (per-epoch order: Val/Test batches -> Val result -> per-class -> Train result), so the parser creates rows at Val result time and backfills train metrics when Train result appears
- Boundary metrics (val_boundary_mIoU, support_bce, etc.) taken from the last Val/Test batch line per epoch, which equals the epoch average since AverageMeter tracks cumulative averages

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed trailing period in val_allAcc values**
- **Found during:** Task 1 (initial verification)
- **Issue:** The trainer logs `Val result: mIoU/mAcc/allAcc 0.7951.` with a sentence-ending period that got captured as part of the float value
- **Fix:** Changed regex from `([\d.]+)` to `(\d+\.\d+)` which only matches proper decimal numbers
- **Files modified:** scripts/analysis/parse_train_log.py
- **Verification:** CSV values now clean floats without trailing periods
- **Committed in:** 48faa85

**2. [Rule 1 - Bug] Fixed epoch-train metric pairing**
- **Found during:** Task 1 (initial verification)
- **Issue:** Initial implementation assumed Train result came before Val result (like many loggers), but the actual log order is Val first, Train after. This caused epoch 1 to have empty train metrics and epoch 2 to get epoch 1's train values.
- **Fix:** Restructured parser to create row at Val result time and backfill train metrics when Train result appears afterward
- **Files modified:** scripts/analysis/parse_train_log.py
- **Verification:** Epoch 1 now correctly has train_loss=2.9700 matching the first Train result line
- **Committed in:** 48faa85

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes essential for data correctness. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both CSVs ready for side-by-side convergence comparison in Plan 02 analysis report
- Key findings available: focus best 0.7265 mIoU at epoch 61, support_only best 0.7457 mIoU at epoch 252

---
*Phase: 09-phase-7-full-training-results-analysis-and-tuning*
*Completed: 2026-04-03*

## Self-Check: PASSED
