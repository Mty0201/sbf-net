---
phase: 09-phase-7-full-training-results-analysis-and-tuning
plan: 02
subsystem: analysis
tags: [training-analysis, mIoU-comparison, loss-balance, tuning-variants, per-class-iou]

requires:
  - phase: 09-phase-7-full-training-results-analysis-and-tuning
    provides: Log parsing script and CSV extraction (Plan 01)
provides:
  - Structured analysis report comparing active route vs support-only baseline
  - Identification of support loss dominance as critical problem
  - Four concrete tuning config variant proposals (support downweight, focus amplification, extended training, soft masking)
affects: [follow-on tuning phase, config variant experiments]

tech-stack:
  added: []
  patterns: [data-grounded analysis with CSV-sourced values, per-class comparison at respective best epochs]

key-files:
  created:
    - .planning/phases/09-phase-7-full-training-results-analysis-and-tuning/09-ANALYSIS.md
  modified: []

key-decisions:
  - "Support loss dominance (70%+ of total) identified as the critical problem -- rebalancing via support_loss_weight reduction is the top tuning priority"
  - "Per-class analysis reveals balustrade (-12.66pp) and advboard (-7.61pp) as the main regression drivers, likely caused by gradient pollution from support BCE"
  - "Active route trained for only 100 epochs vs baseline 300 -- training duration is a confound that must be controlled in the next experiment"
  - "Variant execution order: A (support downweight) > C (extended training control) > B (focus amplification) > D (soft masking)"

patterns-established:
  - "Analysis report structure: executive summary, overall comparison, per-class breakdown, boundary metrics, training dynamics, problem identification, tuning recommendations, next steps"

requirements-completed: [ANALYSIS-03, ANALYSIS-04]

duration: 4min
completed: 2026-04-03
---

# Phase 09 Plan 02: Full Training Results Analysis Summary

**Data-grounded comparison of active route (best mIoU 0.7265) vs support-only baseline (0.7457), identifying support loss dominance as the critical problem and proposing four concrete tuning config variants**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-03T12:57:55Z
- **Completed:** 2026-04-03T13:02:29Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Produced 09-ANALYSIS.md (286 lines) with 7 major sections covering overall mIoU comparison, per-class breakdown, boundary metrics, training dynamics, problem identification, and tuning recommendations
- Identified support loss dominance (70%+ of total loss) as the critical problem driving the 1.92pp mIoU gap
- Per-class analysis found active route gains on boundary-sensitive classes (balcony +6.58pp, eave +0.99pp) but large regressions on balustrade (-12.66pp) and advboard (-7.61pp)
- Proposed 4 concrete tuning variants (A: support downweight, B: support downweight + focus amplification, C: extended training, D: soft masking) with specific config parameter values

## Task Commits

Each task was committed atomically:

1. **Task 1: Run CSV extraction on both logs** - No commit (re-ran existing script; CSV outputs are gitignored)
2. **Task 2: Write analysis report** - `919f0dc` (feat)

## Files Created/Modified
- `.planning/phases/09-phase-7-full-training-results-analysis-and-tuning/09-ANALYSIS.md` - Full training results analysis report with comparison tables, problem identification, and tuning recommendations

## Decisions Made
- Support loss dominance identified as critical problem: at epoch 100, support BCE accounts for 73.7% of total loss while semantic CE (the primary objective) is only 15.7%
- Variant A (support_loss_weight=0.1) is top priority as a config-only fix that directly addresses the loss balance inversion
- Extended training (Variant C) is a necessary control to separate training duration effects from route design effects
- Soft masking (Variant D, Phase 8 Direction 1) requires code changes and should follow the config-only variants

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Analysis report ready as primary input for a follow-on tuning phase
- Four tuning variants defined with concrete config parameters and prioritized execution order
- Decision criteria established: beat 0.7457 mIoU, improve boundary classes, no bulk class regression >2pp

---
*Phase: 09-phase-7-full-training-results-analysis-and-tuning*
*Completed: 2026-04-03*

## Self-Check: PASSED
