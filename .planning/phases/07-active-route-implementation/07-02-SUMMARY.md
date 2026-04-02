---
phase: 07-active-route-implementation
plan: 02
subsystem: losses, evaluator
tags: [loss, evaluator, support-guided, semantic-focus, bce, tversky, boundary-metrics]

# Dependency graph
requires:
  - phase: 07-01
    provides: SharedBackboneSemanticSupportModel with SupportHead
provides:
  - SupportGuidedSemanticFocusLoss with three-term loss contract
  - SupportGuidedSemanticFocusEvaluator with boundary-region and support metrics
affects: [07-03, 07-04]

# Tech tracking
tech-stack:
  added: []
  patterns: [ground-truth-guided focus weighting, boundary-region metric masking, Tversky overlap on sigmoid probabilities]

key-files:
  created:
    - project/losses/support_guided_semantic_focus_loss.py
    - project/evaluator/support_guided_semantic_focus_evaluator.py
  modified:
    - project/losses/__init__.py
    - project/evaluator/__init__.py

key-decisions:
  - "Focus weighting uses ground-truth support_gt (not model prediction) to avoid feedback loops"
  - "CE overlap between loss_semantic and loss_focus is intentional additive boundary emphasis"
  - "Boundary-region evaluator threshold applies to ground-truth support_gt, not sigmoid(support_pred)"
  - "Support evaluator metrics use sigmoid probabilities and Tversky overlap, not legacy dir/dist"

patterns-established:
  - "Ground-truth guidance pattern: focus weighting derived from annotation, not prediction"
  - "Three-term loss pattern: global semantic + support BCE + support-guided focus"
  - "Boundary-region metric pattern: per-class stats masked by ground-truth support threshold"

requirements-completed: [AUX-03, COMP-03]

# Metrics
duration: 3min
completed: 2026-04-02
---

# Phase 07 Plan 02: Loss and Evaluator Summary

**Three-term support-guided semantic focus loss (CE+Lovasz, support BCE, GT-weighted focus CE) and evaluator with boundary-region mIoU, support BCE, and Tversky metrics**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-02T18:28:23Z
- **Completed:** 2026-04-02T18:31:02Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created SupportGuidedSemanticFocusLoss with fully defined three-term loss math: global semantic (CE + Lovasz), support BCE masked by valid_gt, and support-guided semantic focus with ground-truth weighting
- Created SupportGuidedSemanticFocusEvaluator reporting global semantic metrics, boundary-region semantic mIoU/mAcc, and support-specific metrics (BCE on sigmoid, Tversky overlap)
- Both registered and buildable via build_loss and build_evaluator from config
- No legacy dir/dist field metrics or edge_pred in either module

## Task Commits

Each task was committed atomically:

1. **Task 1: Create SupportGuidedSemanticFocusLoss with explicit loss contract** - `f3a0f19` (feat)
2. **Task 2: Create SupportGuidedSemanticFocusEvaluator with defined metric contract** - `7992c81` (feat)

## Files Created/Modified
- `project/losses/support_guided_semantic_focus_loss.py` - Three-term loss: global semantic, support BCE, GT-weighted semantic focus
- `project/evaluator/support_guided_semantic_focus_evaluator.py` - Evaluator with global, boundary-region, and support metrics
- `project/losses/__init__.py` - Registration of SupportGuidedSemanticFocusLoss in build_loss
- `project/evaluator/__init__.py` - Registration of SupportGuidedSemanticFocusEvaluator in build_evaluator

## Decisions Made
- Focus weighting uses ground-truth support_gt (not model prediction support_pred) to prevent feedback loops where model errors amplify themselves
- CE appearing in both loss_semantic and loss_focus is intentional: additive boundary emphasis on top of global supervision, not a replacement for Lovasz
- Boundary-region evaluator threshold (0.2) applies to ground-truth support_gt, not sigmoid(support_pred), so boundary region is defined by annotation
- Support evaluator metrics use sigmoid probabilities (not raw logits) with BCE and Tversky (alpha=0.3, beta=0.7)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all data paths are fully wired.

## Next Phase Readiness
- Loss and evaluator are ready for config YAML definition (07-03) and smoke test (07-04)
- Both consume support_pred + edge interface from SharedBackboneSemanticSupportModel (07-01)

---
*Phase: 07-active-route-implementation*
*Completed: 2026-04-02*
