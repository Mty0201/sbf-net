---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: verifying
stopped_at: Completed 11-02-PLAN.md
last_updated: "2026-04-04T13:51:05.833Z"
last_activity: 2026-04-04
progress:
  total_phases: 7
  completed_phases: 7
  total_plans: 16
  completed_plans: 16
  percent: 50
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-02)

**Core value:** Semantic segmentation remains the primary objective, and any boundary-aware supervision must improve boundary-region semantic quality without dragging the semantic branch into explicit geometric-field learning.
**Current focus:** Phase 11 — boundary-metrics-fix-and-focus-tuning

## Current Position

Phase: 11 (boundary-metrics-fix-and-focus-tuning) — EXECUTING
Plan: 2 of 2
Status: Phase complete — ready for verification
Last activity: 2026-04-04

Progress: [#####-----] 50%

## Recent Context

- Archived milestone `v1.0` to `.planning/milestones/v1.0-ROADMAP.md` and `.planning/milestones/v1.0-REQUIREMENTS.md`
- Preserved milestone audit in `.planning/v1.0-MILESTONE-AUDIT.md`
- Collapsed live roadmap to milestone-level history so the next milestone starts with a clean planning surface
- New active direction: semantic-first boundary supervision pivot away from direct `support` / `axis-side` field supervision
- Phase 6 established support-only as the strongest current reference baseline and support-shape as weaker side evidence only
- Phase 6 defined the `support-guided semantic focus route` as the support-centric candidate route for Phase 7 implementation

## Decisions

- Canonical SBF facts and training guardrails live under `docs/canonical/`
- GSD and local `.planning/` are now the default workflow entry for this repository
- Retained wrapper docs are compatibility-only redirect surfaces, not an active control plane
- Legacy workflow scaffolding now lives under `docs/archive/workflow-legacy/` and only minimal compatibility stubs remain in active paths
- Archived milestone history belongs under `.planning/milestones/`, while live `.planning/ROADMAP.md` remains small and milestone-scoped.
- The active SBF mainline is no longer `support + axis + side`; new milestone work must stay semantic-first and avoid explicit geometric-field supervision as the main target.
- [Phase 07]: SupportHead uses stem+linear pattern with single output channel; SharedBackboneSemanticSupportModel includes adapter pattern for future flexibility
- [Phase 07]: Focus weighting uses ground-truth support_gt to avoid prediction feedback loops; CE overlap between loss_semantic and loss_focus is intentional additive boundary emphasis
- [Phase 07]: support_pred checked before edge_pred in trainer dispatch for active route; train config uses weight=None for train-from-scratch
- [Phase 07]: Three-category config distinction (stable entry, reference baseline, active route) applied consistently across all canonical docs
- [Phase 07]: All docs explicitly note validation is pending Phase 8 to prevent premature claims
- [Phase 08]: Focus activation check uses support_gt > 0.2 threshold matching D-06 specification
- [Phase 08]: Script prints explicit evidence boundary disclaimer per COMP-04
- [Phase 08]: Evidence boundary explicit in all three canonical docs per D-12 and COMP-04
- [Phase 08]: Experiment directions framed as questions per D-11; soft masking ablation is top priority per D-07
- [Phase 09]: Train result backfill pattern: rows created at Val result time, train metrics filled when Train result appears afterward
- [Phase 09]: Support loss dominance (70%+ of total) identified as critical problem; rebalancing via support_loss_weight reduction is top tuning priority
- [Phase 09]: Tuning variant priority: A (support downweight) > C (extended training) > B (focus amplification) > D (soft masking)
- [Phase 10]: Single loss class with focus_mode flag (none/lovasz) instead of two separate classes
- [Phase 10]: Evaluator reports support_reg_error replacing support_bce for consistency with new loss
- [Phase 10]: Both variant configs inherit model/optimizer/scheduler/data from active route verbatim; only loss dict and work_dir differ
- [Phase 11]: support_reg_error key used instead of support_bce to match redesigned evaluator; log parser detects redesigned runs by loss_support_reg= in Train result lines
- [Phase 11]: focus_weight=0.15 keeps Lovasz gradient ~1.07x semantic CE at convergence; 300 eval epochs via total_epoch=6000 and eval_epoch=300 matches baseline duration

## Blockers / Concerns

- No active blockers.
- Accepted debt from `v1.0`: minor canonical-doc provenance drift and missing `*-VALIDATION.md` artifacts.

## Roadmap Evolution

- Phase 9 added: Phase 7 full training results analysis and tuning
- Phase 10 added: Loss redesign — fix support supervision and boundary focus
- Phase 11 added: Boundary metrics fix and focus tuning

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files | Recorded |
|-------|------|----------|-------|-------|----------|

## Session Continuity

Last session: 2026-04-04T13:51:05.830Z
Stopped at: Completed 11-02-PLAN.md
Resume file: None
| Phase 07 P01 | 1min | 1 tasks | 4 files |
| Phase 07 P02 | 3min | 2 tasks | 4 files |
| Phase 07 P03 | 4min | 2 tasks | 2 files |
| Phase 07 P04 | 4min | 2 tasks | 6 files |
| Phase 08 P01 | 2min | 2 tasks | 2 files |
| Phase 08 P02 | 239s | 2 tasks | 3 files |
| Phase 09 P01 | 3min | 1 tasks | 1 files |
| Phase 09 P02 | 4min | 2 tasks | 1 files |
| Phase 10 P01 | 4min | 2 tasks | 4 files |
| Phase 10 P02 | 1min | 2 tasks | 2 files |
| Phase 11 P01 | 3min | 2 tasks | 2 files |
| Phase 11 P02 | 1min | 1 tasks | 1 files |
