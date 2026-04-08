---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: semantic-first boundary supervision reboot
status: Phase 6 complete — serial derivation + 4 experiment configs ready for training
stopped_at: Phase 6 complete, committed and pushed (8aeb4e1)
last_updated: "2026-04-08T01:00:00Z"
last_activity: 2026-04-08
progress:
  total_phases: 7
  completed_phases: 5
  total_plans: 2
  completed_plans: 2
  percent: 85
---

# Project State

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-04-06)

**Core value:** Semantic segmentation remains the primary objective, and any boundary-aware supervision must improve boundary-region semantic quality without dragging the semantic branch into explicit geometric-field learning.
**Current focus:** v2.0 Phase 5 — Part 1 boundary proximity cue experiment (CR-C). Route redesign complete.

## Current Position

Phase: 6 — Serial derivation module g (boundary offset from semantic logits) ✅
Plan: `.planning/phases/06-serial-derivation-module/06-01-PLAN.md`
Status: Complete. All code implemented, BCE weight bug fixed, 4 new experiment configs (CR-C/D/E/F) smoke-validated and pushed (8aeb4e1). Ready for full training runs.
Last activity: 2026-04-08

## Recent Context

- **[2026-04-06]** Milestone v2.0 kicked off — semantic-first boundary supervision reboot.
- Phase numbering reset to 1 per `--reset-phase-numbers` flag.
- 6 phases planned: defect repair → experiment execution → analysis → direction decision → refinement (conditional) → canonical update.
- 100 epochs for CR-A and CR-B (seed=38873367).
- **[2026-04-06]** Phase 1 complete. All 3 integration defects fixed (INT-01, INT-02, INT-03). Both pipelines smoke-validated.
- **[2026-04-07]** Phase 2 complete. Both clean-reset experiments ran to 100 epochs.
  - CR-A (semantic-only): best val_mIoU = 0.7336 (epoch 70)
  - CR-B (support-only): best val_mIoU = 0.7184 (epoch 57)
  - Logs: `outputs/clean_reset_s38873367/{semantic_only,support_only}/train.log`
- **[2026-04-07]** Phase 3 complete. Evidence analysis written to `docs/canonical/clean_reset_analysis.md`.
  - CR-A outperforms CR-B by +1.52pp (best) and +1.19pp (steady-state).
  - CR-A wins 6/8 classes at epoch 100. Current support implementation is a net negative.
  - Diagnostic analysis: failure is implementation/coupling/optimization, not conceptual.
- **[2026-04-07]** Phase 4 complete. Route redesign established via four canonical documents:
  - `docs/canonical/clean_reset_analysis.md` — raw CR-A vs CR-B comparison
  - `docs/canonical/clean_reset_diagnostic.md` — 8 ranked failure hypotheses (gradient competition, missing coupling, disabled focus_mode)
  - `docs/canonical/clean_reset_mechanism_analysis.md` — binary edge classification produces aligned gradient; continuous Gaussian regression produces competing gradient. Target alignment is the strongest explanatory factor.
  - `docs/canonical/route_redesign_discussion.md` — reinterpret `valid + support` as boundary proximity cue. Part 1: confidence-weighted BCE (Option L1). Part 2: geometric field deferred.
  - Old DIR-02 (pivot away) framing superseded. Concept not invalidated; implementation redesigned.
  - New route: Part 1 validates boundary proximity cue (CR-C experiment). Part 2 adds geometric field only if Part 1 succeeds.
- **[2026-04-08]** Phase 6 implemented. Serial derivation module g: derives 3D boundary offset field from semantic logits via KNN local aggregation.
  - New files: `SerialDerivationModel`, `SerialDerivationLoss`, `BoundaryOffsetModule`, CR-D config
  - Edge representation simplified: (dir, dist, support) 5-dim → (offset, support) 4-dim. offset = dir × dist.
  - Smoke validation: all 4 pipelines (CR-A/B/C/D) pass end-to-end.
  - Discussion + literature survey at `docs/canonical/part2_serial_derivation_discussion.md` §7.

## Decisions

- Canonical SBF facts and training guardrails live under `docs/canonical/`
- GSD and local `.planning/` are now the default workflow entry for this repository
- The active SBF mainline is no longer `support + axis + side`; new work must stay semantic-first
- Retract flawed baseline before further tuning — clean-reset workstream re-establishes comparison with fixed seed
- [v2.0] 100 epochs for clean-reset experiments (user preference over 150)
- [v2.0] Route redesign: reinterpret support as boundary proximity cue (confidence-weighted BCE), not geometric regression. Part 1 validates, Part 2 conditional on Part 1 success.
- [v2.0] The old DIR-02 "pivot away" framing is superseded. The concept is not invalidated — the implementation is redesigned.

## Blockers / Concerns

- ~~**[RESOLVED]** CR-B integration defects (INT-01, INT-02, INT-03) — fixed in Phase 1, smoke-validated~~
- ~~**[RESOLVED]** Support supervision net negative under current implementation~~ — Direction decision complete. Route redesigned: boundary proximity cue (confidence-weighted BCE) replaces geometric regression. See `docs/canonical/route_redesign_discussion.md`.

## Workstream Archival

- **[2026-04-08]** `edge-data-quality-repair` workstream archived. Phases 1-5 complete (diagnosis → refactor → algorithm redesign → density-adaptive fix). Pipeline consolidated and committed (124348e). Phases 6-8 (NET-02/NET-03) deferred.

## Roadmap Evolution

- Milestone v2.0 kicked off 2026-04-06. Originally 6 phases, now 7 after route redesign.
- Phase 4 (direction decision) complete — route redesign, not pivot.
- Phase 5 (Part 1: boundary proximity cue experiment) is the next active phase.
- Phase 6 (Part 2: geometric field extension) is conditional on Phase 5 success (CR-C mIoU ≥ 0.7336).
- Phase 7 (canonical update + milestone close) depends on Phase 5 or Phase 6.

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files | Recorded |
|-------|------|----------|-------|-------|----------|

## Session Continuity

Last session: 2026-04-08
Stopped at: Phase 6 complete. Serial derivation module g + 4 experiment configs (CR-C/D/E/F) implemented, smoke-validated, pushed (8aeb4e1).
Resume file: None
Next action: Run CR-C/D/E/F training experiments, collect results, compare against CR-A baseline (0.7336 mIoU). Then Phase 7 (canonical update + milestone close).
