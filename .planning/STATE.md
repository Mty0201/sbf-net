---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: semantic-first boundary supervision reboot
status: "CR-I (boundary upweight + Focal MSE/Dice) and CR-J (CR-I + boundary→semantic gating g v3) implemented. CR-H real training showing positive signals. CR-I 1-epoch validation: dice 2.5x faster than CR-H, boundary_ce_frac 16%, healthy training."
stopped_at: "CR-I and CR-J implemented, smoke-validated, pushed. CR-H running on real training environment with positive results (fast separation, Dice improving, mIoU not degraded). CR-I 1-epoch temp test confirms boundary upweight working. Next: run CR-I and CR-J full 100-epoch training, compare against CR-A baseline (0.7336)."
last_updated: "2026-04-10T00:30:00Z"
last_activity: 2026-04-10
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
**Current focus:** v2.0 Phase 5 — boundary proximity cue experiments (CR-G/H/I/J). Phase 6 g module redesigned as boundary→semantic gating (g v3).

## Current Position

Phase: 5 — Boundary proximity cue experiments (CR-G/H/I/J)
Plan: `.planning/phases/06-serial-derivation-module/06-01-PLAN.md`
Status: CR-I and CR-J implemented. Awaiting full training results.
Last activity: 2026-04-10

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
- **[2026-04-07]** Phase 4 complete. Route redesign established via four canonical documents.
- **[2026-04-08]** Phase 6 implemented. Serial derivation module g (v1/v2): derives boundary offset from semantic logits. Both versions confirmed neutral/dead weight.
- **[2026-04-09]** Trainer refactored: dynamic metric dispatch (35b91bd). Zero trainer changes for new losses.
- **[2026-04-09]** CR-G (BCE on continuous support): best mIoU=0.7240 < CR-A 0.7336. BCE has irreducible entropy lower bound (~0.2) on continuous target.
- **[2026-04-09]** CR-H (Focal MSE + Dice): replaces BCE. MSE lower bound=0. 2-epoch validation healthy. Real training showing positive signals: fast separation, Dice improving, mIoU not degraded.
- **[2026-04-09]** CR-H analysis: MSE maintains ~0.05-0.1 (dynamic equilibrium, not dead), Dice steadily improving. MSE+Dice combo validated in real training.
- **[2026-04-09]** CR-I designed: BFANet-inspired boundary upweight on semantic CE using continuous support as weight, truncated at support>0.5 to prevent tail inflation. Aux reuses CR-H's validated Focal MSE+Dice. 1-epoch temp test: dice_score 0.27 (2.5x faster than CR-H), boundary_ce_frac 16%, healthy training.
- **[2026-04-10]** CR-J designed: CR-I loss + g v3 (BoundaryGatingModule with PTv3 patch attention). g reversed from semantic→boundary (failed) to boundary→semantic (BFANet direction). boundary_feat → patch self-attention → per-channel gate → modulates semantic_feat. No dedicated loss for g — driven entirely by semantic loss gradients through the gate. Zero-init output → initial multiplier 1.5 ≈ identity.

## Experiment Evolution Summary

| Experiment | Loss Design | Aux Target | Result |
|------------|------------|------------|--------|
| CR-A | CE + Lovasz (semantic only) | — | **0.7336** (baseline) |
| CR-B | CE + Lovasz + SmoothL1/Tversky | continuous support | 0.7184 (net negative) |
| CR-G | CE + Lovasz + BCE (pos_weight) | continuous support | 0.7240 (BCE lower bound problem) |
| CR-H | CE + Lovasz + Focal MSE + Dice | continuous support | Running (positive signals) |
| CR-I | **support-weighted** CE + Lovasz + Focal MSE + Dice | continuous support | 1-epoch healthy, awaiting full |
| CR-J | CR-I loss + g v3 (boundary→semantic gating) | continuous support | Implemented, awaiting training |

## Decisions

- Canonical SBF facts and training guardrails live under `docs/canonical/`
- GSD and local `.planning/` are now the default workflow entry for this repository
- The active SBF mainline is no longer `support + axis + side`; new work must stay semantic-first
- Retract flawed baseline before further tuning — clean-reset workstream re-establishes comparison with fixed seed
- [v2.0] 100 epochs for clean-reset experiments (user preference over 150)
- [v2.0] Route redesign: reinterpret support as boundary proximity cue, not geometric regression
- [v2.0] BCE on continuous target has irreducible entropy lower bound — replaced with MSE+Dice (CR-H)
- [v2.0] Semantic CE boundary upweight uses continuous support with truncation at >0.5 (CR-I)
- [v2.0] Module g redesigned from semantic→boundary (failed) to boundary→semantic gating (g v3, CR-J)

## Blockers / Concerns

- **[ACTIVE]** Awaiting CR-H/I/J full training results on real environment. Success criterion: mIoU ≥ CR-A (0.7336).

## Workstream Archival

- **[2026-04-08]** `edge-data-quality-repair` workstream archived. Phases 1-5 complete. Phases 6-8 deferred.

## Roadmap Evolution

- Milestone v2.0 kicked off 2026-04-06. Originally 6 phases, now 7 after route redesign.
- Phase 5 expanded: CR-C → CR-F → CR-G → CR-H → CR-I → CR-J experiment evolution.
- Phase 6 (module g) redesigned: v1/v2 failed, v3 boundary→semantic gating merged into CR-J.
- Phase 7 (canonical update + milestone close) depends on Phase 5 training results.

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files | Recorded |
|-------|------|----------|-------|-------|----------|

## Session Continuity

Last session: 2026-04-10
Stopped at: CR-I and CR-J implemented and pushed (f6bf775, fe63163). CR-H running on real training env with positive signals.
Resume file: None
Next action: Run CR-I and CR-J full training on real environment. Compare all results against CR-A baseline (0.7336). Then Phase 7 (canonical update + milestone close).
