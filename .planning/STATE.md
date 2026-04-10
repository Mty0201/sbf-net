---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: semantic-first boundary supervision reboot
status: "CR-L debugging: boundary_threshold 0.9→0.5 + pos_weight 5→1 after root-cause analysis (positive ratio too sparse at 0.9, combined weighting too heavy). Smoke test showing healthy separation."
stopped_at: "CR-L parameter sweep: threshold=0.5, pos_weight=1, sample_weight_scale=9, local Dice on support>0. Support distribution verified from raw data (s>0.5 = 3.32% raw, s>0.9 = 0.62% raw). 4-epoch smoke at threshold=0.5 + pos_weight=5 showed healthy separation (dice 0.28, ppos-pneg gap growing). Reducing pos_weight to 1 since sample_weight_scale=9 already handles rebalancing. Next: full smoke test → full training."
last_updated: "2026-04-10T05:30:00Z"
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
- **[2026-04-10]** CR-K and CR-L implemented: BFANet-style binary BCE instead of continuous regression. CR-K = GT support weighting only (no aux, no boundary head) = CR-I ablation baseline. CR-L = support-weighted binary BCE + local Dice.
- **[2026-04-10]** CR-L root cause analysis: initial config (threshold=0.9, pos_weight=5, sample_weight_scale=9) had boundary head collapse (prob_pos=0.075, dice_score=0.05). Diagnosed via support distribution: s>0.9 = 0.62% raw → ~0.35% after voxelize (grid_size=0.06), far below what BCE+Dice can learn. Voxel (6cm) > support σ (2cm) caps achievable positive ratio.
- **[2026-04-10]** CR-L parameter fix: `boundary_threshold 0.9→0.5` (positive ratio 2% raw → clean Dice math), `pos_weight 5→1` (sample_weight_scale=9 already handles class-level rebalancing, combined 45x was too heavy). Local Dice kept on `support>0` region. 4-epoch smoke at threshold=0.5 + pos_weight=5 showed dice=0.28 and prob_pos/neg separating (gap 0.005→0.063 in 4 epochs). Next smoke: pos_weight=1.

## Experiment Evolution Summary

| Experiment | Loss Design | Aux Target | Result |
|------------|------------|------------|--------|
| CR-A | CE + Lovasz (semantic only) | — | **0.7336** (baseline) |
| CR-B | CE + Lovasz + SmoothL1/Tversky | continuous support | 0.7184 (net negative) |
| CR-G | CE + Lovasz + BCE (pos_weight) | continuous support | 0.7240 (BCE lower bound problem) |
| CR-H | CE + Lovasz + Focal MSE + Dice | continuous support | Running (positive signals) |
| CR-I | **support-weighted** CE + Lovasz + Focal MSE + Dice | continuous support | 1-epoch healthy, awaiting full |
| CR-J | CR-I loss + g v3 (boundary→semantic gating) | continuous support | Implemented, awaiting training |
| CR-K | GT support-weighted CE + Lovasz (no aux, no head) | — | Implemented, CR-I ablation |
| CR-L | support-weighted binary BCE + local Dice + GT-weighted CE | **binary (s>0.5)** | threshold=0.5, pos_weight=1 — smoke running |

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
- [v2.0] CR-L binary threshold = 0.5 (d < 2.35cm ≈ voxel radius), positive ratio ~2% after voxel (3.32% raw, verified from 10-scene sample)
- [v2.0] CR-L pos_weight = 1: sample_weight_scale=9 already handles class imbalance; combined 45x was too heavy
- [v2.0] Grid voxel size (6cm) caps achievable positive ratio — threshold cannot be raised beyond 0.5 without data pre-processing changes

## Blockers / Concerns

- **[ACTIVE]** Awaiting CR-H/I/J/K/L full training results on real environment. Success criterion: mIoU ≥ CR-A (0.7336).
- **[ACTIVE]** CR-L smoke validation at threshold=0.5 + pos_weight=1 in progress.

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
