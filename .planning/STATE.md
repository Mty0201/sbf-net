---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: semantic-first boundary supervision reboot
status: "BCE lower-bound problem diagnosed on continuous support target. Replaced with Focal MSE + Dice (CR-H). 2-epoch validation confirms healthy training: no collapse, support learning, aux gradient <10% of total."
stopped_at: "CR-H (FocalMSEBoundaryLoss) implemented and validated. BCE on continuous Gaussian target has irreducible entropy lower bound (~0.2), causing persistent noise gradient that competes with semantic loss in late training. MSE+Dice combo: MSE provides stable early gradients (lower bound=0), Dice handles imbalance and prevents all-zero collapse. 2-epoch test: val_mIoU=0.579, aux_prob_boundary_mean rising (0.19→0.24), loss_aux_weighted ~9% of total. CR-G (BCE, aux_weight=1) confirmed harmful: best mIoU=0.7240 < CR-A 0.7336, aux occupied 56% of total loss. CR-G aux_weight=0.3 running on real environment. Next: compare CR-G(0.3) vs CR-H full 100-epoch results."
last_updated: "2026-04-09T21:10:00Z"
last_activity: 2026-04-09
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
Status: Complete. All code implemented. CR-G (soft boundary + pos_weight) training in progress.
Last activity: 2026-04-09

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
- **[2026-04-09]** Trainer refactored: dynamic metric dispatch replaces 4 layers of hardcoded branches (~314 lines removed). Adding new losses now requires zero trainer changes. Smoke-validated all 7 pipelines (CR-A through CR-G).
- **[2026-04-09]** CR-G (soft boundary loss) analysis:
  - Initial run (aux_weight=0.3, no pos_weight): edge branch collapsed to trivial all-zero solution by epoch 2 due to extreme positive/negative imbalance (~2% boundary points).
  - Raising aux_weight to 1.0 did not help — scales entire BCE but doesn't change internal positive/negative gradient ratio.
  - Fix: per-batch `pos_weight = sqrt(neg_ratio / pos_ratio)` ≈ 8, rebalances BCE gradient. Prevents all-zero collapse.
  - Key insight: pos_weight solves the training-initialization problem. Once converged, non-boundary points (target ≈ 0) naturally have near-zero loss and stop contributing gradient. The edge branch then only produces gradient at semantic transition zones — acting as a "boundary highlighter" for the backbone.
  - Epoch 5 validation: edge branch learning confirmed (support_cover 0.15-0.43, aux_prob_boundary_mean separating from aux_prob_mean).
  - Note: current CR-G run uses aux_weight=1 (manually changed in training env, not in git config which still says 0.3). Epoch 14 shows loss_semantic=0.349 vs loss_aux_weighted=0.228 — aux approaching parity with semantic. May need rerun at 0.3 if mIoU underperforms.
- **[2026-04-09]** Module g redesigned through extensive discussion. Key design decisions:
  - **Unified philosophy:** support branch = feature-level "reminder" (detection vs GT). g module = prediction-level "correction" (from seg_logits, gradient nudges backbone).
  - **CR-E removed:** g's task depends on support branch existing; standalone offset-only has no purpose.
  - **Support BCE upgraded to CR-G style** (pos_weight) in SerialDerivationLoss.
  - **v1 (consistency MSE):** g outputs scalar, MSE vs detach(sigmoid(support_pred)). Result: loss_consistency ≈ 0.01 from start — task too easy, g provides zero useful gradient. val_mIoU=0.54 (same as without g).
  - **v2 (tolerant cosine offset + local consistency):** g outputs 3D offset. Cosine direction loss on ~2% valid points + same-side patch consistency.
  - **v2 retest (fair conditions):** Original 1-epoch test was unfair (no full LR cycle). Retest with total_epoch=20/eval_epoch=1 (complete OneCycleLR): val_mIoU=0.5529 (not 0.12). Direction loss still stuck at ~0.98 (no learning), but offset is neutral — not harmful, just dead weight like v1.
  - **Commit:** 1935ab5. Files: heads.py (BoundaryConsistencyModule), serial_derivation_model.py, serial_derivation_loss.py.
- **[2026-04-09]** BCE lower-bound problem diagnosed on continuous support target:
  - **CR-G (BCE, aux_weight=1) full run to epoch 75:** best val_mIoU=0.7240 (epoch 57) < CR-A 0.7336. Loss_aux stuck at ~0.2 (irreducible BCE entropy on continuous Gaussian target). After epoch 57, mIoU oscillates/declines — aux noise gradient (56% of total loss) interferes with semantic fine-tuning.
  - **Root cause:** BCE on continuous target t∈(0,1) has lower bound H(t) = -[t·log(t) + (1-t)·log(1-t)] > 0. Even perfect prediction produces non-zero loss and persistent gradient. With pos_weight≈7, this noise is amplified. CR-A's semantic loss reaches 0.088 by epoch 75, but CR-G's loss_aux_weighted=0.19 keeps injecting noise.
  - **BFANet comparison:** BFANet uses hard 0/1 labels → BCE lower bound = 0, gradient vanishes when learned. Plus 10x semantic upweighting at boundary points (no aux loss competition). Fundamentally different design.
  - **Solution: CR-H (FocalMSEBoundaryLoss)** — MSE + soft Dice replaces BCE:
    - MSE: lower bound = 0, stable early gradients, focal weighting (pos_alpha=9) for imbalance
    - Dice: global overlap, naturally immune to imbalance, prevents all-zero collapse
    - MSE dominates early (stable per-point gradients), Dice refines late (after MSE vanishes)
    - dice_weight=0.25 prevents Dice from dominating early (initial loss_dice≈0.96 >> loss_mse≈0.25)
  - **CR-H 2-epoch validation:** val_mIoU=0.579, aux_prob_boundary_mean rising (0.19→0.24), loss_aux_weighted ~9% of total. No collapse, healthy learning.
  - CR-G (aux_weight=0.3) running on real environment for comparison.

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
