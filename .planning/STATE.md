---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: semantic-first boundary supervision reboot
status: "Phase 5 post-hoc log analysis: CR-L vs CR-A gap (0.7251 vs 0.7336, 0.85pp) is WITHIN seed noise — both val_mIoU curves oscillate by ~0.05-0.07 after ep25. CR-L train_loss_semantic sits 0.15-0.7 ABOVE CR-A for the entire run — aux is choking semantic from ep1, not ep30. 'Zombie aux' hypothesis downgraded: the real symptom is structural gradient competition from step 0, not late-training saturation. CR-L still running (ep72/100) — do not intervene. CR-O (SoftWeightedSemanticLoss) added as minimal smooth extension of CR-A: w = 1 + s*9 full-continuous soft band, no aux head, no boundary branch."
stopped_at: "CR-O implemented (loss + config + smoke). Queue after CR-L completes. Do not intervene CR-L run."
last_updated: "2026-04-10T22:30:00Z"
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
**Current focus:** v2.0 Phase 5 — boundary proximity cue experiments. Active matrix narrowed to CR-K (pending ablation), CR-L (full train running, positive signals), CR-M (committed, awaiting queue). CR-H failed (net negative, matches CR-B). CR-J dropped (g iterated from v3 to v4, superseded by CR-M).

## Current Position

Phase: 5 — Boundary proximity cue experiments (CR-K/L/M active; CR-H/J closed out)
Plan: `.planning/phases/06-serial-derivation-module/06-01-PLAN.md`
Status: CR-L full train running in real environment with positive signals. CR-M committed and awaiting queue. CR-K retained as pending ablation. CR-H verified failed. CR-J dropped (superseded by CR-M g v4).
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
- **[2026-04-10]** CR-M implemented end-to-end per plan 05-02. New files: `project/models/gv4.py` (CrossStreamFusionAttention, K=patch_size fusion-query cross-stream attention), `project/models/boundary_gated_v4_model.py` (BoundaryGatedSemanticModelV4 with v1 CR-L heads + g v4 + v2 heads cloned from v1), `project/losses/dual_supervision_boundary_binary_loss.py` (wrapper runs BoundaryBinaryLoss on v1 and v2, sums totals, v1_/v2_ prefixed keys), `configs/.../clean_reset_gated_v4_model.py` + `boundary_gated_v4_train.py`, `scripts/train/check_cr_m_smoke.py`. Modified: `project/trainer/trainer.py` — additive kwargs forwarding of `seg_logits_v2`/`support_pred_v2` in `_build_loss_inputs` / `_build_eval_inputs`. Smoke (46.2M params, fusion 58k = 0.13%): 6.1 step-0 equivalence `max|v1-v2|=0` exact, 6.2 two-step gradient flow (step1 only `out_proj_*`+v2_head receive grad because zero-init kills upstream, step2 after SGD update full g v4 path gradient-live), 6.3 wrapper key prefixing + summation check, 6.4 trainer `_build_loss_inputs` forwards v2 kwargs through to the wrapper loss, 6.5 CR-L no-regression on SharedBackboneSemanticSupportModel + BoundaryBinaryLoss. Committed as 3fdfbd1; ready for real-env submission.
- **[2026-04-10]** Active experiment matrix culled and updated:
  - **CR-H verified failed** — full training on real environment matched CR-B (net negative vs CR-A 0.7336). Focal MSE+Dice on continuous support does not recover the gap; BCE→MSE fix alone is insufficient when target stays continuous.
  - **CR-J dropped** — g architecture iterated from v3 (per-channel gate) to v4 (cross-stream fusion attention) under CR-M. Running CR-J (g v3) is no longer informative; its role is subsumed by CR-M.
  - **CR-K retained** — pending ablation to isolate whether the GT-only boundary-region CE upweight on its own moves mIoU without any aux or boundary head.
  - **CR-L full train running** on real environment with positive signals.
  - **CR-M committed (3fdfbd1)**, awaiting real-env queue.
- **[2026-04-10 ep52 mid-train diagnosis]** CR-L at ep52/100 (log: `outputs/clean_reset_s38873367/boundary_binary/train.log`):
  - **loss_dice is NOT blocked** — epoch-mean trajectory is single-monotone decreasing ep1→ep52: 0.778→0.693→0.661→0.625→0.603→0.582. `prob_pos_mean` 0.249→0.575, `prob_neg_mean` 0.186→0.046. `loss_bce` 0.486→0.268. Dice gradient `d(dice)/d(p_pos) ≈ 0.09` in current regime — the slope geometry makes it *look* stalled at the single-batch level while the epoch average is cleanly descending.
  - **Real concern: early overfitting + val ceiling.** Peak `val_mIoU = 0.7169 at ep31`. Ep32-52 val oscillates 0.69–0.72 while train loss continues to decline 22% (loss_semantic 0.371→0.292). CR-L peak 0.7169 is only +0.002 above CR-B 0.7184 — current trajectory suggests CR-L will ceiling at CR-B level, -1.67pp below CR-A 0.7336.
  - **Decision: do not intervene, let CR-L run to ep100.** Full curve is needed as the comparison baseline for CR-M dual-stream fusion. Early stop loses diagnostic signal.
  - **Post-CR-L action plan:**
    1. If CR-L best ≥ 0.72 but < 0.7336 → CR-M is the primary next experiment (fusion may close the 1–2pp gap).
    2. If CR-L best stays at 0.71–0.72 matching CR-B → single-stream boundary-aux route has a structural ceiling under the current voxel/threshold geometry. CR-M becomes the last architectural lever; if CR-M also fails, consider voxel grid reduction (6cm→4cm) to break the physical constraint, or fall back to CR-K (pure CE-upweight, no aux head).
    3. **CR-N — pure BFANet control** (added 2026-04-10, refined): faithful BFANet reproduction. Loss structure:
       - **Semantic branch**: `CE(semantic_logits, semantic_gt)` with per-voxel weight. Boundary-positive voxels (`support > 0.5`, hard mask) get weight = **10**, all others = **1**. This is BFANet's 10× semantic upweight on the hard boundary region, distinct from CR-L's `sample_weight_scale=9` on continuous support.
       - **Boundary branch**: `BCE(boundary_logits, boundary_target) + Dice(boundary_logits, boundary_target)`, **unweighted** (`pos_weight=1`). Dice is **global** (whole-scene binary Dice, not local-to-support region). `boundary_target = support > 0.5` hard threshold.
       - **Total**: `L = L_sem + λ · L_boundary`, λ inherits CR-L's current value.
       - **Purpose**: isolate CR-L's deviations from BFANet — (a) `sample_weight_scale=9` on continuous support vs BFANet's 10× on hard mask; (b) local Dice vs global Dice. CR-N is the apples-to-apples BFANet baseline. Queues after CR-M.
- **[2026-04-10 post-hoc log analysis]** Parsed CR-L ep1-72 and CR-A ep1-100 epoch-mean train/val loss curves (`/tmp/cr_l_epochs.csv`, `/tmp/cr_a_epochs.csv`; plot at `.planning/phases/05-boundary-proximity-cue-experiment/cr_l_vs_cr_a_diagnosis.png`). Four findings:
  1. **CR-L train_loss_semantic is 0.15-0.7 higher than CR-A for the ENTIRE run** (ep1: 1.57 vs 0.85, ep70: 0.25 vs 0.09). The two curves never cross. Aux is choking semantic from step 0, not ep30. The "zombie aux ep30+" hypothesis is downgraded — the real symptom is **structural gradient competition from ep1**.
  2. **val_loss_semantic curves are indistinguishable** between CR-L and CR-A (both ~0.55-0.65 oscillating from ep10 onward). Aux pays 0.16-0.72 train-side semantic cost for **zero** val-side regularization benefit.
  3. **val_mIoU 0.0085 gap is within noise floor.** Both CR-L (ep15+) and CR-A (ep25+) oscillate with amplitude ~0.05-0.07 — 5-8× the 0.0085 gap. "CR-L 0.7251 vs CR-A 0.7336" is **not statistically significant** in single-seed comparison. Implication: the relative ordering of ALL v2.0 single-seed results (CR-B 0.7184, CR-G 0.7240, CR-L 0.7251, CR-A 0.7336) may largely be noise.
  4. **CR-L aux descent rate collapse happens at ep10, not ep30** (per memory). Ep1-10 loss_aux drops 0.27, ep10-30 drops 0.08, ep30-72 drops 0.10. Classic fast-early + long slow tail. Slow tail is 70% of training.
- **[2026-04-10] CR-O added to experiment matrix.** Minimal smooth extension of CR-A: same semantic-only model, `SoftWeightedSemanticLoss` = CE · (1 + s·9) + Lovasz, fully continuous soft weighting across [0,1], no truncation, no aux head. Tests whether continuous boundary-proximity weighting alone moves val_mIoU above CR-A within seed noise. Files: `project/losses/soft_weighted_semantic_loss.py`, `configs/semantic_boundary/clean_reset_s38873367/soft_weighted_semantic_train.py`. Smoke-validated: `mean_point_weight ≈ 5.43` (analytical uniform[0,1] expectation 5.5), grad finite. Queued after CR-L completes.

## Experiment Evolution Summary

| Experiment | Loss Design | Aux Target | Result |
|------------|------------|------------|--------|
| CR-A | CE + Lovasz (semantic only) | — | **0.7336** (baseline) |
| CR-B | CE + Lovasz + SmoothL1/Tversky | continuous support | 0.7184 (net negative) |
| CR-G | CE + Lovasz + BCE (pos_weight) | continuous support | 0.7240 (BCE lower bound problem) |
| CR-H | CE + Lovasz + Focal MSE + Dice | continuous support | **Failed** — matches CR-B, net negative |
| CR-I | **support-weighted** CE + Lovasz + Focal MSE + Dice | continuous support | 1-epoch healthy, awaiting full |
| CR-J | CR-I loss + g v3 (boundary→semantic gating) | continuous support | **Dropped** — g iterated to v4, superseded by CR-M |
| CR-K | GT support-weighted CE + Lovasz (no aux, no head) | — | Implemented, pending ablation |
| CR-L | support-weighted binary BCE + local Dice + GT-weighted CE | **binary (s>0.5)** | **Full train running, positive signals** |
| CR-M | CR-L loss on **v1 + v2** (dual supervision) via g v4 cross-stream fusion attn (K=48) | **binary (s>0.5)** | Committed (3fdfbd1), awaiting real-env queue |
| CR-N | **Pure BFANet control**: semantic CE with **10× hard-mask upweight** (weight=10 on `s>0.5`, else 1) + unweighted BCE + **global** Dice on binary aux | **binary (s>0.5)** | Queued post-CR-M (added 2026-04-10) |
| CR-O | **Fully soft-weighted semantic**: CE · (1 + s·9) + Lovasz, no truncation, no aux, no boundary head. Semantic-only model (same as CR-A). | — | Implemented + smoke-validated (added 2026-04-10); queued post-CR-L |

## Decisions

- Canonical SBF facts and training guardrails live under `docs/canonical/`
- GSD and local `.planning/` are now the default workflow entry for this repository
- The active SBF mainline is no longer `support + axis + side`; new work must stay semantic-first
- Retract flawed baseline before further tuning — clean-reset workstream re-establishes comparison with fixed seed
- [v2.0] 100 epochs for clean-reset experiments (user preference over 150)
- [v2.0] Route redesign: reinterpret support as boundary proximity cue, not geometric regression
- [v2.0] BCE on continuous target has irreducible entropy lower bound — replaced with MSE+Dice (CR-H)
- [v2.0] Semantic CE boundary upweight uses continuous support with truncation at >0.5 (CR-I)
- [v2.0] Module g redesigned from semantic→boundary (failed) to boundary→semantic gating (g v3 in CR-J, then g v4 cross-stream fusion in CR-M)
- [v2.0] CR-L binary threshold = 0.5 (d < 2.35cm ≈ voxel radius), positive ratio ~2% after voxel (3.32% raw, verified from 10-scene sample)
- [v2.0] CR-L pos_weight = 1: sample_weight_scale=9 already handles class imbalance; combined 45x was too heavy
- [v2.0] Grid voxel size (6cm) caps achievable positive ratio — threshold cannot be raised beyond 0.5 without data pre-processing changes
- [v2.0] **CR-H failed on full train** — continuous support target (even with MSE+Dice) does not recover the CR-B gap. Further continuous-target work under Part 1 is deprioritized; the active path is the BFANet-style binary threshold (CR-L/M).
- [v2.0] **CR-J dropped before training** — g v3 (per-channel gate) superseded by g v4 (cross-stream fusion attention) in CR-M. No value in running CR-J with the older gating module.

## Blockers / Concerns

- **[ACTIVE]** CR-L full training at ep52/100 on real environment. Peak `val_mIoU = 0.7169 at ep31`, currently oscillating 0.69–0.72 while train loss still declining (early overfitting). Trajectory suggests CR-L will ceiling at CR-B level 0.7184, -1.67pp below CR-A 0.7336. **Do not intervene** — let it run to ep100 for the full curve. Log: `outputs/clean_reset_s38873367/boundary_binary/train.log`.
- **[DIAGNOSED — NOT A BLOCKER]** CR-L `loss_dice` was suspected of stalling around 0.6. Verified single-monotone decrease ep1→ep52: 0.778→0.582. `prob_pos` 0.249→0.575, `prob_neg` 0.186→0.046. The apparent stall is a Dice slope-geometry artifact at the single-batch level — epoch means descend cleanly. No change needed.
- **[ACTIVE]** CR-M queued for real-environment submission after CR-L result lands.
- **[PENDING]** CR-K full training decision — run only if CR-L and CR-M results leave the CE-upweight-only ablation still interesting.
- **[CLOSED]** CR-H — verified failed (net negative, matches CR-B).
- **[CLOSED]** CR-J — dropped (g v3 superseded by g v4 in CR-M).

## Workstream Archival

- **[2026-04-08]** `edge-data-quality-repair` workstream archived. Phases 1-5 complete. Phases 6-8 deferred.

## Roadmap Evolution

- Milestone v2.0 kicked off 2026-04-06. Originally 6 phases, now 7 after route redesign.
- Phase 5 expanded: CR-C → CR-F → CR-G → CR-H → CR-I → CR-J → CR-K → CR-L → CR-M experiment evolution.
- Phase 6 (module g) iterated three times: v1/v2 failed (semantic→boundary), v3 boundary→semantic gating merged into CR-J, v4 cross-stream fusion attention active in CR-M.
- CR-H failed on full train (continuous target insufficient). CR-J dropped (g v3 superseded by g v4). Active matrix now CR-K/L/M.
- Phase 7 (canonical update + milestone close) depends on CR-L and CR-M training results.

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files | Recorded |
|-------|------|----------|-------|-------|----------|

## Session Continuity

Last session: 2026-04-10 (ep52 mid-train diagnosis)
Stopped at: CR-L at ep52/100 on real env. Peak val_mIoU=0.7169 at ep31, now in overfitting regime. loss_dice concern diagnosed and dismissed (epoch-mean descending cleanly, single-batch noise was the visual artifact). CR-M committed (3fdfbd1), awaiting real-env queue. CR-H closed (failed). CR-J closed (dropped). CR-K pending.
Resume file: `outputs/clean_reset_s38873367/boundary_binary/train.log`
Next action: (1) Wait for CR-L to finish ep100 — do not intervene. (2) Record final best val_mIoU. (3) If best ≥ 0.72 → submit CR-M to real env. (4) If best stays ≈ CR-B 0.7184 → single-stream boundary-aux has structural ceiling; CR-M is last architectural lever before falling back to CR-K or voxel grid reduction.
