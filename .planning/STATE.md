---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: semantic-first boundary supervision reboot
status: "CR-L ep100 landed: best val_mIoU 0.7251 @ ep68, ep100 0.6994 — within single-seed noise of CR-A 0.7336 (−0.85pp), single-stream aux route ceilings near CR-B level. CR-P training on real environment after BFANet 4-item alignment pass (3452189 + abe0fc8). Documentation sync in progress: STATE.md updated for CR-L result; ROADMAP.md + PROJECT.md/canonical updates queued next. Post-training test phase is broken — blocks test-split numbers, deferred."
stopped_at: "CR-L ep100 recorded. Updating ROADMAP.md (behind by CR-N/O/P) and PROJECT.md/canonical (BFANet r=0.06 truth + alignment pass not yet captured). CR-P awaiting first epoch metrics on real env. CR-N queued. Test hook bug deferred."
last_updated: "2026-04-11T18:00:00Z"
last_activity: 2026-04-11
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
**Current focus:** v2.0 Phase 5 — boundary proximity cue experiments. Active matrix: CR-K (pending ablation), CR-L (full train landed), CR-M (committed), CR-N (reset to faithful BFANet, now consumes r=0.06 precomputed mask), CR-O (soft semantic baseline), **CR-P (training on real env: full BFANet reproduction = CR-M dual-stream + PureBFANetLoss + v1:v2 = 1:1 after 2026-04-11 alignment pass)**. CR-H failed, CR-J dropped.

## Current Position

Phase: 5 — Boundary proximity cue experiments (CR-K/L/M/N/O/P active; CR-H/J closed out)
Plan: `.planning/phases/06-serial-derivation-module/06-01-PLAN.md`
Status: CR-P smoke-tested and code complete; awaiting commit + real-env queue. CR-N loss path now consumes precomputed `boundary_mask_r060.npy` (pos_weight reverted 8→1 to restore BFANet-faithful semantics). CR-M deliberately still uses `edge[:,3]>0.5` threshold — not migrated to new mask so CR-L vs CR-M stays an architectural (dual-stream) ablation. CR-O queued. CR-L full train landed.
Last activity: 2026-04-11

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
- **[2026-04-11] BFANet radius truth corrected and boundary_mask preprocessing pipeline landed.** Previous framing (`r = 3 × voxel`) was retracted — BFANet's radius on ScanNet is **r = 0.06 m absolute physical**; the 3× identity was a mink-path coincidence (`0.06 = 3 × 0.02`). Built `scripts/data/probe_boundary_radius.py` (10-chunk pilot sweep over r ∈ {0.03, 0.06, 0.09, 0.12}) and `scripts/data/generate_boundary_mask.py` (multiprocessing.Pool(8) generator). Pilot: r=0.06 landed 7/10 chunks in [5,15]% window. Full run over 264 chunks at r=0.06 m: **min 1.65%, mean 7.19%, median 6.76%, max 14.11%, 202/264 (76.5%) in [5,15]%**, all ≤15%, 25.8 s total. Visually verified on chunk 020101 (11.0% positive). Output: `/home/mty0201/data/BF_edge_chunk_npy/{training,validation}/<id>/boundary_mask_r060.npy` shape `(N, 1) uint8`. Dataset (`bf.py`), transforms (`clean_reset_data.py` InjectIndexValidKeys + Collect), trainer (`_build_loss_inputs` / `_build_eval_inputs`), and `PureBFANetLoss.forward` all now accept an optional precomputed `boundary_mask`, falling back silently to `edge[:,3]>0.5` when absent. CR-M deliberately **not** migrated — still uses `edge[:,3]>0.5` to keep the CR-L vs CR-M comparison architecture-only.
- **[2026-04-11] CR-N reset to faithful BFANet control.** Previous session had bumped `pos_weight=8` as a collapse-prevention hack; reverted to `pos_weight=1.0` to preserve the "naive BFANet at 7% positive" semantics. CR-N now consumes `boundary_mask_r060.npy`. Smoke-validated end-to-end: new mask path gives positive ratio 7.86% vs 2.31% from the legacy `edge[:,3]>0.5` fallback (exactly the delta that motivated the refactor). Gradients land on both `seg_logits` (norm 0.011) and `support_pred` (norm 0.0007), no NaN. **BFANet's naive BCE collapse at 7% positive is now an explicit test** — if CR-N's `prob_pos_mean - prob_neg_mean` stays near zero past ep5, that's a real finding, not a bug.
- **[2026-04-11] CR-P added — full BFANet reproduction.** The "one-shot" control that combines CR-M's dual-stream architecture with CR-N's BFANet-faithful loss. Files added: `project/losses/dual_supervision_pure_bfanet_loss.py` (wraps `PureBFANetLoss`, applies to v1 + v2, combines `v1_weight * L_v1 + v2_weight * L_v2`, raises if v2 outputs missing, emits v1_/v2_ prefixed metrics) and `configs/semantic_boundary/clean_reset_s38873367/pure_bfanet_v4_train.py` (reuses `clean_reset_gated_v4_model.py` + `clean_reset_data.py`, `work_dir=outputs/clean_reset_s38873367/pure_bfanet_v4`, optimizer/scheduler/seed/epochs byte-identical to CR-M/N/L). Initial weighting `v1:v2 = 0.5:1.0` was a Claude-introduced guess, superseded 2026-04-11 afternoon alignment pass (see below). Loss term breakdown (per stream): hard-mask 10× CE on `boundary_mask_r060`, unweighted BCE with pos_weight=1, global Dice over the whole scene. **CR-P smoke test PASSED**: (1) dataset emits `boundary_mask` (positive ratio 6.93% on 2-sample batch, within expected 5–15%), (2) model forwards all 4 outputs (`seg_logits_v1/v2`, `support_pred_v1/v2`), (3) step-0 `max|v1-v2| == 0` exact, (4) `loss = 7.4503 = 0.5·4.9669 + 1.0·4.9669`, (5) backward grads: semantic_head_v1=2.35, support_head_v1=0.25, semantic_head_v2=4.71 (exactly 2× v1 as expected from weighting + identity outputs), support_head_v2=0.51, fusion.out_proj=0.776, fusion q/k/v projections zero (expected — zero-init residual zeros out their forward contribution, they come alive on step 1 after SGD updates `out_proj`). **Purpose of CR-P**: definitively answer "does faithful BFANet beat CR-A (0.7336) on our building-facade data?". Differences from CR-N are architectural only (dual-stream + g v4); differences from CR-M are loss-only (PureBFANetLoss + r=0.06 mask + alignment pass weighting).
- **[2026-04-11] CR-L ep100 final (log local: `outputs/clean_reset_s38873367/boundary_binary/train.log`).** Training completed normally through ep100. **Best val_mIoU = 0.7251 at ep68** (mAcc 0.8380, allAcc 0.8614). Ep100 val_mIoU = 0.6994 (late-train oscillation). **Deltas:** vs CR-A 0.7336 = **−0.0085 (−0.85pp, within 0.05–0.07 single-seed noise floor)**; vs CR-B 0.7184 = +0.0067 (+0.67pp). Per-class at best: balustrade 0.7792, balcony 0.4415, advboard 0.7803, wall 0.7489, eave 0.6372, column 0.7921, window 0.7900, clutter 0.6177. **Interpretation:** CR-L's peak trajectory (ep31 peak 0.7169 → ep68 peak 0.7251 → ep100 0.6994) confirms the earlier ep52 diagnosis: single-stream boundary-aux route ceilings at roughly CR-B level, with the CR-L→CR-A gap statistically indistinguishable from seed noise. This is the first piece of v2.0 hard data that says the aux-head architectural route alone cannot meaningfully beat semantic-only under the current voxel/threshold geometry. Post-training test phase is separately broken and deferred (not CR-L-specific; blocks CR-L test-split numbers but not the training verdict).
- **[2026-04-11 afternoon] CR-P/CR-N BFANet 4-item surface alignment pass** (commits 3452189 + abe0fc8). Line-by-line audit of BFANet `train.py:147-188` + `network/BFANet.py:131-221` against CR-P surfaced 5 structural deviations; user elected to fix 4 and accept the rest:
  - **Fix 1 — aux_weight 0.3 → 1.0** (both CR-N `pure_bfanet_train.py` and CR-P `pure_bfanet_v4_train.py`, and both loss `__init__` defaults). BFANet sums `sem_v1 + dice_v1 + margin_v1 + margin_dice_v1 + sem_v2 + dice_v2 + margin_v2 + margin_dice_v2` with no per-branch scaling; CR-P's `aux_weight=0.3` was a Claude-introduced rebalance that had no BFANet basis.
  - **Fix 2 — v1_weight 0.5 → 1.0** (`DualSupervisionPureBFANetLoss` default + CR-P config). The `v1:v2 = 0.5:1.0` "BFANet-faithful" framing was Claude's mechanism interpretation ("v2 is primary, v1 is regularizer"), not in BFANet's code. Actual BFANet weighting is exactly `1:1`.
  - **Fix 3 — CE normalization** `.sum() / valid.sum()` → `.mean()`. BFANet's `semantic_loss.mean()` normalizes by total point count including ignore points (CE at ignore points is 0 via `F.cross_entropy(ignore_index=-1, reduction="none")`). The old `valid`-based denominator slightly inflated per-point loss when ignore fraction was nonzero.
  - **Fix 4 — BCE surface form** attempted `BCEWithLogits` → `sigmoid + binary_cross_entropy`, **reverted on real env** (commit abe0fc8). `F.binary_cross_entropy` is explicitly AMP-unsafe (PyTorch raises `RuntimeError: unsafe to autocast`). Mathematically and gradient-wise `BCEWithLogitsLoss(x, y) ≡ BCELoss(sigmoid(x), y)` via log-sum-exp, so the revert is a surface-form deviation with zero numerical impact, forced by AMP compatibility.
  - **Deviations accepted (not fixed):** (a) CR-P uses Lovasz where BFANet uses multi-class semantic Dice; (b) g v4 patch attention with K=48 + zero-init residual vs BFANet's ES Attention with K=1 (BFANet's original is degenerate — softmax over 1 element = identity, so the fusion query does nothing; g v4 is a structural fix, not a faithful port); (c) margin head sigmoid is applied in the loss not in the model (preserves shared `SupportHead` for CR-L/M without forking).
  - **Smoke validation:** random-tensor unit test passed all 10 assertions across single-stream `PureBFANetLoss` (aux_weight=1 default, CE mean matches hand calc 1e-6, sigmoid+BCE ≡ BCEWithLogits 1e-6, total aggregation, backward grads) and dual wrapper `DualSupervisionPureBFANetLoss` (v1_weight=v2_weight=1 defaults, total = v1 + v2, step-0 v1==v2 exact at 1e-6 when inputs match, all 4 tensors receive grads, 13 v1_ + 13 v2_ prefixed metric keys). Post-revert re-smoke: AMP autocast fp16 path runs without RuntimeError, fp32 path numerically identical to sigmoid+BCE at 1e-6.
  - **Real env status:** initial `git pull` blocked by uncommitted drift, user did `git reset --hard origin/main + clean -fd`. CR-P launch hit `KeyError: boundary_mask` because `/home/mty0201/data/BF_edge_chunk_npy/**/boundary_mask_r060.npy` wasn't synced yet — user copied from dev machine, training now running successfully. CR-N queued to follow (inherits same aux_weight fix from `pure_bfanet_train.py`).

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
| CR-L | support-weighted binary BCE + local Dice + GT-weighted CE | **binary (s>0.5)** | **ep100 complete: best 0.7251 @ ep68**, −0.85pp vs CR-A (0.7336), +0.67pp vs CR-B (0.7184); within noise floor 0.05–0.07 |
| CR-M | CR-L loss on **v1 + v2** (dual supervision) via g v4 cross-stream fusion attn (K=48) | **binary (s>0.5)** | Committed (3fdfbd1), awaiting real-env queue |
| CR-N | **Pure BFANet control** (reset 2026-04-11): hard-mask 10× CE + unweighted BCE (`pos_weight=1`) + **global** Dice; boundary target now `boundary_mask_r060.npy` (radius search r=0.06 m absolute) | **binary (r=0.06 m)** | Reset 2026-04-11 (pos_weight 8→1); queued alongside CR-P |
| CR-O | **Fully soft-weighted semantic**: CE · (1 + s·9) + Lovasz, no truncation, no aux, no boundary head. Semantic-only model (same as CR-A). | — | Implemented + smoke-validated (added 2026-04-10); queued post-CR-L |
| **CR-P** | **Full BFANet reproduction**: CR-M's dual-stream model (`BoundaryGatedSemanticModelV4`) + `DualSupervisionPureBFANetLoss` (PureBFANetLoss on v1 and v2) with **v1:v2 = 1:1** BFANet-faithful weighting (corrected 2026-04-11 afternoon, aux_weight=1.0, CE `.mean()` over total); boundary target = `boundary_mask_r060.npy` | **binary (r=0.06 m)** | **Training on real env 2026-04-11** (smoke passed, alignment pass 3452189 + abe0fc8 applied) |

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
- [v2.0] **BFANet radius is r=0.06 m absolute** (not "3× voxel"). On ScanNet `0.06 = 3 × 0.02` is a coincidence of the mink-path voxel size; octformer uses `r=0.006` because coords are pre-scaled by /10.25. Do NOT port the 3× voxel identity to other datasets — use the absolute 6 cm. Evidence: `train.py:82-87` in weiguangzhao/BFANet@master.
- [v2.0] **CR-M deliberately kept on `edge[:,3]>0.5`** — not migrated to `boundary_mask_r060` so the CR-L vs CR-M comparison isolates architecture (single-stream vs dual-stream g v4), not boundary-target definition. CR-N and CR-P are the experiments that get the new precomputed mask.
- [v2.0] **CR-P v1:v2 = 1:1** (corrected 2026-04-11 afternoon alignment pass, commit 3452189) — BFANet's actual code sums `sem_v1 + dice_v1 + margin_v1 + margin_dice_v1 + sem_v2 + dice_v2 + margin_v2 + margin_dice_v2` with no per-branch scaling. The earlier `0.5:1.0` "v2 is primary, v1 is regularizer" framing was a Claude mechanism interpretation without BFANet code basis and has been retracted. CR-P and CR-M now share 1:1 weighting; they differ only in loss (PureBFANetLoss vs BoundaryBinaryLoss) and boundary target (r=0.06 mask vs `edge[:,3]>0.5`).

## Blockers / Concerns

- **[ACTIVE]** CR-P training on real env 2026-04-11 afternoon. BFANet alignment pass applied (aux_weight 0.3→1.0, v1:v2 0.5/1.0→1.0/1.0, CE `.mean()` over total, BCE reverted to `BCEWithLogits` for AMP safety — commits 3452189 + abe0fc8). Config: `configs/semantic_boundary/clean_reset_s38873367/pure_bfanet_v4_train.py`; work_dir: `outputs/clean_reset_s38873367/pure_bfanet_v4`. Awaiting first epoch metrics.
- **[ACTIVE]** CR-N reset to faithful BFANet (pos_weight 8→1) on 2026-04-11 and now consumes `boundary_mask_r060.npy`. Awaits real-env queue alongside CR-P. Known risk: BFANet's naive BCE (`pos_weight=1`) at our 7% positive ratio may collapse — that's an intentional test, not a bug; if `prob_pos_mean - prob_neg_mean` stays near zero past ep5 it's a real finding about BFANet's default recipe on sparse-positive data.
- **[CLOSED 2026-04-11]** CR-L full training complete. Best val_mIoU = 0.7251 @ ep68, ep100 = 0.6994. vs CR-A: −0.0085 (within single-seed noise 0.05–0.07). vs CR-B: +0.0067. Verdict: single-stream boundary-aux route ceilings near CR-B level, does not reach CR-A. Log: `outputs/clean_reset_s38873367/boundary_binary/train.log`.
- **[ACTIVE]** Post-training test phase (trainer test hook) broken — CR-L ep100 log shows `>>> Post-Training Test >>>` then truncation. Blocks getting test-split numbers for CR-L / CR-M / CR-N / CR-O / CR-P. Deferred repair; not blocking v2.0 verdict on training metrics. To be investigated separately.
- **[ACTIVE]** CR-M queued for real-environment submission after CR-L result lands.
- **[PENDING]** CR-K full training decision — run only if CR-L and CR-M results leave the CE-upweight-only ablation still interesting.
- **[CLOSED 2026-04-11]** Boundary-mask preprocessing handoff — r=0.06 m decided (pilot + full stats verified), `scripts/data/probe_boundary_radius.py` + `scripts/data/generate_boundary_mask.py` built, 264 chunks generated, dataset/transforms/trainer/loss all consume it. CR-N and CR-P now wired end-to-end.
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

Last session: 2026-04-11 (BFANet radius truth + boundary_mask preprocessing + CR-N reset + CR-P full reproduction, all code landed, CR-P smoke-passed, single atomic commit pending)
Stopped at: Smoke test complete, STATE.md updated. Remaining: phase plan artifact (task #6) + single atomic commit (task #7). CR-L full curve at ep100 still needs to be recorded separately.
Resume file: `.continue-here.md` (previous handoff, task #4 now done) + this STATE.md
Next action: (1) Decide whether CR-P goes in `.planning/phases/05-boundary-proximity-cue-experiment/05-02-PLAN.md` or a new 05-03. (2) Single atomic commit (config + code + preprocessing scripts + STATE.md + plan artifact). (3) Queue CR-N and CR-P on real env when GPU frees. (4) CR-L ep100 recording is a separate task — not blocked on CR-P.
