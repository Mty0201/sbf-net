# Roadmap: sbf-net

## Milestones

- [x] [`v1.0` workflow migration](.planning/milestones/v1.0-ROADMAP.md) — shipped 2026-04-02, 4 phases, 12 plans, GSD plus `.planning/` cutover complete
- [x] [`v1.1` semantic-first boundary supervision pivot](.planning/milestones/v1.1-ROADMAP.md) — shipped 2026-04-06, 8 phases, 16 plans, semantic-first route implemented + loss redesigned + baseline retracted

## v2.0 — Semantic-first boundary supervision reboot

**Goal:** Establish a credible controlled comparison (semantic-only vs support-only), determine whether support supervision helps, and if so refine it — if not, document evidence and stop at a pivot decision.

**Phase numbering reset to 1 for this milestone.**

### Phase 1: Integration defect repair ✅

**Goal:** Fix the 3 confirmed integration defects in the CR-B config so both experiment configs can run cleanly.
**Requires:** FIX-01, FIX-02, FIX-03
**Depends on:** Nothing — prerequisite for all experiment work
**Status:** Complete (2026-04-06, commit 699e399)

### Phase 2: Clean-reset experiment execution ✅

**Goal:** Run CR-A (semantic-only) and CR-B (support-only) for 100 epochs each with seed=38873367 and collect results.
**Requires:** EXP-01, EXP-02, EXP-03
**Depends on:** Phase 1
**Status:** Complete (2026-04-07). CR-A best mIoU=0.7336 (ep70), CR-B best mIoU=0.7184 (ep57).

### Phase 3: Evidence analysis and comparison ✅

**Goal:** Produce a structured CR-A vs CR-B comparison with a clear verdict on support's effect.
**Requires:** ANA-01, ANA-02
**Depends on:** Phase 2
**Status:** Complete (2026-04-07). Analysis at `docs/canonical/clean_reset_analysis.md`. Current implementation is net negative (+1.52pp mIoU advantage for semantic-only). Diagnostic analysis at `docs/canonical/clean_reset_diagnostic.md` — failure attributed to implementation/coupling/optimization, not conceptual invalidity.

### Phase 4: Direction decision — route redesign ✅

**Goal:** Based on Phase 3 evidence + diagnostic analysis + mechanism analysis, determine the next research route.
**Requires:** DIR-01
**Depends on:** Phase 3
**Status:** Complete (2026-04-07). Route redesign established via four canonical documents:
- `docs/canonical/clean_reset_analysis.md` — raw CR-A vs CR-B comparison
- `docs/canonical/clean_reset_diagnostic.md` — 8 ranked failure hypotheses
- `docs/canonical/clean_reset_mechanism_analysis.md` — why binary edge helps, support regression does not
- `docs/canonical/route_redesign_discussion.md` — Part 1/Part 2 staged route

**Conclusion:** The old DIR-02 (pivot away) framing is superseded. The concept of boundary-aware dual supervision is not invalidated — the current *implementation* (continuous Gaussian regression, no coupling, raw additive loss) is what fails. The new route reinterprets `valid + support` as a boundary proximity cue / confidence-weighted boundary supervision, aligning with the DLA-Net / PTv3 reproduction mechanism that is known to work. Geometric field objectives are deferred to Part 2, conditional on Part 1 validation.

---

### — Part 1: Boundary Proximity Cue Validation (Phase 5) —

### Phase 5: Boundary proximity cue experiments (CR-C/F/G/H/I/J/K/L/M/N/O/P)

**Goal:** Validate that treating support as a boundary proximity indicator produces a positive semantic effect — matching or exceeding CR-A (semantic-only, 0.7336 mIoU).
**Requires:** CUE-01, CUE-02, CUE-03, CUE-04
**Depends on:** Phase 4

**Experiment evolution:**
- **CR-C** (BoundaryProximityCueLoss): confidence-weighted BCE on binary valid.
- **CR-F** (UnweightedBoundaryCueLoss): unweighted BCE on binary valid.
- **CR-G** (SoftBoundaryLoss): BCE on continuous support. Best mIoU=0.7240 < CR-A. Failed: BCE has irreducible entropy lower bound (~0.2) on continuous target, persistent noise gradient competes with semantic loss.
- **CR-H** (FocalMSEBoundaryLoss): MSE + soft Dice replaces BCE. Early signals looked healthy (MSE dynamic equilibrium, Dice improving), but **full training on the real environment matched CR-B (net negative vs CR-A 0.7336)**. Conclusion: switching from BCE to MSE+Dice removes the BCE floor but does not fix the core mismatch — continuous Gaussian regression does not generate a semantically aligned gradient. Closed out.
- **CR-I** (BoundaryUpweightLoss): CR-H aux + BFANet-inspired semantic CE upweight. Boundary points (support>0.5) get up to 10x CE weight using continuous support, truncated to prevent tail inflation. 1-epoch validation: dice_score 0.27 (2.5x faster than CR-H), boundary_ce_frac 16%. Full training not scheduled (superseded by CR-L).
- **CR-J** (BoundaryGatedSemanticModel + BoundaryUpweightLoss with g v3): CR-I loss + g v3 (BoundaryGatingModule, PTv3 patch self-attention, per-channel gate). **Dropped before full training** — g v3 is superseded by g v4 (cross-stream fusion attention) in CR-M, so running CR-J with the older gating module is no longer informative.
- **CR-K** (GT support-weighted CE + Lovasz, no aux, no boundary head): CR-I ablation baseline. Removes both the Focal MSE+Dice aux and the support head to isolate the effect of the boundary-region semantic CE upweight on its own. Retained as a pending ablation to be run only if CR-L / CR-M results leave the CE-upweight-only question still interesting.
- **CR-L** (BoundaryBinaryLoss): BFANet-style binary classification instead of continuous regression. Aux = support-weighted binary BCE + local Dice on the `support>0` transition zone; semantic CE keeps the CR-I boundary upweighting. First config (`threshold=0.9, pos_weight=5, sample_weight_scale=9`) had boundary head collapse — root cause: voxel radius (~2.35cm at grid 6cm) is larger than support σ (2cm), so `s>0.9` leaves only ~0.35% positives post-voxel, below what BCE+Dice can learn. Fixed config: `threshold=0.5` (physical lower bound → ~2% positive), `pos_weight=1` (the `1 + s*9` sample weight already handles class rebalancing; stacking BCE pos_weight drove collapse), local Dice kept on the `support>0` region. **Ep100 complete: best val_mIoU = 0.7251 @ ep68** (mAcc 0.8380, allAcc 0.8614), ep100 = 0.6994. vs CR-A: −0.0085 (within 0.05–0.07 single-seed noise floor). vs CR-B: +0.0067. **Verdict: single-stream aux route ceilings near CR-B level, cannot meaningfully beat CR-A under the current voxel/threshold geometry.**
- **CR-M** (BoundaryGatedSemanticModelV4 + DualSupervisionBoundaryBinaryLoss): CR-L loss applied to both a v1 (pre-fusion) and v2 (post-fusion) stream, coupled by g v4 (CrossStreamFusionAttention, K=patch_size fusion-query cross-stream attention). Model clones the v1 CR-L heads into v2; loss wrapper runs BoundaryBinaryLoss on both streams with v1_/v2_ prefixed keys and sums totals. Trainer `_build_loss_inputs` additively forwards `seg_logits_v2` / `support_pred_v2`. Local smoke 6.1–6.5 all pass (step-0 v1==v2 equivalence exact, two-step gradient flow through g v4, wrapper keys, trainer forwarding, CR-L no-regression). Committed as 3fdfbd1. **Awaiting real-environment queue.** Deliberately still uses `edge[:,3]>0.5` threshold (not the new r=0.06 mask) so CR-L vs CR-M stays an architecture-only ablation.
- **CR-N** (PureBFANetLoss on single-stream model): faithful BFANet reproduction — hard-mask 10× CE on `boundary_mask_r060` + unweighted BCE (`pos_weight=1`) + **global** Dice over the whole scene. Boundary target comes from the new r=0.06 m precomputed mask (see Preprocessing below). Committed 530f55f. After the 2026-04-11 afternoon alignment pass: `aux_weight=1.0`, CE `.mean()` normalization over total points, BCE surface form `BCEWithLogits` (AMP-safe). **Purpose:** isolate whether BFANet's published recipe works on 7% positive-ratio data, given that the published paper was on ScanNet with a different class distribution. Queued for real env.
- **CR-O** (SoftWeightedSemanticLoss): minimal smooth extension of CR-A — same semantic-only model, loss = CE · (1 + s · 9) + Lovasz, fully continuous soft weighting across [0,1], no truncation, no aux, no boundary head. Committed a8cae64. Tests whether continuous boundary-proximity weighting *alone* moves val_mIoU above CR-A within seed noise. Queued after CR-L/M/N.
- **CR-P** (DualSupervisionPureBFANetLoss on g v4 dual-stream model): the cross product — CR-M's architecture × CR-N's loss × the r=0.06 m radius mask. Committed 3d797b0 + alignment 3452189 + AMP revert abe0fc8. v1:v2 weighting = 1:1 (corrected from initial 0.5:1.0 guess per BFANet code audit). **ep1-78 complete (interrupted at ep79 step 35, do NOT complete ep100): best val_mIoU 0.7241 @ ep45**, vs CR-A −0.0095, vs CR-L Δ=0.001 (statistically identical). **Four structural findings from 70MB train.log post-hoc analysis:** (1) BFANet naive BCE (pos_weight=1) did NOT collapse at 7% positive — Dice climbed 0.14→0.72, prob_pos/neg cleanly separated; (2) g v4 fusion is near-identity — v2 − v1 only 0.003-0.007 on all metrics, fusion path has no meaningful amplitude even after the K=48 fix over BFANet's degenerate K=1; (3) CR-P train_sem is 0.18-0.64 HIGHER than CR-L at every matched epoch while val_mIoU is identical → falsifies the 2026-04-10 "CR-L aux crowds sem gradient from ep1" diagnosis, train_sem is not a val predictor under single-seed single-run; (4) Peak at ep45, then flat 30+ epoch plateau while train loss kept descending — earlier and more severe overfit than CR-L. Plot: `.planning/phases/05-boundary-proximity-cue-experiment/cr_p_vs_crl_cra_ep78.png`. Log: `outputs/clean_reset_s38873367/pure_bfanet_v4/train.log`. **Verdict: dual-stream + g v4 + faithful PureBFANetLoss lands in the same 0.72-0.73 band as CR-L. Aux-head route ceiling at ~0.725 is now confirmed across architectures.**

**Preprocessing infrastructure (2026-04-11):** BFANet ScanNet radius truth corrected from misread "3× voxel" framing to the real invariant **r = 0.06 m absolute physical**. Built `scripts/data/probe_boundary_radius.py` (pilot sweep over r ∈ {0.03, 0.06, 0.09, 0.12} on 10 chunks) + `scripts/data/generate_boundary_mask.py` (multiprocessing.Pool(8) full-dataset generator). Full run on 264 chunks at r=0.06: min 1.65%, mean 7.19%, median 6.76%, max 14.11%, 202/264 (76.5%) in [5,15]% window, all ≤15%. Output: `{training,validation}/<chunk_id>/boundary_mask_r060.npy` (uint8). Dataset loader, transforms, trainer loss-input builder, and loss forward all accept an optional precomputed `boundary_mask`, falling back silently to `edge[:,3]>0.5` when absent. CR-N and CR-P consume the new mask; CR-M deliberately does not.

**Infrastructure:** Trainer refactored to dynamic metric dispatch (commit 35b91bd). Adding new losses requires zero trainer changes.

**Success criterion:** CR-L and/or CR-M reach mIoU ≥ CR-A (0.7336). Any positive delta confirms boundary-aware auxiliary supervision helps once the target is properly aligned (binary threshold within the voxel radius).

**Status after CR-L ep100 + CR-P ep78:** Both the single-stream aux route (CR-L, 0.7251 @ ep68) and the dual-stream aux route (CR-P, 0.7241 @ ep45) ceiling at ~0.725, both −0.85 to −0.95pp below CR-A, Δ=0.001 between them (statistically identical). **Aux-head route ceiling at ~0.725 is confirmed across architectures.** Two of the three post-CR-L open questions are now answered: (a) dual-stream CR-M will likely land in the same band (predicted, low value to confirm); (b) faithful BFANet CR-P did NOT close the gap. The only remaining open question is (c) **does CR-O's pure soft-weighted CE (no aux head at all) beat CR-A?** CR-O is now the single gating experiment for the milestone verdict. CR-N matters independently as the single-stream BFANet cross-arch control for the "does naive BCE collapse at 7% positive" finding.

---

### — Part 2: Boundary→Semantic Feature Gating (Phase 6, merged into Phase 5) —

### Phase 6: Module g redesign — boundary→semantic gating ✅ (merged into Phase 5)

**Original goal:** Serial derivation of boundary offset from semantic logits (semantic→boundary direction).
**Outcome:** g v1 (consistency MSE) and v2 (cosine offset) both failed — semantic logits too weak at boundaries to derive useful geometric information. Direction reversed.

**Redesigned goal:** Use boundary features to gate/enhance semantic features (boundary→semantic direction, BFANet-inspired).

**g iteration history:**
- **g v3** (BoundaryGatingModule, commit fe63163): PTv3 patch self-attention on boundary_feat → per-channel gate → `semantic_feat * (1 + gate)`. Carried in CR-J.
- **g v4** (CrossStreamFusionAttention, commit 3fdfbd1): K=patch_size fusion-query cross-stream attention. Fuses v1 (pre-fusion) and v2 (post-fusion) semantic/boundary streams. Carried in CR-M with dual supervision. **Active version.**

**CR-J status:** **Dropped before full training.** g v3 per-channel gating is superseded by g v4 cross-stream fusion; running CR-J alone (g v3 + CR-I loss) is no longer informative once CR-M exists.

**Key insight (preserved):** Semantic loss now supervises both branches. Gate/fusion creates a feedback loop: boundary features that help semantic classification get reinforced; those that don't get suppressed. This is the coupling mechanism that CR-A through CR-H lacked.

---

### Phase 7: Canonical update and milestone close

**Goal:** Update canonical docs to reflect clean-reset findings, route redesign conclusions, and experiment results. Close milestone.
**Requires:** GUARD-01, GUARD-02, GUARD-03
**Depends on:** Phase 5 CR-O training result (now the single gating experiment). CR-L ep100 landed 2026-04-11 (best 0.7251 @ ep68, −0.85pp vs CR-A). CR-P ep1-78 landed 2026-04-11 evening (best 0.7241 @ ep45, −0.95pp vs CR-A, Δ=0.001 vs CR-L). Aux-head route ceiling at ~0.725 confirmed across single-stream and dual-stream architectures. CR-M now low value (predicted to land in same band); CR-N queued as BFANet cross-arch control; CR-K conditional on CR-O. CR-H closed as failed; CR-J dropped. Post-training test hook separately broken and deferred — blocks test-split numbers but not the training-metric verdict.
