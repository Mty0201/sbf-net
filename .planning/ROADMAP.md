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

### Phase 5: Boundary proximity cue experiments (CR-C/F/G/H/I/J/K/L/M)

**Goal:** Validate that treating support as a boundary proximity indicator produces a positive semantic effect — matching or exceeding CR-A (semantic-only, 0.7336 mIoU).
**Requires:** CUE-01, CUE-02, CUE-03, CUE-04
**Depends on:** Phase 4

**Experiment evolution:**
- **CR-C** (BoundaryProximityCueLoss): confidence-weighted BCE on binary valid.
- **CR-F** (UnweightedBoundaryCueLoss): unweighted BCE on binary valid.
- **CR-G** (SoftBoundaryLoss): BCE on continuous support. Best mIoU=0.7240 < CR-A. Failed: BCE has irreducible entropy lower bound (~0.2) on continuous target, persistent noise gradient competes with semantic loss.
- **CR-H** (FocalMSEBoundaryLoss): MSE + soft Dice replaces BCE. Early signals looked healthy (MSE dynamic equilibrium, Dice improving), but **full training on the real environment matched CR-B (net negative vs CR-A 0.7336)**. Conclusion: switching from BCE to MSE+Dice removes the BCE floor but does not fix the core mismatch — continuous Gaussian regression does not generate a semantically aligned gradient. Closed out.
- **CR-I** (BoundaryUpweightLoss): CR-H aux + BFANet-inspired semantic CE upweight. Boundary points (support>0.5) get up to 10x CE weight using continuous support, truncated to prevent tail inflation. 1-epoch validation: dice_score 0.27 (2.5x faster than CR-H), boundary_ce_frac 16%. Full training not yet scheduled.
- **CR-J** (BoundaryGatedSemanticModel + BoundaryUpweightLoss with g v3): CR-I loss + g v3 (BoundaryGatingModule, PTv3 patch self-attention, per-channel gate). **Dropped before full training** — g v3 is superseded by g v4 (cross-stream fusion attention) in CR-M, so running CR-J with the older gating module is no longer informative.
- **CR-K** (GT support-weighted CE + Lovasz, no aux, no boundary head): CR-I ablation baseline. Removes both the Focal MSE+Dice aux and the support head to isolate the effect of the boundary-region semantic CE upweight on its own. Retained as a pending ablation to be run only if CR-L / CR-M results leave the CE-upweight-only question still interesting.
- **CR-L** (BoundaryBinaryLoss): BFANet-style binary classification instead of continuous regression. Aux = support-weighted binary BCE + local Dice on the `support>0` transition zone; semantic CE keeps the CR-I boundary upweighting. First config (`threshold=0.9, pos_weight=5, sample_weight_scale=9`) had boundary head collapse — root cause: voxel radius (~2.35cm at grid 6cm) is larger than support σ (2cm), so `s>0.9` leaves only ~0.35% positives post-voxel, below what BCE+Dice can learn. Fixed config: `threshold=0.5` (physical lower bound → ~2% positive), `pos_weight=1` (the `1 + s*9` sample weight already handles class rebalancing; stacking BCE pos_weight drove collapse), local Dice kept on the `support>0` region. **Smoke complete; full training in flight on real environment with positive signals.**
- **CR-M** (BoundaryGatedSemanticModelV4 + DualSupervisionBoundaryBinaryLoss): CR-L loss applied to both a v1 (pre-fusion) and v2 (post-fusion) stream, coupled by g v4 (CrossStreamFusionAttention, K=patch_size fusion-query cross-stream attention). Model clones the v1 CR-L heads into v2; loss wrapper runs BoundaryBinaryLoss on both streams with v1_/v2_ prefixed keys and sums totals. Trainer `_build_loss_inputs` additively forwards `seg_logits_v2` / `support_pred_v2`. Local smoke 6.1–6.5 all pass (step-0 v1==v2 equivalence exact, two-step gradient flow through g v4, wrapper keys, trainer forwarding, CR-L no-regression). Committed as 3fdfbd1. **Awaiting real-environment queue.**

**Infrastructure:** Trainer refactored to dynamic metric dispatch (commit 35b91bd). Adding new losses requires zero trainer changes.

**Success criterion:** CR-L and/or CR-M reach mIoU ≥ CR-A (0.7336). Any positive delta confirms boundary-aware auxiliary supervision helps once the target is properly aligned (binary threshold within the voxel radius).

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
**Depends on:** Phase 5 CR-L and CR-M training results (CR-H closed out as failed; CR-J dropped; CR-K retained as optional ablation)
