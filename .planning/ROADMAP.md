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
- **CR-H** (FocalMSEBoundaryLoss): MSE + soft Dice replaces BCE. MSE lower bound=0, Dice handles imbalance. Real training positive: fast separation, Dice improving, mIoU not degraded. MSE maintains ~0.05-0.1 dynamic equilibrium (not dead weight).
- **CR-I** (BoundaryUpweightLoss): CR-H aux + BFANet-inspired semantic CE upweight. Boundary points (support>0.5) get up to 10x CE weight using continuous support, truncated to prevent tail inflation. 1-epoch validation: dice_score 0.27 (2.5x faster than CR-H), boundary_ce_frac 16%.
- **CR-J** (BoundaryGatedSemanticModel + BoundaryUpweightLoss): CR-I loss + g v3 architecture. BoundaryGatingModule uses boundary_feat + PTv3 patch self-attention to produce per-channel gates that modulate semantic_feat. Boundary→semantic direction (replacing failed semantic→boundary g v1/v2). No dedicated loss — semantic loss drives g through the gate, creating a feedback loop where semantic loss also supervises boundary features.
- **CR-K** (GT support-weighted CE + Lovasz, no aux, no boundary head): CR-I ablation baseline. Removes both the Focal MSE+Dice aux and the support head to isolate the effect of the boundary-region semantic CE upweight on its own. If CR-K ≥ CR-I, the boundary head is not paying for itself and the aux term is redundant.
- **CR-L** (BoundaryBinaryLoss): BFANet-style binary classification instead of continuous regression. Aux = support-weighted binary BCE + local Dice on the `support>0` transition zone; semantic CE keeps the CR-I boundary upweighting. First config (`threshold=0.9, pos_weight=5, sample_weight_scale=9`) had boundary head collapse — root cause: voxel radius (~2.35cm at grid 6cm) is larger than support σ (2cm), so `s>0.9` leaves only ~0.35% positives post-voxel, below what BCE+Dice can learn. Fixed config: `threshold=0.5` (physical lower bound → ~2% positive), `pos_weight=1` (the `1 + s*9` sample weight already handles class rebalancing; stacking BCE pos_weight drove collapse), local Dice kept on the `support>0` region.

**Infrastructure:** Trainer refactored to dynamic metric dispatch (commit 35b91bd). Adding new losses requires zero trainer changes.

**Success criterion:** Any of CR-I / CR-J / CR-K / CR-L reaches mIoU ≥ CR-A (0.7336). Any positive delta confirms boundary-aware auxiliary supervision helps when the target is properly aligned.

---

### — Part 2: Boundary→Semantic Feature Gating (Phase 6, merged into CR-J) —

### Phase 6: Module g redesign — boundary→semantic gating ✅ (merged into Phase 5 CR-J)

**Original goal:** Serial derivation of boundary offset from semantic logits (semantic→boundary direction).
**Outcome:** g v1 (consistency MSE) and v2 (cosine offset) both failed — semantic logits too weak at boundaries to derive useful geometric information. Direction reversed.

**Redesigned goal:** Use boundary features to gate/enhance semantic features (boundary→semantic direction, BFANet-inspired).

**Architecture (g v3, commit fe63163):**

```
backbone → feat
             ├──→ boundary_adapter → boundary_feat → support_head → support_pred
             │                             ↓
             │                    g(boundary_feat, neighbors) → gate (sigmoid)
             │                             ↓
             └──→ semantic_adapter → semantic_feat * (1 + gate) → semantic_head → seg_logits
```

**Module g v3 design (BoundaryGatingModule):**
- Input: boundary_feat (N, 64) from boundary adapter
- Neighborhood: PTv3 serialized patch self-attention (patch_size=48, 4 heads)
- Output: per-channel gate (N, 64) → sigmoid → residual gating on semantic_feat
- Zero-init output layer → initial gate=0.5, multiplier=1.5 ≈ identity
- No dedicated loss — semantic loss gradients flow through gate → g → boundary_feat → backbone

**Key insight:** Semantic loss now supervises both branches. Gate creates a feedback loop: boundary features that help semantic classification get reinforced; those that don't get suppressed. This is the missing coupling mechanism that CR-A through CR-H lacked.

**Status:** Implemented as CR-J (BoundaryGatedSemanticModel + BoundaryUpweightLoss). Awaiting full training.

---

### Phase 7: Canonical update and milestone close

**Goal:** Update canonical docs to reflect clean-reset findings, route redesign conclusions, and experiment results. Close milestone.
**Requires:** GUARD-01, GUARD-02, GUARD-03
**Depends on:** Phase 5 (or Phase 6 if executed)
