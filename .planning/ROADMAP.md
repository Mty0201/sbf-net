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

### Phase 5: Boundary proximity cue experiment (CR-C)

**Goal:** Validate that treating support as a boundary proximity indicator (confidence-weighted BCE, not geometric regression) produces a positive semantic effect — matching or exceeding CR-A (semantic-only, 0.7336 mIoU).
**Requires:** CUE-01, CUE-02, CUE-03, CUE-04
**Depends on:** Phase 4

**Changes from CR-B:**
1. Loss: Replace SmoothL1+Tversky regression with confidence-weighted BCE classification (Option L1 from route redesign)
2. Target interpretation: `valid` = boundary/not-boundary label. `support` = confidence weight for boundary points.
3. Auxiliary weight: 0.2–0.5 (mild regularizer, not competing primary objective)
4. No other changes to data, model, or optimization

**What Phase 5 does NOT include:**
- No direction field prediction (columns 0-2 of edge.npy)
- No distance field prediction (column 3 of edge.npy)
- No geometric field regression of any kind
- No complex feature fusion, gating, or attention
- No data pipeline changes

**Success criterion:** CR-C mIoU ≥ CR-A (0.7336). Any positive delta confirms boundary-aware auxiliary supervision helps when the target is properly aligned.
**Failure diagnosis:** If CR-C also fails, the problem is deeper than target design — likely architectural or in the backbone feature-sharing mechanism.

---

### — Part 2: Serial Derivation of Boundary Offset Field (Phase 6, conditional) —

### Phase 6: Serial derivation module g — boundary offset field from semantic predictions

**Goal:** Implement a learnable module g that derives a 3D boundary offset vector field from semantic logits, enabling boundary-nearby points to project onto the boundary surface. This replaces the original parallel geometric prediction approach (which failed due to gradient competition) with a serial derivation architecture where edge supervision reinforces semantic prediction quality.

**Requires:** SER-01, SER-02, SER-03, SER-04, SER-05
**Depends on:** Phase 5 (assumes CUE-04 success criterion met)
**Precondition:** Phase 5 CR-C mIoU ≥ CR-A (0.7336)

**Architecture (from discussion 2026-04-07):**

```
backbone → semantic_head → seg_logits
                               ↓
backbone → support_head → support (BCE, retained from CR-C)
                               ↓
               g(softmax(seg_logits), coord) → offset (3D, smooth-L1)
```

**Module g design:**
- Input: softmax(seg_logits) (N,8) + coord (N,3) → (N,11)
- Neighborhood: KNN on coord (K neighbors per point, not reusing PTv3 Z-curve)
- Edge features: [feat_i, feat_j - feat_i, coord_j - coord_i] per neighbor pair (dim=25)
- Aggregation: shared MLP (25→64→64) + max-pool over K neighbors → (N,64)
- Output head: MLP (64→32→3) → offset prediction
- Support remains a separate head from backbone (CR-C route, not inside g)

**Edge representation:**
- offset (3D): displacement vector from point to nearest boundary. `coord + offset` = projected boundary position
- Supervised by smooth-L1 vs `dir_gt × dist_gt` (combined from existing edge ground truth)
- Only valid points (edge[:,5]=1) receive offset supervision
- This is a novel representation for 3D point cloud segmentation (literature gap — see §7 of part2_serial_derivation_discussion.md)

**Training:** No extra warmup — g shares learning rate group with semantic head. Existing cosine annealing + LR grouping provides natural warmup.

**Design principle:** Backbone only learns semantic features. Geometric field is derived FROM semantic output, not predicted in parallel. Edge supervision gradients through g reinforce "improve semantic predictions at boundaries" — fully aligned with semantic task. No gradient competition.

---

### Phase 7: Canonical update and milestone close

**Goal:** Update canonical docs to reflect clean-reset findings, route redesign conclusions, and experiment results. Close milestone.
**Requires:** GUARD-01, GUARD-02, GUARD-03
**Depends on:** Phase 5 (or Phase 6 if executed)
