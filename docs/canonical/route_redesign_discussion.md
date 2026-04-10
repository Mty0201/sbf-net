# Route Redesign Discussion: Support as Boundary Proximity Cue

**Date:** 2026-04-07
**Status:** Design discussion, not implementation spec.
**Context:** The mechanism analysis (`clean_reset_mechanism_analysis.md`) established that the current support regression target is misaligned with the semantic task. This document discusses the route redesign: redefining support as a boundary proximity cue rather than a geometric distance field.

## 1. The New Research Judgment

The core reframing:

> Support should not be defined as "precise geometric field prediction target." It should be defined as a **boundary proximity cue** — a soft indicator that a point is near a semantic class transition boundary.

This reframing changes support's role at every level:

| Aspect | Old Role (geometric field) | New Role (proximity cue) |
|--------|---------------------------|-------------------------|
| **What support_gt=0.8 means** | "This point is 0.017m from a fitted boundary primitive" | "This point is confidently near a semantic class transition" |
| **What the head learns** | Metric distance in 3D space | Boundary proximity confidence |
| **Feature demand on backbone** | Geometric localization features | Class-transition-detection features |
| **Alignment with semantic task** | Low (distance ≠ class identity) | High (class transition detection ≈ class discrimination) |
| **Gradient character** | Persistent regression pressure | Classification-like, converges when confident |

## 2. The valid + support Representation: Design Analysis

### 2.1 What valid and support currently encode

In the existing data pipeline (`edge.npy` column layout `[dir, dir, dir, dist, support, valid]`):

- **valid** (column 5): Binary. 1 if the point is within `support_radius` (0.08m) of a fitted semantic boundary primitive. 0 otherwise. Currently ~16-17% of points are valid.
- **support** (column 4): Continuous in [0, 1]. Truncated Gaussian: `exp(-d²/2σ²)` where `d` is the 3D distance to the nearest boundary and `σ = 0.04m`. Only meaningful where `valid = 1`.

### 2.2 The reinterpretation

Under the new framing, valid + support answer the same question at different levels of precision:

- **valid** = "is this point near a semantic class transition?" (hard boundary, binary)
- **support** = "how confidently near?" (soft weight within the hard boundary region)

This is **not** a regression target. It is a **confidence-weighted binary cue**:
- `valid = 0`: definitely not near a boundary. No supervision needed.
- `valid = 1, support ≈ 1.0`: very confidently near a boundary (essentially on the boundary).
- `valid = 1, support ≈ 0.2`: weakly near a boundary (at the fringe of the supervision radius).

The network is not being asked "predict the Gaussian distance field." It is being asked "detect boundary proximity, with the Gaussian providing a soft confidence weight."

### 2.3 Why this interpretation is mechanistically more sound

**Gradient alignment:** When the auxiliary task is "detect boundary proximity," the features it demands are class-transition-detection features — the same features the semantic head needs to correctly classify points near class boundaries. This is the same alignment that makes DLA-Net's binary edge classification effective.

**Reduced optimization burden:** The network does not need to learn to encode metric distance. It only needs to learn "near boundary" vs "far from boundary," with a soft gradient of certainty. This is closer to a classification problem than a regression problem, even though the target is continuous.

**Natural convergence:** Once the network confidently predicts "near boundary" for valid points and "not near boundary" for invalid points, the loss gradient drops. The auxiliary task stops exerting pressure on the backbone. This is the "converge and stop" behavior of binary classification, not the "persistent floor" behavior of continuous regression.

## 3. Comparison with DLA-Net's Binary Edge Classification

### 3.1 What DLA-Net does

DLA-Net identifies semantic edge points (points near class transitions in the KNN neighborhood) and trains a binary auxiliary head to predict which points are edge points. The task is:

- **Target:** Binary (0 or 1). Is this point a semantic edge point?
- **Loss:** Binary cross-entropy
- **Scope:** All points receive supervision (either 0 or 1)
- **Feature demand:** Class-transition detection

### 3.2 How the reinterpreted valid + support differs

| Property | DLA-Net binary edge | Reinterpreted valid + support |
|----------|-------------------|------------------------------|
| **Hard boundary** | Implicit (KNN threshold) | Explicit (`valid` mask from geometric fitting) |
| **Confidence weighting** | None (all edge points are equally edge) | Gaussian decay: points closer to the boundary core get stronger weight |
| **Target distribution** | Bimodal: ~20% edge, ~80% non-edge | Three-level: ~83% non-boundary (`valid=0`), ~7% weak boundary (`valid=1, support<0.5`), ~10% strong boundary (`valid=1, support≥0.5`) |
| **Training signal density** | 100% of points supervised (every point is either edge or non-edge) | 100% of points supervisable IF formulated correctly (see Section 5) |
| **Boundary definition** | KNN-based semantic heterogeneity | Fitted geometric primitives (lines/polylines) aligned to semantic boundaries |
| **Geometric precision** | Coarse (KNN neighborhood radius) | Higher (fitted boundary + Gaussian decay) |

### 3.3 Potential advantages of valid + support over DLA-Net binary

1. **Richer boundary-region training signal.** DLA-Net treats all edge points equally. The support weighting creates a gradient of certainty: points very close to the boundary core contribute more to the auxiliary loss, while fringe points contribute less. This naturally focuses learning on the most informative boundary-region points.

2. **More points participate in training.** The valid mask defines a broader boundary zone (0.08m radius) than a typical KNN edge detection (which identifies only the edge points themselves). Within this zone, the support weight controls how much each point contributes. This gives the auxiliary task more training data than pure binary edge detection.

3. **Geometrically grounded boundary definition.** DLA-Net's KNN-based edge detection can be noisy (isolated misclassified neighbors create spurious edges). The fitted geometric primitives in the support pipeline produce cleaner, more coherent boundary definitions because they require DBSCAN clustering and line/polyline fitting to confirm a boundary.

4. **Graduated transition.** The Gaussian decay provides a smooth transition from "definitely boundary" to "definitely not boundary," which may produce smoother gradient flow than a hard binary cutoff.

### 3.4 Risks relative to DLA-Net

1. **Reduced supervision scope.** DLA-Net supervises 100% of points (every point is edge or non-edge). If the support auxiliary only supervises the 16-17% of valid points, the remaining 83% receive no auxiliary gradient. This means the auxiliary task cannot regularize features for interior points. (This is addressable — see Section 5.)

2. **Continuous target may still invite regression behavior.** Even if we *intend* support to be a proximity cue, the network and loss function don't know that. If the loss is SmoothL1 on a continuous target, the optimizer will try to regress the exact values, re-introducing the geometric encoding problem. The loss design must explicitly reflect the cue interpretation (see Section 5).

3. **Geometric fitting pipeline introduces complexity and potential noise.** DLA-Net's KNN edge detection is simple and robust. The support pipeline's multi-stage fitting (boundary centers → DBSCAN → line/polyline fitting → Gaussian field) may introduce systematic errors in specific geometric configurations. If the fitted primitives are wrong, the Gaussian field is wrong, and the supervision signal is misleading.

## 4. Why the Simple Binary Edge Task Can Succeed Without Sophisticated Coupling

This is an important question: if DLA-Net and the author's PTv3 reproduction show that even a simple two-head setup (no feature fusion, no gating) works with binary edges, why should the redesigned support approach need more?

### 4.1 The implicit coupling mechanism of aligned targets

When the auxiliary target is naturally aligned with the semantic task, explicit coupling is less important. The mechanism is:

1. The auxiliary head's gradient pushes backbone features toward class-transition detection.
2. These features are inherently useful for the semantic head (because class transitions are where semantic decisions are hardest).
3. The backbone autonomously develops features that serve both tasks, even without explicit feature exchange.

This is why the PTv3 reproduction works with simple head separation: the binary edge target creates aligned gradient pressure that improves the shared features for both tasks simultaneously.

### 4.2 When explicit coupling becomes necessary

Explicit coupling (gating, fusion, attention) becomes necessary when:
- The auxiliary target is NOT naturally aligned (like geometric distance regression)
- The auxiliary task needs more backbone capacity than the "mild regularizer" level
- The auxiliary task's knowledge needs to be explicitly transferred to the primary head

For the reinterpreted support-as-cue approach, explicit coupling should be **less necessary** than for the old support-as-field approach, because the cue interpretation aligns the target with the semantic task. However, coupling may still provide additional benefit:
- The support prediction could modulate the semantic loss (boundary-region emphasis via focus weighting)
- The support prediction could gate or condition semantic features at the decoder level

The key design principle: **design for implicit coupling first** (aligned target, mild gradient), and add explicit coupling only if empirical evidence shows it helps further.

## 5. Part 1 Implementation Design: Support as Boundary Proximity Cue

### 5.1 Target definition

The existing `valid` and `support` data can be reused without any data pipeline changes. The reinterpretation is purely in how the loss function uses them:

- **valid** remains the hard boundary indicator.
- **support** remains the continuous Gaussian value within the valid region.
- The semantic change: the loss should treat support as a soft confidence weight for a boundary-detection objective, not as a regression target to be precisely matched.

### 5.2 Label construction: no data pipeline change needed

The existing `edge.npy` contains `[dir, dir, dir, dist, support, valid]`. For Part 1, only columns 4 (support) and 5 (valid) are needed. Columns 0-3 (direction and distance) are ignored — they belong to Part 2's geometric field objectives.

This means **zero data preprocessing cost** for Part 1. The existing labels are reinterpreted, not rebuilt.

### 5.3 Loss design: the critical choice

The loss design must reflect the cue interpretation, not the regression interpretation. Here are the options, in order of increasing sophistication:

**Option L1: Binary classification with confidence weighting**

```
auxiliary_target = valid                          # binary: boundary or not
confidence_weight = where(valid, support, 1.0)    # Gaussian weight for boundary points, 1.0 for non-boundary
loss_aux = weighted_BCE(support_pred, valid, weight=confidence_weight)
```

Properties:
- 100% of points supervised (binary: every point is boundary or not)
- Boundary points weighted by support confidence (points closer to boundary core contribute more)
- Non-boundary points contribute with weight 1.0 (interior classification is easy, converges fast)
- Pure classification: no regression, no metric distance encoding
- Gradient naturally decays as the classifier becomes confident
- Closest analogy to DLA-Net, enhanced with confidence weighting

**Option L2: Soft binary classification (label smoothing via support)**

```
soft_target = valid * support                     # 0.0 for non-boundary, support ∈ (0,1] for boundary
loss_aux = BCE(support_pred, soft_target)
```

Properties:
- 100% of points supervised
- Non-boundary points: target = 0 (clear negative)
- Boundary points: target = soft value (closer to boundary → target closer to 1.0)
- Still classification-like, but with soft labels that encode proximity
- Risk: this is mathematically equivalent to the current SmoothL1 regression on support values, just with BCE. The optimizer may still try to regress exact values. The soft labels may cause the sigmoid to never reach confident predictions for fringe boundary points.

**Option L3: Hard binary classification + separate focus reweighting**

```
loss_aux = BCE(support_pred, valid)               # pure binary: detect the boundary zone
loss_focus = Lovasz(seg_logits[boundary], segment[boundary])  # where boundary = valid > 0.5
total = loss_semantic + w_aux * loss_aux + w_focus * support_weighted(loss_focus)
```

Properties:
- The auxiliary head does the simplest possible boundary detection (binary)
- The focus term uses ground-truth support weights to emphasize semantic accuracy near boundaries
- Clean separation: the auxiliary branch provides the boundary detection signal, the focus term provides the semantic-boundary coupling
- Two-stage benefit: auxiliary sharpens boundary features (implicit coupling), focus concentrates semantic learning on boundary region (explicit coupling)

### 5.4 Recommended loss design: Option L1 for initial validation, then L3

**Start with L1** (binary classification with confidence weighting) because:
- It is the closest to the DLA-Net mechanism that is known to work
- It is the simplest to implement and interpret
- It directly tests the core hypothesis: does treating support as a boundary proximity cue (instead of a regression target) recover the positive effect seen in DLA-Net and the PTv3 reproduction?
- If L1 works, it provides a solid baseline for subsequent experiments

**Graduate to L3** if L1 validates the direction, because:
- L3 adds the focus coupling mechanism (Lovasz on boundary region weighted by support confidence)
- This exploits the continuous support information beyond what pure binary classification can
- It tests whether the Gaussian confidence weighting provides additional value beyond binary detection

### 5.5 Head design

The existing `SupportHead` (2-layer MLP: Linear(64,64) + ReLU + Linear(64,1)) is appropriate for boundary proximity detection. No architectural change needed for Part 1.

Output: single sigmoid logit, interpreted as P(near semantic boundary).

### 5.6 Coupling with the semantic branch

**Part 1 coupling strategy:** Start with implicit coupling only (aligned gradient from the boundary detection task), matching the DLA-Net / PTv3 reproduction setup. The auxiliary head's gradient through the shared backbone is the coupling mechanism.

**Gradient management:** Unlike the old regression formulation, the binary classification task should naturally produce smaller and faster-converging gradient. But as a safety measure:

- Set `w_aux` (auxiliary loss weight) to 0.2-0.5, not 1.0. The auxiliary task should be a mild regularizer, not a competing primary objective.
- Monitor the auxiliary gradient share over training. If it exceeds 20% of total gradient by epoch 50, reduce the weight.

### 5.7 What Part 1 does NOT include

- No direction field prediction (columns 0-2 of edge.npy)
- No distance field prediction (column 3 of edge.npy)
- No geometric field regression of any kind
- No complex feature fusion, gating, or attention
- No data pipeline changes

Part 1 is deliberately minimal: same data, same architecture, different loss interpretation. The only changes are in the loss function and the auxiliary loss weight.

## 6. Part 2: Geometric Field Objectives (Deferred)

Part 2 should only be considered after Part 1 validates that the boundary proximity cue produces a positive semantic effect. If Part 1 succeeds, Part 2 asks:

> Now that the network reliably detects boundary proximity, can we additionally teach it geometric structure — direction, distance, attraction field — without losing the semantic benefit?

Part 2 design questions (for later):
- Can direction/distance prediction be added as a second-stage objective that activates only after the boundary detection task has converged?
- Should the geometric field be predicted from the boundary-adapter features (task-specific) rather than from the shared backbone (to avoid re-introducing gradient competition)?
- Is the geometric field better framed as a separate downstream consumer of boundary-detection features (post-hoc refinement) rather than as an additional training objective?

These questions are premature now. Part 1 comes first.

## 7. Route Summary

### Part 1: Boundary Proximity Cue (immediate next step)

**Goal:** Validate that treating support as a boundary proximity indicator (not a geometric field) produces a positive semantic effect, analogous to DLA-Net / PTv3 reproduction.

**Changes from CR-B:**
1. Loss: Replace SmoothL1+Tversky regression with confidence-weighted BCE classification (Option L1)
2. Target interpretation: `valid` = boundary/not-boundary label. `support` = confidence weight for boundary points.
3. Auxiliary weight: 0.2-0.5 (not 1.0). Mild regularizer, not competing primary objective.
4. No other changes to data, model, or optimization.

**Success criterion:** The new formulation either matches or exceeds CR-A (semantic-only, 0.7336 mIoU). Any positive delta over CR-A confirms that boundary-aware auxiliary supervision can help semantics when the target is properly aligned.

**Failure diagnosis:** If Part 1 also fails, the problem is deeper than target design — likely in the architecture or the backbone feature-sharing mechanism. At that point, the valid + support representation itself may need rethinking, or the coupling mechanism may need to be explicit rather than implicit.

### Part 2: Geometric Field Extension (conditional on Part 1 success)

**Goal:** Add structured geometric predictions (direction, distance, attraction) as a secondary objective on top of the validated boundary proximity cue.

**Precondition:** Part 1 produces mIoU ≥ CR-A (0.7336).

**Design principle:** Geometric field objectives should consume boundary-detection features without re-introducing gradient competition at the backbone level.

## 8. What This Discussion Concludes

1. The valid + support representation, reinterpreted as a hard boundary mask + soft confidence weight, is mechanistically more aligned with the semantic task than the old geometric field interpretation. The reinterpretation makes support functionally closer to DLA-Net's boundary classification — enhanced with geometric confidence — rather than a competing geometric regression target.

2. The reinterpreted formulation has structural advantages over pure DLA-Net binary classification: confidence weighting, broader boundary zone participation, and geometrically grounded boundary definitions. But these advantages only materialize if the loss function treats support as a cue, not as a regression target.

3. The simplest Part 1 experiment — confidence-weighted BCE with `w_aux` = 0.2-0.5 — directly tests whether the target alignment hypothesis holds. No data pipeline change, no architecture change, one loss function replacement.

4. Part 2 (geometric field objectives) is deferred until Part 1 validates the foundation. The staged approach prevents premature geometric burden from contaminating the boundary-awareness benefit.

---

## 9. Implementation Evolution Log (2026-04-08 → 2026-04-10)

Sections 1–8 above are the 2026-04-07 design record. This section logs what actually happened when Option L1 was executed and why the route branched beyond the original single-variant plan. These entries are updates, not retractions — the core reframing (support as proximity cue, geometric field deferred to Part 2) still holds.

### 9.1 CR-C / CR-F / CR-G: Option L1 as specified, and why continuous BCE did not land

CR-C (confidence-weighted BCE on binary `valid`) and CR-F (unweighted BCE on binary `valid`) were implemented as direct realizations of Option L1. Neither reached CR-A. The deeper lesson arrived with CR-G (`SoftBoundaryLoss`), which applied BCE directly to the continuous support target instead of the binary mask: best mIoU = 0.7240 < CR-A = 0.7336.

The failure mode is structural, not tunable: `BCE(p, s_gt) ≥ H(s_gt)`, where `H(s_gt)` is the entropy of the continuous target distribution. On this dataset `H(s_gt) ≈ 0.2`. The BCE loss cannot be driven below that floor no matter how well the network predicts, so the auxiliary loss retains a persistent gradient that competes with the semantic branch for the entire training run. Section 5.3's Option L2 ("soft binary classification") inherits the same floor in a different form and would have hit the same wall.

**Correction to Section 5.3:** BCE is an acceptable loss on a binary target, but it is not acceptable on a continuous `support_gt ∈ [0, 1]` target. Any future continuous-target variant must use a loss whose lower bound is 0.

### 9.2 CR-H: MSE + Dice replaces BCE on the continuous target

CR-H (`FocalMSEBoundaryLoss`) replaces BCE with Focal MSE plus a soft Dice term. MSE's lower bound on a continuous target is 0, and Dice addresses the imbalance between the dense background and the sparse boundary zone. Real-training telemetry showed fast separation of `prob_pos` vs `prob_neg`, Dice steadily improving, and MSE settling into a ~0.05–0.1 dynamic equilibrium without flatlining — the auxiliary task converges without going dead. Full-run result pending.

### 9.3 CR-I: BFANet-style boundary-region semantic CE upweight

CR-I (`BoundaryUpweightLoss`) keeps CR-H's aux and adds a BFANet-inspired upweight on the semantic CE itself: points with `support_gt > 0.5` get up to 10× CE weight, using the continuous support as the weight and truncating at 0.5 to prevent the Gaussian tail from dragging in too many interior points. This is the first design in the sweep where the semantic branch is *explicitly* told which points are the difficult-near-boundary ones, rather than relying only on gradient flow from the aux head. 1-epoch validation: `dice_score` 0.27 (~2.5× faster than CR-H), `boundary_ce_frac` 16%, healthy training.

### 9.4 CR-J: g direction reversed from semantic→boundary to boundary→semantic

Phase 6's original plan was a module g that derives a boundary offset vector field from the semantic logits (semantic→boundary, Part 2 spirit). Two iterations of that design (g v1 consistency MSE, g v2 cosine offset) both came back neutral in fair retests: semantic logits are too weak at boundaries to derive useful geometric information from.

g was redesigned as a **boundary→semantic gate** (g v3, `BoundaryGatingModule`). It takes the boundary adapter's feature map, runs PTv3 serialized patch self-attention over it, and emits a per-channel gate that modulates the semantic feature map multiplicatively (`semantic_feat * (1 + gate)`). The output layer is zero-initialized so the initial multiplier is ≈1.5 and the model starts near identity. There is no dedicated loss for g — the semantic loss drives g through the gate, creating a feedback loop where boundary features that help semantic classification get reinforced and those that do not get suppressed. This is the missing implicit-coupling mechanism Section 4 suggested might eventually be needed. Phase 6 is subsumed into CR-J.

### 9.5 CR-K / CR-L: Ablation and BFANet binary return

Two additional variants were added to isolate what is actually paying its way:

- **CR-K** keeps only the CR-I semantic CE upweight — no aux, no boundary head, no g. If CR-K ≥ CR-I, the auxiliary support head is redundant and the CE upweight is doing all the work. This is the purest ablation of the "boundary-region semantic emphasis" hypothesis.

- **CR-L** (`BoundaryBinaryLoss`) returns to a BFANet-style *binary* BCE + local Dice on the support>0 transition zone, while keeping the CR-I semantic CE upweight. This is Option L1's spirit — binary target, classification-like behavior — but with a hard-fought parameterization. The initial attempt (`threshold=0.9, pos_weight=5, sample_weight_scale=9`) made the boundary head collapse (`prob_pos=0.075, dice_score=0.05`).

  Root cause: the grid voxel size is 6cm (radius ~2.35cm) but support σ = 2cm. At `support > 0.9` the true positives are only ~0.62% of the raw data, and after voxelization collapse the ratio drops to ~0.35%. That is below what a BCE+Dice combination can learn at all. Secondary cause: `sample_weight_scale = 9` already makes core boundary points carry 10× the background weight, so stacking `pos_weight = 5` on top of it produces ~45× effective imbalance correction, which also drives collapse.

  Fix: `boundary_threshold = 0.5` (≈2% post-voxel, clean Dice math), `pos_weight = 1` (rebalancing handled entirely by the sample weight), local Dice restricted to `support > 0` so the background class cannot dominate the Dice denominator. The 0.5 threshold is a *physical lower bound* set by the voxel radius, not a heuristic — pushing it higher would require preprocessing changes.

### 9.6 Updated success criterion

The original criterion — "CR-C reaches mIoU ≥ CR-A (0.7336)" — is generalized to the sweep:

> Any of CR-H / CR-I / CR-J / CR-K / CR-L reaches mIoU ≥ CR-A (0.7336).

A positive delta from *any* variant confirms boundary-aware auxiliary supervision helps when the target is aligned. The sweep also supports structural conclusions independent of any single outcome: CR-K vs CR-I isolates whether the support head pays for itself; CR-J vs CR-I isolates whether explicit coupling (the gate) provides additional benefit beyond implicit coupling; CR-L vs CR-H isolates whether binary vs continuous auxiliary targets produce different semantic outcomes. Phase 7 (canonical update + milestone close) consumes these comparisons directly.
