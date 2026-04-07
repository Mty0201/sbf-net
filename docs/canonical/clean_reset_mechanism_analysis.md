# Mechanism Analysis: Why Binary Edge Supervision Helps While Support Regression Currently Does Not

**Date:** 2026-04-07
**Status:** Open diagnostic. Concept not invalidated.
**Prerequisite:** Read `clean_reset_analysis.md` (raw data) and `clean_reset_diagnostic.md` (ranked failure hypotheses).

## The Core Tension

Three pieces of evidence are in tension:

1. **DLA-Net** (arXiv:2106.00376), operating in the same building-facade point-cloud domain, demonstrates that boundary-aware / semantic-edge auxiliary supervision improves semantic segmentation.
2. **The author's own PTv3 reproduction** confirms that even a simple two-head setup (semantic + edge-related auxiliary) produces a positive effect on the semantic side, without sophisticated feature interaction.
3. **The current CR-B result** shows the opposite: adding support supervision to a PTv3 backbone makes semantic segmentation worse (CR-A 0.7336 vs CR-B 0.7184, -1.52pp).

The question is not whether dual supervision can work -- it provably can. The question is: why does DLA-Net-style binary edge classification help, while the current support formulation does not?

## 1. Functional Difference: Binary Edge Classification vs Continuous Support Regression

### What the binary edge task asks the network to learn

A binary semantic-edge classification target asks: "is this point on or near a class transition boundary?" The answer is yes/no. The gradient signal has these properties:

| Property | Binary Edge Classification |
|----------|--------------------------|
| **Target complexity** | 1 bit per point (boundary / not boundary) |
| **Optimization surface** | BCE -- smooth, well-behaved, single sigmoid decision |
| **Gradient magnitude** | Proportional to prediction error; largest at the decision boundary, decays when confident |
| **Spatial scope** | Covers all points near class transitions (typically defined by KNN semantic heterogeneity) |
| **What features it rewards** | Features that discriminate "near a class transition" from "interior of a class" |
| **Alignment with semantic task** | High -- the features that detect class transitions are exactly the features that distinguish classes near boundaries |

### What the current support task asks the network to learn

The support target asks: "what is the truncated Gaussian distance (sigma=0.04m) from this point to the nearest fitted geometric boundary primitive (line/polyline)?" The answer is a continuous scalar in [0, 1].

| Property | Continuous Support Regression |
|----------|------------------------------|
| **Target complexity** | Continuous scalar per point, encoding geometric distance |
| **Optimization surface** | SmoothL1 on sigmoid output + Tversky on binary region mask |
| **Gradient magnitude** | SmoothL1 gradient is capped at 1.0 and linear near zero; small gradients for points with small support values |
| **Spatial scope** | Only 16-17% of points have valid supervision (within 0.08m of a fitted boundary). 83% of points produce zero gradient. |
| **What features it rewards** | Features that encode metric distance to geometric boundary primitives |
| **Alignment with semantic task** | Low-Medium -- "how far am I from a boundary" is a geometric question, not a semantic discrimination question |

### The critical asymmetry

The binary edge task is **semantically aligned**: the features that predict "this point is near a class transition" are the same features the semantic head needs to correctly classify ambiguous boundary-region points. Learning to detect class transitions directly improves the representation for class discrimination at boundaries.

The support task is **geometrically aligned but semantically indirect**: the features that predict "this point is 0.03m from a fitted line segment" encode metric geometry, not class identity. A point's distance to a boundary primitive does not tell the network which class the point belongs to -- it tells the network where the boundary is in 3D space. This is useful information in principle, but learning to encode it requires backbone features to represent geometric distance fields, which may compete with (rather than complement) class-discriminative features.

## 2. Why DLA-Net / PTv3 Reproduction Works With Simple Head Separation

The author's successful PTv3 reproduction used a simple two-head setup: semantic head + edge-related auxiliary head, without sophisticated feature fusion. Why does this work?

### The auxiliary edge task acts as a feature-sharpening regularizer

A binary edge classification task produces a gradient signal that says: "your backbone features should be able to distinguish boundary-region points from interior points." This is a low-complexity regularizer with two key properties:

1. **It sharpens class boundaries in feature space.** Points near class transitions get pushed toward features that encode "which side of the boundary am I on?" -- which is exactly what the semantic head also needs.

2. **Its gradient is small relative to semantic CE.** A binary classification on ~20% of points (boundary region) with BCE loss produces much less total gradient than 8-class CE on 100% of points. The auxiliary task influences the backbone mildly, acting as a regularizer rather than a competing objective.

The combination is: cheap gradient signal, naturally aligned with semantic discrimination, mild enough to not dominate the backbone representation.

### Why the support task does NOT act as a feature-sharpening regularizer

The support regression task produces a fundamentally different gradient signal:

1. **It asks for metric precision rather than categorical discrimination.** Learning "this point is 0.04m from the boundary" requires encoding absolute distance in feature space. This is a harder task that demands more backbone capacity.

2. **Its gradient grows in relative share over training.** From the CR-B loss dynamics: support loss starts at 12% of total gradient but reaches 44% by epoch 100. As semantic loss converges, support loss becomes an increasingly dominant force on the backbone. This is the opposite of the "mild regularizer" pattern -- it becomes a competing primary objective.

3. **The SmoothL1 + Tversky combination has a persistent loss floor.** The support regression loss converges slowly (0.19 to 0.11 over 100 epochs, 40% reduction), meaning it contributes non-trivial gradient throughout training even when the support head has largely learned what it can. A binary classifier converges faster and stops exerting gradient pressure once it's confident.

## 3. Why Current CR-B Specifically Fails

### 3a. The support objective learns geometry that does not feed back into class separation

The support head learns the boundary field reasonably well (support_cover = 0.63), confirming the task is learnable. But what did the backbone have to learn to enable this?

The Gaussian support field is defined as `exp(-(d^2) / (2*0.04^2))` where `d` is the 3D Euclidean distance to the nearest fitted boundary primitive. Learning to predict this requires features that encode:
- How far each point is from the nearest class boundary in 3D
- The shape of nearby boundary primitives (lines/polylines)

These are **geometric localization features**, not **class-discriminative features**. A feature that says "I am 0.03m from a wall-column boundary" does not help distinguish walls from columns -- it helps locate the boundary itself.

In contrast, a binary edge classifier's features encode: "am I near a class transition?" This is directly useful for semantic discrimination because the features that detect transitions also encode "which classes are transitioning here."

### 3b. The supervision target is too sparse for the gradient to be informative

Only 16-17% of points receive non-zero support supervision. For the other 83%, the support branch receives zero gradient. This means:
- The support head learns from a small, boundary-localized fraction of the data
- The backbone features for interior points are shaped only by semantic loss, not by support loss
- The boundary-region features are shaped by both, with support loss potentially overriding semantic-useful features in that critical zone

A binary edge classifier, by contrast, can supervise 100% of points (every point is either "near boundary" or "not near boundary"). This gives the auxiliary task influence over the entire feature space, encouraging globally boundary-aware representations.

### 3c. The continuous regression target fights the sigmoid activation

The support head outputs `sigmoid(logit)` and is supervised with `SmoothL1(sigmoid(logit), target)`. This creates a gradient compression problem:

- For targets near 0 or 1, the sigmoid must be pushed to extreme values, producing vanishing gradients through the sigmoid derivative
- For targets near 0.5, the sigmoid is in its most linear regime, so gradient flows well -- but the Gaussian support field has value 0.5 at distance ~0.033m from the boundary, meaning only a thin shell of points produces efficient gradient
- Points with small support values (0.05-0.2) occupy the steep part of the Gaussian, where the target changes rapidly in space. The sigmoid+SmoothL1 combination produces small gradients for these targets because SmoothL1(sigmoid(x), 0.1) requires pushing sigmoid to a value that's deep in the saturating tail

The net effect: the support loss produces informative gradient only for points very close to the boundary (support_gt > 0.3), and even there, the gradient is dominated by geometric distance encoding rather than class discrimination.

### 3d. focus_mode="none" removes the coupling mechanism

CR-B was run with `focus_mode="none"`, which disables the focus term entirely. The focus term was designed to be the coupling mechanism between support knowledge and semantic supervision: it applies extra Lovasz loss to boundary-region points, weighted by the support field. Without it, the support branch is a pure parasite -- it consumes backbone capacity without returning any benefit to the semantic path.

This is a design-level failure: CR-B tests the support branch as an independent auxiliary task rather than as a coupled boundary-awareness mechanism. The experiment answers "does learning a Gaussian distance field help semantics?" (no) rather than "does boundary-weighted semantic emphasis help semantics?" (untested).

## 4. Is the Problem Target Design or Branch Existence?

### It is primarily target design, secondarily coupling design

The evidence supports a clear ordering:

**Target design is the primary factor.** A binary edge target is naturally aligned with semantic discrimination: the features that detect class transitions directly help classify boundary-region points. The continuous Gaussian support target is geometrically precise but semantically indirect: the features that predict distance-to-boundary do not directly help distinguish classes. The regression nature of the support task also produces persistent, slowly-converging gradient that increasingly dominates the backbone.

**Coupling design is the secondary factor.** Even a suboptimal auxiliary target could help semantics if the coupling mechanism explicitly transfers auxiliary knowledge to the semantic path. CR-B has no coupling: focus_mode="none", no feature fusion, no gating, no conditioning. The support branch learns in isolation and returns nothing to the semantic branch. This compounds the target design problem but is not the root cause -- even with perfect coupling, the support features (distance encoding) may not be what the semantic head needs.

**Branch existence is NOT the problem.** The author's PTv3 reproduction confirms that adding a second head does not inherently hurt semantics -- it can help, IF the auxiliary target is appropriate. The problem is not "two heads are worse than one" but "this specific auxiliary target produces competing rather than complementary gradient."

## 5. Properties That Make an Auxiliary Boundary Task Beneficial vs Harmful

| Property | Beneficial (binary edge) | Harmful (current support) |
|----------|-------------------------|--------------------------|
| **Label simplicity** | 1 bit: boundary/not-boundary. Cheap to learn, fast to converge, stops exerting gradient when confident. | Continuous scalar: truncated Gaussian distance. Slow to converge, persistent gradient, never fully confident. |
| **Class-transition sensitivity** | Directly encodes "is there a class transition here?" Features learn to detect semantic boundaries. | Encodes "how far to nearest geometric primitive." Features learn to measure distance, not detect transitions. |
| **Localization precision** | Coarse: "near boundary" vs "far from boundary." Precision is in the class discrimination, not the boundary location. | Very precise: sigma=0.04m Gaussian. Precision is in the geometric localization, not the class discrimination. |
| **Optimization difficulty** | Low: binary classification with BCE. Converges fast, smooth loss landscape. | High: continuous regression through sigmoid with SmoothL1+Tversky. Gradient compression at extremes, slow convergence. |
| **Gradient compatibility with semantic loss** | High: boundary detection features ARE class-discrimination features near edges. Same feature representation serves both tasks. | Low: distance-field features compete with class-discrimination features for backbone capacity. Different representations needed. |
| **Feature reuse potential** | High: features that say "class transition here" are directly useful for "which class is which." | Low: features that say "0.03m from boundary" need additional processing to become "class A is here, class B is there." |
| **Requirement for explicit feature fusion** | Low: even without fusion, the implicit regularization effect sharpens boundary features. The author's PTv3 reproduction confirms this works with simple head separation. | High: without explicit fusion/gating, the support knowledge never reaches the semantic path. The support head learns alone and contributes only competing gradient. |

## 6. Assessment: Target Design, Coupling Design, or Optimization Design?

### Most plausible explanation: a combination of A and B (with evidence insufficient to fully separate them)

**Explanation A: support is the wrong target** -- plausible, directly supported by the mechanism analysis. The continuous Gaussian distance field demands geometric encoding features that compete with class-discriminative features. A binary edge target would be cheaper, more aligned, and less gradient-dominant. The per-class evidence supports this: support helps balustrade (large, simple geometric boundaries where distance encoding happens to align with class structure) but hurts complex classes (balcony, eave) where geometric precision does not translate to class discrimination.

**Explanation B: support could work, but current integration is wrong** -- also plausible, but harder to confirm without an experiment. The coupling mechanism (focus_mode) was disabled in CR-B. The support features are never fused, gated, or conditioned with the semantic path. There is no gradient management (no warmup, no weight decay schedule, no GradNorm). A sufficiently well-integrated support branch might overcome the target alignment disadvantage through explicit feature transfer. However, the optimization dynamics (44% gradient share, persistent loss floor) suggest that even with perfect coupling, the support task would require careful gradient management to avoid dominating the backbone.

**Explanation C: both are possible, evidence insufficient to separate** -- the honest assessment. Distinguishing "wrong target" from "wrong integration" requires running the same experiment with (i) a binary edge target in the same architecture, and (ii) the support target with proper coupling and gradient management. Without these controls, the relative contribution of target design vs coupling design remains uncertain.

### Ranked conclusions

1. **Target alignment is the strongest explanatory factor.** (HIGH CONFIDENCE) The mechanism analysis clearly shows that binary edge classification produces gradient aligned with semantic discrimination, while continuous support regression produces competing gradient. This is a structural property of the targets, not an integration detail.

2. **Missing coupling compounds the target problem.** (HIGH CONFIDENCE) focus_mode="none" ensures the support branch cannot benefit semantics even if the target were better aligned. But this alone does not explain the failure -- even with coupling, the support features (distance encoding) may not be what the semantic decoder needs.

3. **Optimization policy amplifies the damage.** (MEDIUM CONFIDENCE) Growing gradient share (12% to 44%), no warmup, no curriculum, and persistent support loss floor all make the competition worse than it needs to be. But these are treatable symptoms, not root causes.

4. **The support target is not necessarily worthless, but it is harder to exploit.** (MEDIUM CONFIDENCE) Support is a richer signal than binary edge classification -- it encodes boundary shape, distance, and spatial extent. In principle, this information could improve semantic predictions if properly integrated (e.g., boundary-conditioned refinement at inference time). But as a training supervision signal competing for backbone features, its richness becomes a liability because it demands geometric encoding that the semantic task does not need.

## 7. Concrete Next-Step Experiments

### To test Explanation A (target design):

**Experiment A1: Binary edge baseline in same architecture.** Replace the support head's continuous target with a binary "boundary / not boundary" classification (using `valid_gt` as the target). Keep everything else identical to CR-B. If this recovers or exceeds CR-A performance, the target design hypothesis is confirmed.

**Experiment A2: Simplified support target.** Replace the continuous Gaussian with a coarser signal: `1.0` if within 0.08m, `0.0` otherwise (binary distance threshold). This tests whether the harm comes from the continuous regression or from the auxiliary task's existence.

### To test Explanation B (coupling design):

**Experiment B1: CR-B with focus_mode="lovasz".** Re-enable the coupling mechanism. If performance improves meaningfully toward CR-A, coupling was the missing piece.

**Experiment B2: Gradient-isolated support.** Detach backbone features before the support head. The support head still learns (from its own parameters) but the backbone receives zero support gradient. This isolates whether the support gradient is harmful (if CR-A performance is recovered) or merely unhelpful.

### To separate A from B:

**Experiment AB1: Binary edge target + no coupling.** If this helps (like DLA-Net), the target design is the key factor regardless of coupling. If it also fails, coupling matters more than target choice.

## 8. Synthesis

The current CR-B failure is best explained as a target-coupling interaction:

- The support target demands geometric encoding features that compete with class-discriminative features (**target misalignment**)
- The architecture provides no mechanism for support knowledge to benefit semantic predictions (**missing coupling**)
- The optimization policy allows the support gradient to grow from 12% to 44% of total signal (**gradient domination**)

DLA-Net and the author's PTv3 reproduction succeed because the binary edge target avoids all three problems: it produces semantically-aligned gradient (no target misalignment), it naturally regularizes boundary features even without explicit coupling (low coupling requirement), and its binary classification converges fast enough to avoid gradient domination.

The implication is not that support is fundamentally wrong, but that support-as-training-signal is much harder to make work than binary-edge-as-training-signal. The support field's geometric richness is a strength in principle but a liability in practice because it demands features and gradient resources that compete with the primary semantic objective. Exploiting support effectively would require either (a) explicit architectural coupling that transfers support knowledge to the semantic path without backbone gradient competition, or (b) replacing the continuous regression target with something cheaper and more semantically aligned while retaining the boundary-awareness benefit.

The simplest path forward is to test a binary edge target in the current architecture (Experiment A1). If it helps, it confirms the target alignment hypothesis and provides a working baseline for further refinement. If it also fails, the problem is deeper than target design and lies in the architecture or optimization.
