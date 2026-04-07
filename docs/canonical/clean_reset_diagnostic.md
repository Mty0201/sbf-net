# Diagnostic Memo: Why Support Supervision Underperforms in the Current Implementation

**Date:** 2026-04-07
**Status:** Open diagnosis -- not a closure memo. The dual-supervision concept remains plausible.
**Prerequisite:** Read `docs/canonical/clean_reset_analysis.md` for the raw comparison data.

## Framing

The clean-reset experiment shows that the current implementation of support supervision is a net negative for semantic segmentation: CR-A (semantic-only) outperforms CR-B (support-only) by +1.52pp best mIoU under controlled conditions.

This result does NOT demonstrate that semantic-edge dual supervision is fundamentally unworkable. Prior art (DLA-Net, arXiv:2106.00376) and the author's own successful reproduction of joint semantic + semantic-edge supervision on PTv3 in the same building-facade domain confirm the concept's feasibility. The negative result should be treated as evidence against the **current realization**, not against the underlying research direction.

This memo diagnoses the gap between "known to be feasible" and "currently failing" by examining specific implementation, coupling, and optimization failure modes.

## Ranked Failure Hypotheses

### Hypothesis 1: Gradient Competition Without Gradient Benefit (HIGH CONFIDENCE)

**The problem:** The support loss backpropagates freely through the shared backbone with no gradient isolation, no gradient scaling, and no delayed activation. By epoch 100, support loss consumes 44% of total gradient despite producing no measurable semantic benefit. The backbone is forced to split its representational capacity between two tasks that happen to share features but do not share objectives.

**Implementation evidence:**
- `SharedBackboneSemanticSupportModel.forward()` branches `feat` to both heads with no `.detach()` or stop-gradient
- `RedesignedSupportFocusLoss.forward()` sums semantic and support losses into a single scalar before backward
- Support loss share grows from 12% (epoch 1) to 44% (epoch 100) as semantic loss decreases faster
- No learning rate scaling, gradient clipping, or normalization differentiates the two gradient streams

**Why this is the top hypothesis:** In successful multi-task learning, auxiliary gradients either (a) are explicitly controlled (GradNorm, uncertainty weighting, PCGrad) or (b) naturally align with the primary task. Neither holds here. The support task is a continuous regression problem (SmoothL1 on a Gaussian field) with fundamentally different gradient geometry than 8-class cross-entropy. Raw additive combination at the backbone produces gradient interference, not gradient synergy.

**Comparison to feasible prior art:** DLA-Net and the author's successful PTv3 reproduction likely differ in how the auxiliary task's gradient interacts with the backbone -- either through explicit weight balancing, a different loss scale, or architectural isolation that prevents the auxiliary gradient from dominating shared features.

**Test experiments:**
1. **Gradient isolation:** Detach backbone features before the support head (`support_pred = support_head(feat.detach())`). If semantic performance recovers to CR-A level, the gradient competition hypothesis is confirmed. The support head still learns (from its own parameters), but the backbone is protected.
2. **GradNorm / uncertainty weighting:** Apply dynamic loss weighting (e.g., Kendall et al. 2018, or GradNorm) to automatically balance semantic and support gradient magnitudes at the backbone.
3. **Delayed auxiliary activation:** Train semantic-only for the first N epochs, then activate support supervision. This lets the backbone establish strong semantic features before introducing the auxiliary gradient.

---

### Hypothesis 2: Auxiliary Task Is Decoupled From the Semantic Objective (HIGH CONFIDENCE)

**The problem:** The support head learns the boundary field (support_cover = 0.63), but this learned knowledge does not flow back to improve semantic predictions. The two heads are architecturally independent: they share backbone features but never exchange task-specific information. The support head's output is used only for its own loss computation -- it never modulates, gates, or conditions the semantic head.

**Implementation evidence:**
- Both adapters are `nn.Identity()` in CR-B config -- no task-specific feature conditioning
- `forward()` computes `seg_logits = semantic_head(feat)` and `support_pred = support_head(feat)` independently
- The focus mechanism (which would use support predictions to weight semantic loss) is disabled: `focus_mode="none"`
- No attention, gating, or feature fusion between the two branch outputs

**Why this matters:** The theoretical value of boundary-aware supervision is that boundary knowledge should improve the semantic representation -- not just coexist with it. In the current design, the support head is a parasite: it consumes backbone capacity (gradient competition) without returning any benefit to the semantic path. The only mechanism for benefit would be implicit -- the hope that support-gradient-modified backbone features happen to also be better for semantics. The evidence shows this implicit transfer does not occur.

**Comparison to feasible prior art:** Successful boundary-aware architectures typically feature explicit coupling: boundary predictions gating semantic features, attention mechanisms that use edge information to refine class boundaries, or decoder-level fusion where boundary and semantic streams interact before the final prediction. The current design has none of these.

**Test experiments:**
1. **Re-enable focus mode:** Run CR-B with `focus_mode="lovasz"`, which applies additional semantic Lovasz loss weighted by the boundary region. This creates a direct coupling: boundary knowledge modulates semantic supervision intensity.
2. **Boundary-conditioned feature fusion:** Add a lightweight cross-attention or gating module where support predictions modulate semantic features before the semantic head. This makes the support branch's output directly useful for semantics.
3. **Feature adapter activation:** Enable `ResidualFeatureAdapter` for both heads (currently `None`). This gives each head a task-specific feature conditioning layer, allowing the backbone to maintain shared representations while each head adapts them.

---

### Hypothesis 3: The CR-B Experiment Design Removes the Coupling Mechanism (MEDIUM-HIGH CONFIDENCE)

**The problem:** CR-B uses `focus_mode="none"`, which disables the focus term entirely. The focus term is the mechanism that was designed to make support knowledge useful for semantics -- it applies boundary-weighted Lovasz loss to emphasize semantic accuracy in boundary regions. Without it, CR-B tests "does adding a support regression head to the backbone help semantics?" rather than "does boundary-aware semantic emphasis help semantics?"

**Implementation evidence:**
- `focus_mode="none"` in CR-B config
- `RedesignedSupportFocusLoss.forward()`: when focus_mode is "none", `loss_focus = 0.0` and the focus term is never added to total_loss
- The entire rationale for the support branch was to enable focus weighting -- testing the support branch without focus is testing a partial implementation

**Why this matters:** The clean-reset comparison answers "does multi-task learning with support regression help?" (answer: no). But the real hypothesis is "does boundary-aware semantic emphasis help?" -- which requires the focus mechanism to be active. The current experiment is too harsh on the support branch because it removes the very mechanism that was supposed to make support useful.

**Caveat:** This does not explain why the old support-only baseline (with a different loss formulation) achieved 74.6 mIoU in uncontrolled runs. The old result used different seeds, different training duration, and potentially different hyperparameters. The clean-reset comparison is more reliable, but the old result suggests that under some configurations, support supervision can at least not harm semantic performance.

**Test experiments:**
1. **CR-C: support + focus:** Same as CR-B but with `focus_mode="lovasz"` and `focus_weight=0.5`. This tests the complete system as designed.
2. **CR-D: support + focus + reduced support weight:** `focus_mode="lovasz"`, `support_reg_weight=0.1`, `support_cover_weight=0.05`. Combines focus activation with gradient competition mitigation.

---

### Hypothesis 4: Loss Scale Mismatch and Optimization Policy (MEDIUM CONFIDENCE)

**The problem:** The semantic loss (CE + Lovasz) and support loss (SmoothL1 + Tversky) operate on fundamentally different scales and with different convergence dynamics. The semantic loss drops from 1.38 to 0.14 over 100 epochs (90% reduction), while the support loss drops from 0.19 to 0.11 (40% reduction). As semantic loss converges, support loss becomes relatively larger, increasing gradient competition precisely when the model should be fine-tuning semantic features.

**Implementation evidence:**
- Epoch 1: semantic = 88.2%, support = 11.8% of total loss
- Epoch 100: semantic = 56.4%, support = 43.6% of total loss
- No loss normalization, no gradient magnitude balancing
- SmoothL1 on sigmoid output has different gradient properties than CE on logits
- OneCycleLR applies the same schedule to both tasks (no per-task LR)

**Specific issues:**
- **No warmup for support:** Support loss is active from epoch 1, when the backbone features are still random. Early support gradients may push the backbone toward support-favorable representations that are hard to undo later.
- **No curriculum:** The support task difficulty is constant (same Gaussian targets throughout training). A curriculum that starts with easy/coarse boundary detection and progressively sharpens could help.
- **Tversky with beta=0.7:** Heavy false-negative penalty may cause the support head to predict high support values everywhere, degrading the quality of the boundary signal.

**Test experiments:**
1. **Support warmup:** Zero support loss weight for the first 20 epochs, then ramp linearly to full weight over epochs 20-40. Lets backbone establish semantic features first.
2. **Cosine support weight schedule:** Start support weight at 0.0, peak at 0.2 mid-training, decay to 0.05 late. Prevents late-epoch gradient domination.
3. **Per-task LR:** Use a lower learning rate for the support head (e.g., 0.1x the backbone LR) to reduce its influence on shared features.

---

### Hypothesis 5: Support Supervision Target May Be Misaligned With Semantic Benefit (MEDIUM CONFIDENCE)

**The problem:** The support field is a truncated Gaussian distance field (sigma=0.04m, radius=0.08m) centered on fitted geometric boundaries. This is a geometrically precise signal, but it may not be the right supervision target for improving semantic segmentation near edges.

**Specific concerns:**
- **Too narrow:** At sigma=0.04m, the Gaussian decays to near-zero within ~0.08m of the boundary. Many semantic boundary errors occur further from the geometric edge -- e.g., a misclassified region 0.15m from the true boundary. The support field doesn't supervise these points.
- **Too geometric:** The support field treats all boundary points equally regardless of semantic difficulty. A geometrically sharp wall-column boundary and a difficult balcony-wall boundary get the same support signal, but the semantic model struggles much more with the latter.
- **Regression vs. classification mismatch:** The support head regresses a continuous Gaussian value, which is a fundamentally different task from semantic classification. The features useful for "how far is this point from the nearest boundary?" may not be the same features useful for "which class does this point belong to?"

**Implementation evidence:**
- Support_cover reaches 0.63 -- the model can predict boundaries reasonably well -- but this doesn't help semantics
- Per-class analysis shows support helps balustrade (large, regular boundaries) but hurts balcony (irregular, difficult boundaries), consistent with the hypothesis that the geometric supervision target doesn't capture semantic difficulty

**Test experiments:**
1. **Wider support field:** Increase sigma to 0.1m or 0.15m to capture a broader boundary transition zone.
2. **Class-weighted support:** Scale support loss by per-class difficulty (e.g., inverse IoU weighting). Classes with low IoU get stronger boundary supervision.
3. **Binary boundary detection instead of regression:** Replace the continuous Gaussian target with a binary "near boundary / not near boundary" classification. This may produce features more aligned with the semantic task.

---

### Hypothesis 6: Architectural Mismatch -- Head Depth and Feature Level (MEDIUM CONFIDENCE)

**The problem:** The support head is very shallow (1-layer stem + linear projection). It attaches to the final decoder output (64-dim), which is the same feature level as the semantic head. This means:
- The support head cannot learn complex boundary patterns -- it's essentially a linear classifier on backbone features
- Both tasks compete for the same feature level with no multi-scale interaction
- The support task may need earlier/deeper features (fine-grained spatial information) while semantics benefits from later/higher-level features

**Implementation evidence:**
- SupportHead: `Linear(64,64) + ReLU + Linear(64,1)` -- 2 layers total
- SemanticHead: `Linear(64,8)` -- 1 layer
- Both operate on the same 64-dim feature vector from the final decoder stage
- No skip connections, no multi-scale fusion, no FPN-like structure for the support branch

**Comparison to feasible prior art:** DLA-Net uses a dedicated boundary detection branch with its own feature hierarchy. Successful semantic-boundary architectures typically attach the boundary head at multiple scales or use feature pyramid structures that give the boundary task access to fine-grained spatial features while the semantic task uses coarser features.

**Test experiments:**
1. **Deeper support head:** Replace the 2-layer MLP with a 3-4 layer MLP or a small U-Net-like structure.
2. **Multi-scale support:** Attach the support head to intermediate encoder features (not just the final decoder output).
3. **Skip-connection support:** Feed both early (high-resolution) and late (high-semantic) features to the support head.

---

### Hypothesis 7: The Old Uncontrolled Results Were Misleading (LOW-MEDIUM CONFIDENCE)

**The problem:** The canonical fact "support-only = 74.6, semantic-only = 73.8" was established under uncontrolled conditions (different seeds, possibly different training durations, variable hyperparameters). The clean-reset experiment reverses this ranking: CR-A = 73.36, CR-B = 71.84. This raises the possibility that the old support advantage was a seed/hyperparameter artifact rather than evidence of genuine benefit.

**Evidence:**
- Phase 12 (v1.1) already retracted the old baseline narrative as unreliable
- The clean-reset comparison is the first controlled head-to-head comparison
- Both old results (73.8, 74.6) are higher than both clean-reset results (73.36, 71.84), suggesting the old runs may have used luckier seeds or hyperparameter choices

**Implication:** If the old support-only advantage was artifactual, then the bar for proving support supervision helps is higher than previously assumed. The concept may still be feasible, but prior evidence from this specific codebase is weaker than believed.

**This does NOT invalidate the concept:** The author's independent successful reproduction on PTv3 and the DLA-Net prior art are separate, stronger evidence that the concept works in the domain. The old SBF-net numbers are just weaker support.

---

### Hypothesis 8: Class-Specific Effects Reveal a Coupling Problem, Not a Concept Problem (MEDIUM CONFIDENCE)

**The problem:** CR-B helps balustrade (+5.64pp at epoch 100) and marginally helps window (+0.91pp), but hurts balcony (-9.95pp), eave (-3.29pp), advboard (-2.01pp), column (-1.40pp), wall (-0.80pp), and clutter (-0.54pp). This pattern is not random.

**Analysis:**
- **Balustrade** is a large, regular class with well-defined geometric boundaries (walls, rails). Support supervision correctly identifies these boundaries, and the backbone features that predict "distance to wall-balustrade boundary" also happen to improve wall-balustrade classification.
- **Balcony** is rare, irregularly shaped, and often surrounded by multiple other classes. Support supervision on balcony boundaries is noisy (many short, complex boundary segments), and the gradient from these noisy boundary predictions degrades the already-fragile balcony features.
- **Eave** is geometrically thin and boundary-dominated. One would expect boundary supervision to help here, but it doesn't -- possibly because the eave support field is too narrow (sigma=0.04m) relative to the eave's own geometry.

**Implication:** The auxiliary signal IS useful for some classes (large, regular, well-bounded) but harmful for others (small, irregular, boundary-complex). A blanket auxiliary loss applied uniformly across all classes and all boundary types produces a net negative because the harmful effects on difficult classes outweigh the benefits on easy ones.

**Test experiments:**
1. **Class-conditional support loss:** Only compute support loss for boundary segments involving specific class pairs where the benefit is expected (e.g., wall-balustrade, wall-column). Zero out support loss for class pairs where it hurts (e.g., balcony-*).
2. **Adaptive per-class support weighting:** Scale support loss contribution by inverse class frequency or by recent per-class IoU delta.

## Summary: Concept vs. Implementation

| Aspect | Concept Status | Implementation Status |
|--------|:-------------:|:--------------------:|
| Semantic-edge dual supervision feasibility | **Plausible** (DLA-Net, author's PTv3 reproduction) | **Current realization fails** |
| Boundary-aware feature learning | **Theoretically beneficial** | **Not tested** (focus_mode="none" disables the coupling mechanism) |
| Multi-task gradient management | **Well-studied, solvable** | **Not implemented** (raw additive loss, no balancing) |
| Auxiliary-to-primary knowledge transfer | **Requires explicit design** | **Not implemented** (no fusion, no gating, no conditioning) |
| Supervision target design | **Domain-appropriate** | **Possibly too narrow/geometric** (sigma=0.04m Gaussian) |
| Optimization policy | **Needs per-task tuning** | **Single policy for both tasks** (no warmup, no curriculum) |

## Prioritized Next Steps

**Tier 1 -- Directly tests the top failure hypothesis with minimal code change:**
1. Re-run CR-B with `focus_mode="lovasz"` and `focus_weight=0.5` (tests H3: was focus the missing coupling?)
2. Re-run CR-B with `feat.detach()` before support head (tests H1: is gradient competition the problem?)
3. Re-run CR-B with support weight warmup: 0 for epochs 1-20, linear ramp to 1.0 over epochs 20-40 (tests H4: is early gradient interference the problem?)

**Tier 2 -- Requires moderate code change, tests architecture hypotheses:**
4. Add GradNorm or uncertainty-based dynamic loss weighting (tests H1 + H4)
5. Add cross-branch feature gating: support predictions condition semantic features (tests H2)
6. Enable `ResidualFeatureAdapter` for both heads (tests H2 + H6)

**Tier 3 -- Requires design rethinking, tests supervision target hypotheses:**
7. Widen support field (sigma=0.1m) and test (tests H5)
8. Class-conditional support loss (tests H8)
9. Replace continuous Gaussian regression with binary boundary classification (tests H5)

## What This Memo Does NOT Conclude

- It does NOT conclude that semantic-edge dual supervision is unworkable.
- It does NOT recommend abandoning the research direction.
- It does NOT advance the milestone beyond a diagnostic/reasoning state.
- It does NOT treat the clean-reset result as evidence against the concept -- only against the current realization.

The clean-reset result is a negative signal about **this specific implementation** (raw additive multi-task loss, no coupling mechanism, no gradient management, shallow auxiliary head). The concept of strengthening semantic segmentation with boundary-aware supervision remains supported by independent evidence.
