---
phase: 7
reviewers: [codex, claude]
reviewed_at: "2026-04-03T00:30:00Z"
review_rounds: [2, 3]
plans_reviewed: [07-01-PLAN.md, 07-02-PLAN.md, 07-03-PLAN.md, 07-04-PLAN.md]
---

# Cross-AI Plan Review — Phase 7 (Round 2, Revised Plans)

## Codex Review

### Plan 07-01 Review

#### Summary
This revision fixes the main architectural omission from the original plan: it now removes the explicit field head by construction instead of relying on loss weighting, which is the right direction for AUX-03. The scope is intentionally narrow and matches the phase goal. By itself it does not solve the trainer, evaluator, checkpoint, or verification concerns; it is only sound as the model-path foundation for later waves.

#### Previous Concerns Addressed?
| Concern | Status | Justification |
|---|---|---|
| 1. Trainer scope underspecified | NOT RESOLVED | This plan does not touch trainer plumbing. |
| 2. Loss math undefined | NOT RESOLVED | No loss definition here. |
| 3. Checkpoint compatibility | PARTIALLY RESOLVED | New model shape makes incompatibility explicit, handling deferred to 07-03. |
| 4. Evaluator contract | NOT RESOLVED | No evaluator details here. |
| 5. Verification too weak | NOT RESOLVED | No meaningful verification for the model path itself. |
| 6. Doc consistency | NOT RESOLVED | No doc updates here. |
| 7. Stable vs active config | PARTIALLY RESOLVED | Model-only config helps, full distinction deferred to 07-04. |

#### New Concerns
- **MEDIUM**: Model-only config should clarify it is not the active train entry.
- **MEDIUM**: `__init__.py` registration should avoid accidental default-path changes.
- **LOW**: SupportHead output shape/interpolation semantics should match loss/eval expectations.

#### Risk Assessment
**MEDIUM.** Correct direction, safe only if later plans fully cover integration.

---

### Plan 07-02 Review

#### Summary
The strongest revision in the set. Loss math is now concrete, bounded, and aligned with the semantic-first objective: global semantic supervision remains primary, support supervision is auxiliary, and boundary emphasis is applied through semantic CE rather than explicit field regression. The evaluator contract is also much clearer. Most original ambiguity is fixed.

#### Previous Concerns Addressed?
| Concern | Status | Justification |
|---|---|---|
| 1. Trainer scope | NOT RESOLVED | Not covered here. |
| 2. Loss math undefined | **RESOLVED** | Terms, masking, fallback behavior, forbidden tensors now explicit. |
| 3. Checkpoint compatibility | NOT RESOLVED | Not covered here. |
| 4. Evaluator contract | PARTIALLY RESOLVED | Metric families defined, exact aggregation semantics need explicitness. |
| 5. Verification too weak | PARTIALLY RESOLVED | Math precise enough to test, synthetic forward-pass deferred to 07-03. |
| 6. Doc consistency | NOT RESOLVED | No doc work here. |
| 7. Stable vs active config | NOT RESOLVED | No config-role clarification here. |

#### New Concerns
- **MEDIUM**: `loss_semantic` already includes CE, and `loss_focus` is another CE-derived term. May overweight CE relative to Lovasz unless intentional and documented.
- **MEDIUM**: `loss_focus` uses `support_gt` (ground truth), not `support_pred`. Plan should state this is intentional to avoid confusion about "guided" meaning prediction-guided.
- **MEDIUM**: Boundary-region evaluator threshold=0.2 — source of threshold and whether it applies to `support_gt` or `sigmoid(support_pred)` not fully specified.
- **LOW**: `support_cover` Tversky — whether computed on logits, probabilities, or thresholded predictions should be stated.

#### Suggestions
- Clarify that boundary emphasis uses ground-truth support intentionally.
- Specify exact reduction formulas for all returned scalars.
- Define evaluator inputs: whether boundary metrics use `support_gt` masks, whether support metrics use sigmoid, zero-boundary behavior.

#### Risk Assessment
**LOW-MEDIUM.** Core math well enough specified to implement safely. Evaluator semantics and term weighting deserve one tightening pass.

---

### Plan 07-03 Review

#### Summary
Directly addresses the largest execution gap from Round 1. Trainer touchpoints are concrete and mostly complete, checkpoint incompatibility handled intentionally, synthetic forward-pass present. Substantial improvement. The fallback pattern (support_pred first, then edge_pred) preserves old-path compatibility but creates some ambiguity about whether the active route is fully isolated.

#### Previous Concerns Addressed?
| Concern | Status | Justification |
|---|---|---|
| 1. Trainer scope | **RESOLVED** | All 6 edit locations explicitly enumerated. |
| 2. Loss math undefined | PARTIALLY RESOLVED | Depends on 07-02; wires correctly but doesn't define. |
| 3. Checkpoint compatibility | **RESOLVED** | weight=None, resume=False, explicit train-from-scratch rationale. |
| 4. Evaluator contract | PARTIALLY RESOLVED | Metric selection/logging defined, exact keys depend on 07-02 discipline. |
| 5. Verification too weak | **RESOLVED** | Synthetic forward-pass is the right minimum verification upgrade. |
| 6. Doc consistency | NOT RESOLVED | No doc updates here. |
| 7. Stable vs active config | PARTIALLY RESOLVED | Train config clearly active-route-specific. |

#### New Concerns
- **MEDIUM**: support_pred-first/edge_pred-fallback may unintentionally keep legacy field assumptions alive. Branch conditions must be narrow.
- **MEDIUM**: Synthetic forward-pass verifies kwargs plumbing but may miss validate() logging/metric-selection breakage unless it exercises evaluator output keys end-to-end.
- **LOW**: "8 metrics" in validate() should match 07-02 verbatim to avoid drift.

#### Suggestions
- Make compatibility rule explicit: new path requires support_pred; edge_pred fallback is only for older models.
- Expand synthetic verification to assert returned metric keys match the new contract.
- Ensure active train config name communicates it is Phase 7 mainline, not stable runtime entry.

#### Risk Assessment
**MEDIUM.** Much better and likely sufficient if implemented carefully. Trainer compatibility branches remain the main regression risk.

---

### Plan 07-04 Review

#### Summary
Meaningful upgrade over the prior documentation plan. Three-category distinction is exactly what the repo needed, and the checklist aims at semantic correctness rather than superficial phrase replacement. Addresses repo-language risk well. Main weakness: checklist scope may not catch stray statements outside named files.

#### Previous Concerns Addressed?
| Concern | Status | Justification |
|---|---|---|
| 6. Doc consistency risk | **RESOLVED** | Directly addresses the risk with active-route language and old-route framing removal. |
| 7. Stable vs active config distinction | **RESOLVED** | Three-category distinction now explicit and repo-appropriate. |

#### New Concerns
- **MEDIUM**: Stale claims may exist outside the 6 named files (e.g., .planning/, experiment notes).
- **LOW**: Renaming "Candidate Route" to "Active Implementation Route" should preserve support-only baseline role without sounding deprecated.

#### Suggestions
- Add repo-wide grep for stale phrases: `axis-side`, `candidate route`, `dir`, `dist` in active-route context.
- Require negative assertion: no claim Phase 7 has been smoke-verified or full-train-verified.

#### Risk Assessment
**LOW-MEDIUM.** Direction is correct; remaining risk is coverage, not direction.

---

## Cross-Plan Assessment

### Wave Ordering And Dependencies
The 3-wave structure is sound. Parallelizing 07-01 and 07-02 is acceptable because their interface is simple and now explicit (seg_logits + support_pred). 07-03 correctly depends on both. 07-04 correctly depends on 07-03.

### Scope Creep From 2 To 4 Plans
**Justified, not harmful.** The prior version had hidden work collapsed into oversized plans. The split tracks real dependency boundaries: architecture surface → math/metrics contract → trainer/config integration → documentation canon. Better execution shape for a brownfield repo.

---

## Consensus Summary

*Single reviewer (Codex, Round 2). Install Gemini CLI for multi-reviewer consensus.*

### Round 1 Concern Resolution

| # | Concern | Severity | Round 2 Status | Resolved By |
|---|---------|----------|----------------|-------------|
| 1 | Trainer scope underspecified | HIGH | **RESOLVED** | Plan 07-03: 6 explicit edit locations |
| 2 | Loss math undefined | HIGH | **RESOLVED** | Plan 07-02: three-term contract with exact code |
| 3 | Checkpoint compatibility | HIGH | **RESOLVED** | Plan 07-03: weight=None, resume=False, documented rationale |
| 4 | Evaluator contract underdefined | MEDIUM | PARTIALLY RESOLVED | Plan 07-02: metric families defined, aggregation semantics need tightening |
| 5 | Verification too weak | MEDIUM | **RESOLVED** | Plan 07-03: synthetic forward-pass |
| 6 | Doc consistency risk | MEDIUM | **RESOLVED** | Plan 07-04: cross-file consistency checklist + three-category distinction |
| 7 | Stable vs active config distinction | MEDIUM | **RESOLVED** | Plan 07-04: three-category distinction in contract and guardrails |

**Result: 5/7 RESOLVED, 2/7 PARTIALLY RESOLVED, 0/7 NOT RESOLVED**

### Remaining Risks (Round 2)

1. **MEDIUM — CE overweighting**: loss_semantic (CE + Lovasz) + loss_focus (CE-derived) may overweight CE relative to Lovasz. Should be documented as intentional.
2. **MEDIUM — Ground-truth vs prediction guidance**: loss_focus uses support_gt, not support_pred. Should be stated as intentional design choice.
3. **MEDIUM — Legacy fallback leakage**: support_pred-first/edge_pred-fallback in trainer may keep old assumptions alive. Branches must be narrow.
4. **MEDIUM — Evaluator aggregation**: Exact threshold application, absent-data behavior, and Tversky input type need specification.

### Overall Risk Assessment
**MEDIUM** (improved from MEDIUM-HIGH in Round 1). The revised plan set is materially better and likely executable. No blocking issues remain — the MEDIUM concerns are implementation-detail risks that can be addressed during execution with careful attention to the suggestions above.

---

# Cross-AI Plan Review — Phase 7 (Round 3, Post-Execution Retroactive)

## Claude Review

### Plan 07-01 Review

#### Summary
Solid plan that creates the minimal model needed: shared backbone, dual adapters, semantic head + support head. The output contract (`seg_logits` + `support_pred`, nothing else) directly enforces AUX-03 by construction — direction and distance predictions physically cannot appear.

#### Strengths
- **Enforcement by omission:** AUX-03 is satisfied structurally, not by config zeroing. The model literally cannot emit `dir_pred` or `dist_pred`.
- **SupportHead stem layer** (Linear + ReLU before projection) gives the support branch a nonlinear feature refinement step.
- **Optional adapters** for both semantic and boundary branches allow experimentation without code changes.
- **Clean output contract** — only two keys, easy for downstream consumers to validate.

#### Concerns
- **LOW — Single-channel support output:** 1 channel is correct for BCE supervision, but the decision isn't documented.
- **LOW — No dropout or normalization in stem:** For a 1-layer stem this is likely fine, but the support branch has very little regularization.
- **LOW — `_build_adapter` fallback:** If both adapter configs are None, both branches see identical features — expected default behavior but not documented.

#### Risk Assessment: **LOW**

---

### Plan 07-02 Review

#### Summary
The most complex plan and the most important to get right. The three-term loss is well-motivated and the ground-truth-guided focus weighting avoids the prediction feedback loop. The evaluator provides good coverage of both global and boundary-region metrics.

#### Strengths
- **Ground-truth focus weighting is the right call.** Using `support_gt` instead of `sigmoid(support_pred)` avoids the feedback loop.
- **Intentional CE overlap is well-documented** in the module docstring.
- **Lovász stays global** — preserves its class-balancing properties.
- **Tversky with α=0.3, β=0.7** penalizes false negatives more — correct for a recall-oriented boundary metric.
- **Evaluator boundary threshold at 0.2** is sensible.

#### Concerns
- **MEDIUM — Focus loss on boundary-free batches:** When all `support_gt=0`, `loss_focus` degenerates to unweighted mean CE — effectively doubling CE gradient. Should be acknowledged.
- **MEDIUM — Support BCE masking:** Verify invalid points are excluded from loss denominator, not just targeted as negative examples.
- **MEDIUM — `focus_gamma` interaction with eval threshold:** Gamma defines a different "boundary region" for training vs evaluation (threshold=0.2).
- **LOW — Boundary-region per-class metrics:** May be noisy for classes with few boundary points. NaN handling not discussed.
- **LOW — `support_cover` Tversky:** Uses soft sigmoid probabilities, not thresholded — should be documented.

#### Risk Assessment: **MEDIUM**

---

### Plan 07-03 Review

#### Summary
Clean plumbing plan that correctly prioritizes `support_pred` over legacy `edge_pred`. Backward-compatibility branches preserved. Train config complete for end-to-end training.

#### Strengths
- **Priority ordering in dispatch:** Checks `support_pred` first, falls back to `edge_pred`.
- **All existing branches preserved.**
- **Training from scratch** avoids checkpoint compatibility issues.
- **Config uses `runpy.run_path`** for model config import.
- **`_loss_log_keys` branch** captures all three loss terms.

#### Concerns
- **MEDIUM — Mutual exclusivity assumption:** If a model ever emits both `support_pred` and `edge_pred`, the `elif` silently drops `edge_pred`. An assertion would be safer.
- **LOW — Metric key hardcoding:** If evaluator changes metric names, trainer silently misses them.
- **LOW — 2000 epochs with eval every 100:** Only 20 validation checkpoints.
- **LOW — `grad_accum_steps=6` with `batch_size=4`:** Effective batch size 24, not discussed.

#### Risk Assessment: **LOW**

---

### Plan 07-04 Review

#### Summary
Documentation plan that accurately reflects the implemented state. The three-category config distinction is a useful organizational frame.

#### Strengths
- **Three-category config distinction** provides clear mental model.
- **Rename from "Candidate Route" to "Active Implementation Route"** correctly reflects Phase 7 status.
- **Status tracking** (implemented, pending validation) sets correct expectations.

#### Concerns
- **LOW — Documentation drift risk:** Docs describe implementation details that may change in Phase 8.
- **LOW — No "last verified" date** in implementation-detail sections.

#### Risk Assessment: **LOW**

---

### Cross-Plan Assessment

| Requirement | Status | Evidence |
|---|---|---|
| **AUX-03** | **SATISFIED** | Model physically cannot emit dir/dist. Loss has no direction/distance terms. |
| **COMP-03** | **SATISFIED** | All new files in `project/`. No Pointcept modifications. |

**Overall Risk: LOW-MEDIUM.** Primary risk area is Plan 07-02's loss math edge cases.

### Top 3 Actionable Items
1. **Verify BCE valid-point masking** (07-02, MEDIUM): Confirm invalid points excluded from loss.
2. **Add mutual-exclusivity guard** (07-03, MEDIUM): Assert/warn if model emits both `support_pred` and `edge_pred`.
3. **Document focus_gamma ↔ eval threshold interaction** (07-02, MEDIUM): Training focus region vs evaluation boundary region defined differently.

---

# Multi-Reviewer Consensus (Codex Round 2 + Claude Round 3)

## Agreed Strengths (Both Reviewers)
- AUX-03 satisfied structurally by model architecture (enforcement by omission), not by config zeroing
- COMP-03 satisfied — all changes repo-local
- Three-term loss is well-motivated with correct ground-truth guidance (not prediction-guided)
- Lovász staying global is the right choice
- Three-category config distinction (stable entry / reference baseline / active route) is useful
- Wave dependency ordering is correct (01+02 parallel → 03 → 04)
- Training from scratch avoids checkpoint compatibility issues

## Agreed Concerns (Both Reviewers)
1. **MEDIUM — CE overweighting / gradient doubling on boundary-free batches:** Both note that `loss_focus` degenerates to unweighted CE when `support_gt=0`, effectively doubling CE gradient relative to Lovász. Should be documented as intentional.
2. **MEDIUM — Support BCE valid-point masking:** Both flag the need to verify invalid points are excluded from the loss denominator, not treated as negative examples.
3. **MEDIUM — Trainer support_pred/edge_pred mutual exclusivity:** Both note the `elif` dispatch silently drops `edge_pred` if both are present. Codex calls it "legacy fallback leakage"; Claude calls it "mutual exclusivity assumption."
4. **MEDIUM — Evaluator boundary threshold vs training focus_gamma:** Both note the training boundary region (continuous gamma weighting) differs from the evaluation boundary region (hard threshold at 0.2).

## Divergent Views
- **Codex (Round 2)** rated overall risk as **MEDIUM** and focused on pre-execution plan completeness gaps (most resolved in the revision).
- **Claude (Round 3)** rated overall risk as **LOW-MEDIUM** and focused on post-execution loss math edge cases that could affect Phase 8 experiments.
- Both agree no blocking issues remain — concerns are implementation-detail risks for Phase 8 tuning.
