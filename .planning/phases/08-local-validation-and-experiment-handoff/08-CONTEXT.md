# Phase 8: Local Validation And Experiment Handoff - Context

**Gathered:** 2026-04-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Run smoke/sample validation for the support-guided semantic focus route (implemented in Phase 7) and produce the next full-train experiment direction with clear evidence boundaries. This phase does NOT run full training or implement new supervision variants — it validates the existing implementation and documents what to try next.

</domain>

<decisions>
## Implementation Decisions

### Smoke validation scope
- **D-01:** Smoke validation covers model forward, loss forward, and backward pass on the active route — no full-epoch training required.
- **D-02:** Validation must confirm `support_pred` and `seg_logits` are both present and correctly shaped in model output.
- **D-03:** All three loss terms (semantic, support, focus) must be non-NaN.
- **D-04:** A new smoke config is required for the active route (`*-support-guided-semantic-focus-*-smoke.py`) since none exists yet.

### Validation pass criteria
- **D-05:** Pass criteria are: (1) non-NaN losses for all three terms, (2) correct output keys (`seg_logits`, `support_pred`), (3) backward + optimizer step OK, (4) focus activation check.
- **D-06:** Focus activation check: in regions where `support_gt > 0.2`, mean `focus_weight` must be significantly higher than in other regions — proving boundary-region weighting actually activates rather than being all zeros.

### Experiment direction content
- **D-07:** Core ablation direction (top priority): soft masking with `support_gt` continuous weight vs current hard `valid_gt` mask.
- **D-08:** Negative sample calibration: document the risk that removing `valid_gt` masking causes support prediction over-activation (false positives in non-boundary regions).
- **D-09:** Alpha-sigma coupling: document the relationship between soft-masking exponent `alpha` and Gaussian `sigma` used to generate `support_gt` — record for future investigation, do not implement in this phase.
- **D-10:** Future adaptive route: inference-time use of `support_pred` instead of GT-derived focus weight — document as a direction only, do not implement.
- **D-11:** Writeup style: give directions and open questions, do not lock hyperparameter values. The handoff should frame what to explore, not prescribe exact settings.

### Evidence boundaries
- **D-12:** Local validation proves "runs correctly" (shapes, non-NaN, gradients flow, focus activates). It does NOT prove "works well" or claim full-train performance. This distinction must be explicit in all validation docs (COMP-04 requirement).

### Claude's Discretion
- Exact structure of the smoke config (epoch count, batch limits, etc.) — follow existing smoke config conventions
- How to implement the focus activation check (inline in script or separate diagnostic)
- Organization of the experiment handoff document

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Active route implementation (Phase 7)
- `project/models/semantic_model.py` — SharedBackboneSemanticSupportModel, SupportHead
- `project/losses/support_guided_semantic_focus_loss.py` — SupportGuidedSemanticFocusLoss (loss math, focus weighting logic)
- `project/evaluator/semantic_boundary_evaluator.py` — SupportGuidedSemanticFocusEvaluator
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-model.py` — Active route model config
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train.py` — Active route train config

### Canonical docs
- `docs/canonical/sbf_semantic_first_route.md` — Route definition, baseline (support-only 74.6), prohibition rules
- `docs/canonical/sbf_semantic_first_contract.md` — Support-centric candidate-route contract
- `docs/canonical/sbf_training_guardrails.md` — Training/config guardrails

### Existing smoke/validation patterns
- `scripts/check_data/check_model_forward.py` — Model forward validation pattern
- `scripts/check_data/check_loss_forward.py` — Loss forward validation pattern
- `scripts/check_data/check_validation_step.py` — Validation step pattern (model + loss + evaluator)
- `scripts/train/check_train_step.py` — Backward + optimizer step pattern
- `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-axis-side-train-smoke.py` — Existing smoke config convention

### Trainer and dataset
- `project/trainer/trainer.py` — Training loop, checkpoint selection by val_mIoU
- `project/datasets/bf.py` — BFDataset, edge.npy loading, edge_support_id.npy

### Requirements
- `.planning/REQUIREMENTS.md` — VAL-01, VAL-02, COMP-04

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Existing smoke scripts (`check_model_forward.py`, `check_loss_forward.py`, `check_train_step.py`) provide the structural template — need adaptation for `SharedBackboneSemanticSupportModel` and `SupportGuidedSemanticFocusLoss`
- Existing smoke configs (e.g., `*-axis-side-train-smoke.py`) show the convention: reduce epochs, limit batches, use same data path
- Sample fixtures in `samples/training/020101` and `samples/validation/020102` are available for validation

### Established Patterns
- Smoke scripts use `bootstrap_paths()` + `runpy.run_path(...)` for config loading
- Pass/fail is determined by non-exception exit, expected output keys, non-NaN metrics, and `backward_ok: True` / `optimizer_step_ok: True`
- Smoke configs append `-smoke.py` to the training config name and reduce epoch/batch counts

### Integration Points
- New smoke config inherits from the active route model config (`*-support-guided-semantic-focus-model.py`)
- Smoke/check scripts must import `project.models` for registry side effects (SharedBackboneSemanticSupportModel registration)
- Trainer dispatch checks `support_pred` before `edge_pred` for the active route

</code_context>

<specifics>
## Specific Ideas

- Focus activation check uses `support_gt > 0.2` threshold to define "boundary region" and compares mean focus_weight inside vs outside
- Experiment handoff should frame directions as questions to investigate, not prescribe hyperparameter values
- Soft masking formula reference: `loss = BCE(support_logit, support_gt) * support_gt.pow(alpha)` — this is the core ablation idea
- Future adaptive route idea: at inference time, replace GT-derived focus weight with `support_pred`-derived focus weight

</specifics>

<deferred>
## Deferred Ideas

- Implementing soft masking as an actual code variant — belongs in a future phase after Phase 8 handoff
- Implementing inference-time adaptive focus weighting — future phase
- Alpha-sigma coupling experiments — future phase
- Formal pytest-based test suite for route contracts (FOLL-04) — deferred to v2

</deferred>

---

*Phase: 08-local-validation-and-experiment-handoff*
*Context gathered: 2026-04-03*
