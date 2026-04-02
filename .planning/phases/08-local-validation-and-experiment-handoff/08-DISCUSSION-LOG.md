# Phase 8: Local Validation And Experiment Handoff - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-03
**Phase:** 08-local-validation-and-experiment-handoff
**Areas discussed:** Smoke validation scope, Experiment direction content, Validation pass criteria

---

## Smoke Validation Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Model forward + loss forward + backward | Cover model forward, loss forward, backward pass; verify support_pred and seg_logits | |
| Full short-epoch smoke run | Run a few complete training epochs end-to-end | |
| Minimal forward-only | Just model forward to check shapes | |

**User's choice:** Model forward + loss forward + backward pass. Must also verify focus weighting activates in boundary regions (not all zeros). No full epoch needed. New smoke config required.

**Notes:** User specified that all three loss terms (semantic, support, focus) must be checked for NaN. Existing smoke scripts need adaptation for the new model/loss types.

---

## Experiment Direction Content

| Option | Description | Selected |
|--------|-------------|----------|
| Specific directions with open questions | Frame ablation directions and research questions without locking hyperparameters | ✓ |
| Prescriptive experiment plan | Lock down exact hyperparameter grids and experiment sequence | |
| Minimal pointers | Just list topic areas without detail | |

**User's choice:** Four directions, priority-ordered:
1. Core ablation: soft masking (continuous `support_gt` weight) vs hard `valid_gt` mask — top priority
2. Negative sample calibration: support over-activation risk without `valid_gt`
3. Alpha-sigma coupling: document relationship, don't implement
4. Future adaptive route: inference-time `support_pred` for focus weight — document only

**Notes:** User explicitly said "give directions and questions, don't lock hyperparameter values." The adaptive route idea (inference-time support_pred replacing GT) was flagged as a future direction to document but not implement.

---

## Validation Pass Criteria

| Option | Description | Selected |
|--------|-------------|----------|
| Non-NaN + correct keys + backward OK | Standard smoke pass criteria | |
| Above + focus activation check | Add boundary-region focus weight verification | ✓ |
| Above + metric sanity | Also check evaluator metric ranges | |

**User's choice:** Four checks: (1) non-NaN losses for all three terms, (2) correct output keys, (3) backward + optimizer step OK, (4) focus activation: mean focus_weight in `support_gt > 0.2` regions significantly higher than elsewhere.

**Notes:** The focus activation check is the key addition beyond standard smoke criteria — it verifies the novel contribution of the support-guided semantic focus route actually works.

---

## Claude's Discretion

- Smoke config structure (epochs, batch limits) — follow existing conventions
- Focus activation check implementation approach
- Experiment handoff document organization

## Deferred Ideas

- Implementing soft masking variant — future phase
- Implementing adaptive inference-time focus weighting — future phase
- Alpha-sigma coupling experiments — future phase
