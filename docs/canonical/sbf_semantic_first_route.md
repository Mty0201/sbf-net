# Semantic-First Route Definition

## Purpose

This document defines the evidence-aligned semantic-first route selection rules for milestone `v1.1`.

As of Phase 8, the candidate route is implemented and locally smoke-validated. Local validation confirms the route runs correctly (non-NaN losses, gradient flow, focus activation). Full-train validation is pending — see `docs/canonical/sbf_validation_and_experiment_handoff.md` for the experiment handoff. This document records:

- the strongest current semantic-first reference baseline
- the weaker side evidence that should not be promoted to the new mainline
- the active implementation route and its prohibition rules

## Current Reference Baseline

- `support-only is the strongest current reference baseline`.
- The current reference point is `support-only (reg=1, cover=0.2) = 74.6`.
- The baseline matters because it improved semantic performance without forcing the model into explicit geometric-field learning.
- This baseline is the comparison target the next semantic-first route should beat.

## Side Evidence

- `support-shape is weaker side evidence only`.
- It is not the canonical semantic-first route.
- The current interpretation is that its extra shape supervision pressure still hurts the semantic objective relative to the lighter-touch support-only baseline.

## Active Implementation Route

The active implementation route is the **support-guided semantic focus route**, implemented in Phase 7.

Implementation: `SharedBackboneSemanticSupportModel` + `SupportGuidedSemanticFocusLoss` + `SupportGuidedSemanticFocusEvaluator`. Config: `configs/semantic_boundary/old/semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train.py`.

Its definition is:

- support remains the only explicit boundary-side prediction target
- the route must improve semantic behavior near boundaries relative to the support-only baseline
- any added structure must serve semantic quality rather than learn a richer geometric field

The route explicitly prohibits these as mainline supervision targets:

- no direction target
- no side target
- no distance target
- no coherence target that recreates geometric pressure indirectly
- no ordinal shape pressure as the mainline objective

## Validation Status

- **Local smoke validation:** Complete (Phase 8). Confirms forward/backward/optimizer pass, non-NaN losses, focus activation.
- **Full-train validation:** Not started. The comparison target is the support-only baseline (val_mIoU = 74.6).
- **Evidence boundary:** Local validation proves "runs correctly." It does NOT prove "works well" or claim any performance improvement.
- **Handoff:** `docs/canonical/sbf_validation_and_experiment_handoff.md`

## Route-Selection Rules

Choose the semantic-first route by these rules:

1. Start from the support-only baseline rather than from weaker exploratory routes.
2. Preserve semantic segmentation as the governing objective.
3. Add only support-centric or semantic-serving structure that can plausibly outperform support-only.
4. Reject any route whose extra supervision pressure becomes a new geometric learning target.

## Historical Reference Boundary

The following remain comparison evidence only:

- `axis-side`
- `support + axis + side`
- `support-shape`
- other direction, coherence, or geometry-centered variants

They remain useful for understanding failure modes, but they do not define the candidate semantic-first mainline.

## Next Link

For the exact Phase 6 candidate-route contract, read [docs/canonical/sbf_semantic_first_contract.md](./sbf_semantic_first_contract.md).
