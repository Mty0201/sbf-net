# Semantic-First Candidate Route Contract

## Purpose

This document defines the contract for the **support-guided semantic focus route**, originally defined in Phase 6 and implemented in Phase 7.

It is a route-definition artifact, not a claim that the route is full-train verified.

## Baseline And Goal

- `support-only` is the strongest current reference baseline.
- This contract uses the `support-only baseline` as the comparison target.
- The candidate route must improve on that baseline under the semantic-first objective.
- The route must stay support-centric and avoid the extra supervision pressure seen in weaker side evidence such as `support-shape`.

## Core Contract

- support remains the only explicit boundary prediction target
- semantic segmentation remains the governing objective
- any additional structure must operate to improve semantic behavior near boundaries
- the route must not add a new geometric target just because that target is available in historical artifacts

The candidate route explicitly forbids:

- no direction target
- no side target
- no distance target
- no coherence target as a mainline objective
- no ordinal shape pressure as a mainline objective

## Architecture Boundary

The candidate route must keep the repo-local architecture minimal:

- keep the backbone and main training architecture largely intact
- keep the existing trainer entrypoint at `scripts/train/train.py`
- keep the existing dataset contract and `edge.npy` format unchanged during route definition
- keep all changes inside `semantic-boundary-field`
- no Pointcept changes

## Runtime Contract Shape

The route definition assumes:

- stable runtime entry config remains `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py`
- historical reference configs remain historical only
- The route now has its own repo-local model, loss, evaluator, and config implemented in Phase 7: `SharedBackboneSemanticSupportModel`, `SupportGuidedSemanticFocusLoss`, `SupportGuidedSemanticFocusEvaluator`, `semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train.py`. The stable runtime entry config remains unchanged.

The implementation satisfies the semantic-first prohibitions above.

## Three-Category Config Distinction

Maintainers must distinguish three config roles:

1. **Stable runtime entry config**: `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-train.py` — unchanged by Phase 7.
2. **Strongest reference baseline**: `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-edge-support-only-train.py` — the comparison target, not the active route.
3. **Active implementation route**: `configs/semantic_boundary/semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train.py` — Phase 7 implementation.

These are intentionally separate. The stable entry config is the repository's default runtime path. The reference baseline is the evidence comparison target. The active implementation route is the new semantic-first path being evaluated.

## Relationship To Support-Shape

`support-shape` remains side evidence only.

It is weaker than the support-only baseline and therefore does not define this candidate route. The candidate route should be lighter-touch than support-shape with respect to auxiliary supervision pressure.

## Runtime Guidance Link

Maintainer-facing runtime guidance should point back to this file when describing the Phase 6 semantic-first candidate route.
Route-definition context lives in [docs/canonical/sbf_semantic_first_route.md](./sbf_semantic_first_route.md).
