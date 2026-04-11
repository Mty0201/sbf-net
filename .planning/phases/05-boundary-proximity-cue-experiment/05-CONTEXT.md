# Phase 5 Context: Boundary Proximity Cue Experiment (CR-C)

> **Retrospective note — 2026-04-11 (v2.0 Phase 8 precondition shift)**
>
> The verdict and analysis below were formed under the precondition `grid_size = 0.06`.
> On 2026-04-11 direct measurement of the training transform pipeline showed that grid=0.06
> rasterises the r=0.06 m boundary band to ~1 voxel thickness (6.75% of non-b voxels within
> 6 cm; b→nearest-nonb p50 = 6.03 cm). At grid=0.04 this rises to 10.28% (+52%). This verdict
> is therefore **gated on grid=0.06**; re-validation at grid=0.04 is pending in Phase 8
> (CR-R..CR-U). Do not read the body below as "disproven" — it remains correct for the
> precondition under which it was measured. Phase 5 closed 2026-04-11 with verdict
> 'aux-head route ceilings at ~0.725 across single/dual-stream architectures at grid=0.06';
> the verdict is preserved; re-validation at grid=0.04 is Phase 8. See
> `.planning/phases/08-grid04-range-shift-validation/08-CONTEXT.md`.

**Phase:** 5
**Milestone:** v2.0
**Status:** Ready for planning
**Prerequisite reading:** `docs/canonical/route_redesign_discussion.md` (Section 5 — Part 1 Implementation Design)

## Goal

Validate that treating support as a boundary proximity indicator (confidence-weighted BCE classification) produces a positive semantic effect, matching or exceeding CR-A (semantic-only, best mIoU = 0.7336).

## What This Phase Does

1. **Implement a new loss function** that treats `valid` as a binary boundary/not-boundary label and `support` as a confidence weight for boundary points. This is Option L1 from the route redesign discussion:

   ```
   auxiliary_target = valid                          # binary: boundary or not
   confidence_weight = where(valid, support, 1.0)    # Gaussian weight for boundary points, 1.0 for non-boundary
   loss_aux = weighted_BCE(support_pred, valid, weight=confidence_weight)
   ```

   Properties:
   - 100% of points supervised (binary: every point is boundary or not)
   - Boundary points weighted by support confidence (closer to boundary core = higher weight)
   - Non-boundary points contribute with weight 1.0 (easy negatives, converge fast)
   - Pure classification: no regression, no metric distance encoding
   - Gradient naturally decays as classifier becomes confident

2. **Create a CR-C training config** identical to CR-B except:
   - Loss function: confidence-weighted BCE (replaces SmoothL1+Tversky)
   - Auxiliary loss weight: 0.2-0.5 (not 1.0)
   - Everything else identical: same seed (38873367), same backbone, same data, same optimizer, same 100 epochs

3. **Run CR-C for 100 epochs** and produce a structured comparison against CR-A.

4. **Evaluate against success criterion:** CR-C mIoU >= CR-A (0.7336).

## What This Phase Does NOT Do

- No direction field prediction (columns 0-2 of edge.npy)
- No distance field prediction (column 3 of edge.npy)
- No geometric field regression of any kind
- No complex feature fusion, gating, or attention mechanisms
- No data pipeline changes
- No architectural changes to the model or heads

## Key Design Decisions (from route redesign)

### Target interpretation change

| Aspect | Old (CR-B) | New (CR-C) |
|--------|-----------|-----------|
| What support_gt=0.8 means | "0.017m from a fitted boundary primitive" | "Confidently near a semantic class transition" |
| What the head learns | Metric distance in 3D space | Boundary proximity confidence |
| Loss function | SmoothL1 + Tversky (regression) | Confidence-weighted BCE (classification) |
| Feature demand on backbone | Geometric localization features | Class-transition-detection features |
| Alignment with semantic task | Low | High |

### Why this should work

The mechanism analysis (`docs/canonical/clean_reset_mechanism_analysis.md`) establishes that:

1. Binary edge classification (DLA-Net style) produces gradient aligned with semantic discrimination
2. The author's PTv3 reproduction confirms even a simple two-head setup works with aligned targets
3. The current continuous Gaussian regression produces competing gradient (44% of gradient signal by epoch 100)
4. The confidence-weighted BCE preserves the DLA-Net alignment while adding graduated certainty from the support Gaussian

### Auxiliary weight

Set `w_aux` to 0.2-0.5. The auxiliary task should be a mild regularizer, not a competing primary objective. Monitor gradient share during training — if it exceeds 20% of total by epoch 50, the weight is too high.

## Implementation Scope

### Loss function changes

The existing `RedesignedSupportFocusLoss` needs to be replaced (or a new loss class created) that:
- Computes BCE(support_pred, valid) weighted by `where(valid, support, 1.0)`
- Adds this as `w_aux * loss_aux` to the semantic CE loss
- Does NOT compute SmoothL1 on support values
- Does NOT compute Tversky coverage
- The focus term (Lovasz on boundary region) is NOT included in Part 1 — this is pure implicit coupling via aligned gradient

### Config changes

- New config file: `configs/semantic_boundary/clean_reset_s38873367/cr_c_proximity_cue.py` (or similar)
- Same as CR-B config except loss class and auxiliary weight

### Evaluation

- Same evaluator as CR-B (boundary metrics still useful for tracking)
- Primary metric: val_mIoU (compared to CR-A = 0.7336)
- Secondary: per-class IoU breakdown, convergence dynamics, auxiliary gradient share

## Success Criterion

**CR-C mIoU >= CR-A (0.7336)**

Any positive delta over CR-A confirms that boundary-aware auxiliary supervision can help semantics when the target is properly aligned. This validates the route redesign and opens Part 2.

## Failure Diagnosis

If CR-C also fails (mIoU < CR-A), the problem is deeper than target design. Possible next investigations:
- Gradient-isolated support (`feat.detach()` before support head) — tests whether ANY auxiliary gradient at the backbone is harmful
- Architecture: multi-scale support head attachment
- The valid + support representation itself may need rethinking

## Requirements Traceability

| Requirement | Description |
|-------------|-------------|
| CUE-01 | Implement confidence-weighted BCE loss |
| CUE-02 | Create CR-C training config |
| CUE-03 | Run CR-C 100 epochs, produce structured comparison |
| CUE-04 | CR-C mIoU >= CR-A (0.7336) — success gate for Part 2 |
