# Validation And Experiment Handoff

## Purpose

This document records the Phase 8 local validation results for the support-guided semantic focus route and defines the next full-train experiment directions. It is the handoff artifact between the local implementation milestone (v1.1) and the first full-training experiment.

## Evidence Boundary

**Local validation proves:** the active route runs correctly — model forward produces `seg_logits` and `support_pred` with correct shapes, all three loss terms (`loss_semantic`, `loss_support`, `loss_focus`) are non-NaN, gradients flow via backward + optimizer step, and focus weighting activates in boundary regions (where `support_gt > 0.2`, mean `focus_weight` exceeds non-boundary regions).

**Local validation does NOT prove:** that the route improves semantic segmentation performance, that it beats the support-only baseline, or that it generalizes across the full dataset. Those claims require full-train experiments in a separate environment.

Validation script: `scripts/train/check_active_route_train_step.py`
Smoke config: `configs/semantic_boundary/old/semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train-smoke.py`

## Smoke Validation Results

| Check | Status |
|-------|--------|
| Output keys (`seg_logits`, `support_pred`) | PASS |
| Non-NaN losses (semantic, support, focus) | PASS |
| Backward + optimizer step | PASS |
| Focus activation (boundary > non-boundary) | PASS |

**Overall:** ALL_PASS

Results source: Plan 08-01 created the smoke config and validation script. Run `check_active_route_train_step.py` with `POINTCEPT_ROOT` set to reproduce.

## Comparison Baseline

The comparison target for the next full-train experiment is:

- **Route:** support-only (reg=1, cover=0.2)
- **Metric:** val_mIoU = 74.6
- **Source:** `docs/canonical/sbf_semantic_first_route.md`

The support-guided semantic focus route must beat this baseline on val_mIoU to justify its additional complexity (focus weighting term + boundary-region semantic emphasis).

## Next Full-Train Experiment Directions

### Direction 1: Soft Masking Ablation (Top Priority)

**Question:** Does replacing the hard `valid_gt` binary mask in the support loss with a continuous `support_gt`-derived soft weight improve support prediction quality and downstream focus accuracy?

**Current implementation:** `loss_support` uses `valid_gt` as a hard binary mask — points with `valid_gt = 0` contribute zero gradient to support prediction. This is safe but potentially wasteful: points just outside the valid region still carry partial boundary proximity signal in `support_gt`.

**Ablation formula:**
```
loss_support_soft = BCE(support_logit, support_gt) * support_gt.pow(alpha)
```
where `alpha` controls how aggressively the soft mask suppresses low-proximity points.

**Open questions:**
- What range of `alpha` values produces stable training? (start with `alpha in {0.5, 1.0, 2.0}`)
- Does removing the hard `valid_gt` mask cause support prediction over-activation (false positives in non-boundary regions)?
- Is the focus weighting quality (measured by `val_boundary_mIoU` improvement over baseline) sensitive to `alpha`?

### Direction 2: Negative Sample Calibration

**Question:** When the hard `valid_gt` mask is removed (soft masking), does the support head over-predict boundary proximity in non-boundary regions?

**Risk:** Without `valid_gt` masking, the support BCE loss sees all points. If `support_gt` is near-zero for most non-boundary points, the loss landscape may still push the model toward conservative predictions, but there is a risk that the implicit negative weighting is insufficient.

**Diagnostic to implement:**
- After N epochs of soft-mask training, measure `sigmoid(support_logit)` statistics in regions where `support_gt < 0.05` — if mean prediction > 0.3, over-activation is occurring.
- Compare support_cover metric (Tversky overlap) between hard-mask and soft-mask variants.

**Open questions:**
- Does adding an explicit negative margin (e.g., `support_gt.pow(alpha) + eps` to ensure non-zero gradient everywhere) help or hurt?
- Is a two-stage approach needed (train with hard mask first, then fine-tune with soft mask)?

### Direction 3: Alpha-Sigma Coupling (Record Only)

**Observation:** The `alpha` exponent in soft masking interacts with the `sigma` parameter used to generate `support_gt` from boundary distance during preprocessing. A large `sigma` produces a broad support field; a large `alpha` then concentrates the soft mask back toward high-proximity points. These two parameters may partially cancel each other if not co-tuned.

**Open questions:**
- What is the effective support radius (`sigma` effect) after applying `support_gt.pow(alpha)` soft masking?
- Should `alpha` and `sigma` be swept jointly (grid search) or is sequential tuning sufficient?
- Is there a closed-form relationship between `sigma` and `alpha` that simplifies the search space?

This direction is recorded for future investigation. Do not implement in the current experiment cycle.

### Direction 4: Adaptive Inference Route (Future)

**Question:** At inference time, can `support_pred` (the model's own boundary proximity prediction) replace the GT-derived `focus_weight` to provide adaptive boundary emphasis without ground-truth access?

**Concept:** During training, focus weighting uses GT `support_gt` (per design note in `support_guided_semantic_focus_loss.py`). At inference, GT is unavailable. If `support_pred` is accurate enough, `focus_weight_inference = 1.0 + lambda * sigmoid(support_pred).pow(gamma)` could provide boundary-aware post-processing or confidence weighting.

**Prerequisites:** This direction only makes sense after confirming that (a) the route beats the support-only baseline and (b) `support_pred` achieves reasonable boundary coverage (measured by `support_cover` metric during full training).

This direction is documented only. Do not implement in the current experiment cycle.

## Training Config Reference

Active route train config: `configs/semantic_boundary/old/semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train.py`
Active route smoke config: `configs/semantic_boundary/old/semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train-smoke.py`

Key hyperparameters for the first full-train run:
- `focus_lambda=1.0`, `focus_gamma=1.0` (baseline focus weighting)
- `support_loss_weight=1.0`, `focus_loss_weight=1.0`
- `seed=3407`, `total_epoch=2000`, `eval_epoch=100`
- Comparison metric: `val_mIoU` (checkpoint selected by best `val_mIoU`)

## Status

- **Phase 8 validation:** Local smoke complete (run `check_active_route_train_step.py` to reproduce)
- **Full-train experiment:** Not started — this document is the handoff
- **Next action:** Run the active route train config in a full-training environment and compare `val_mIoU` against the support-only baseline (74.6)
