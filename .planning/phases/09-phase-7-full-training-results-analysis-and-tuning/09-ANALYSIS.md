# Phase 9: Active Route Full Training Results Analysis

**Date:** 2026-04-03
**Runs compared:**
- Active route: support-guided semantic focus (focus_lambda=1, focus_gamma=1, all weights=1) -- 100 epochs evaluated
- Baseline: support-only (reg=1, cover=0.2, dir=0, dist=0) -- 300 epochs evaluated

**Evidence boundary:** This analysis covers one run of the baseline active route config only. No ablation variants have been tested yet. The active route was trained for 100 eval epochs (killed early relative to the 2000-epoch budget), while the support-only baseline ran for 300 eval epochs. Convergence and final-performance comparisons must account for this asymmetry.

## Executive Summary

The active route's best val_mIoU of 0.7265 (epoch 61) falls 1.92 percentage points short of the support-only baseline's 0.7457 (epoch 252). The active route shows a promising advantage on two boundary-sensitive classes (eave, balcony) but significant regressions on balustrade and advboard. The dominant problem is that the support loss consumes ~70% of total training loss, leaving the semantic and focus terms under-resourced. Tuning the loss balance is the highest-priority next step.

## 1. Overall mIoU Comparison

| Metric | Active Route | Support-Only Baseline | Delta |
|--------|-------------|----------------------|-------|
| Best val_mIoU | 0.7265 | 0.7457 | -0.0192 |
| Best val_mAcc | 0.8500 | 0.8568 | -0.0068 |
| Best val_allAcc | 0.8625 | 0.8776 | -0.0151 |
| Epoch of best mIoU | 61 | 252 | - |
| Total eval epochs | 100 | 300 | - |

Note: The support-only baseline best is 0.7457 from the log (not the rounded 74.6 in canonical docs). The active route was evaluated every epoch for 100 epochs; the support-only baseline was evaluated every epoch for 300 epochs.

## 2. Per-Class mIoU Breakdown

Per-class IoU at each run's respective best-mIoU epoch:

| Class | Active Route (ep61) | Support-Only (ep252) | Delta | Winner |
|-------|:-------------------:|:--------------------:|:-----:|:------:|
| balustrade | 0.7840 | 0.9106 | -0.1266 | Baseline |
| balcony | 0.3973 | 0.3315 | +0.0658 | Active |
| advboard | 0.7796 | 0.8557 | -0.0761 | Baseline |
| wall | 0.7413 | 0.7710 | -0.0297 | Baseline |
| eave | 0.7030 | 0.6931 | +0.0099 | Active |
| column | 0.8197 | 0.8144 | +0.0053 | Active |
| window | 0.7783 | 0.7896 | -0.0113 | Baseline |
| clutter | 0.6318 | 0.6258 | +0.0060 | Active |

**Winners (active route better):** balcony (+6.58pp), eave (+0.99pp), column (+0.53pp), clutter (+0.60pp)

**Losers (active route worse):** balustrade (-12.66pp), advboard (-7.61pp), wall (-2.97pp), window (-1.13pp)

**Analysis:** The active route's largest gain is on balcony, the class with the lowest absolute IoU in both runs (meaning it has the most room for improvement and is likely hard because it often appears near boundaries of other classes). The eave class, which tends to be geometrically thin and boundary-adjacent, also improves slightly.

However, the two largest regressions -- balustrade (-12.66pp) and advboard (-7.61pp) -- are both classes that the support-only baseline handles very well. Balustrade in particular drops from 0.91 to 0.78, suggesting the active route's additional loss terms may be interfering with the backbone features that the support-only route optimizes cleanly for these large, well-defined classes. The balustrade regression alone accounts for more than the entire overall mIoU gap.

A critical caveat: the active route was evaluated at only 100 epochs vs. 300 for the baseline. The support-only run's best at epoch 100 was approximately 0.71, comparable to the active route. The comparison at respective bests is valid, but the active route may not have fully converged. Late-epoch per-class IoU in the active route (epochs 90-100) shows balcony stabilizing around 0.47 and eave around 0.70-0.72, suggesting the gains on these classes are real, not noise.

## 3. Boundary-Region Metrics (Active Route Only)

These metrics are specific to the active route's evaluator and have no support-only counterpart:

### Boundary Metrics at Best Epoch (61)

| Metric | Value |
|--------|-------|
| val_boundary_mIoU | 0.5612 |
| val_boundary_mAcc | 0.6631 |
| boundary_point_ratio | 0.1474 |
| support_bce | 0.6533 |
| support_cover | 0.3752 |
| valid_ratio | 0.1642 |

### Boundary Metrics Trend

| Epoch | val_boundary_mIoU | val_boundary_mAcc | boundary_point_ratio | support_bce | support_cover |
|------:|:-----------------:|:-----------------:|:--------------------:|:-----------:|:-------------:|
| 1 | 0.4493 | 0.7114 | 0.1468 | 0.6774 | 0.3299 |
| 10 | 0.4453 | 0.5566 | 0.1461 | 0.6651 | 0.3585 |
| 20 | 0.5672 | 0.7043 | 0.1468 | 0.6563 | 0.3617 |
| 30 | 0.5502 | 0.6822 | 0.1464 | 0.6548 | 0.3657 |
| 40 | 0.5254 | 0.6388 | 0.1465 | 0.6558 | 0.3641 |
| 50 | 0.5270 | 0.6453 | 0.1457 | 0.6590 | 0.3725 |
| 61 | 0.5612 | 0.6631 | 0.1474 | 0.6533 | 0.3752 |
| 70 | 0.5799 | 0.6849 | 0.1469 | 0.6557 | 0.3812 |
| 80 | 0.5555 | 0.6746 | 0.1471 | 0.6664 | 0.3832 |
| 90 | 0.5608 | 0.6760 | 0.1468 | 0.6610 | 0.3874 |
| 100 | 0.5777 | 0.6885 | 0.1470 | 0.6599 | 0.3884 |

**Key observations:**

- **val_boundary_mIoU** improves from 0.45 to ~0.58 over 100 epochs, but remains well below the overall mIoU (0.73), indicating that boundary regions are genuinely harder for the model.
- **boundary_point_ratio** is essentially constant at ~0.147 throughout training, confirming that the boundary region definition (support_gt > 0.2) is stable and not an artifact of training dynamics.
- **support_bce** barely moves (0.677 to 0.660), indicating the support head is not learning effectively. The BCE loss on support prediction is barely improving over 100 epochs.
- **support_cover** improves slowly from 0.33 to 0.39 (Tversky overlap), meaning the support head's spatial coverage of the boundary region improves only marginally.

The near-flat support_bce trajectory is concerning: it suggests the support head gradient signal is either too weak (dominated by the support loss weight at 1.0 consuming 70% of total loss but not producing useful gradients) or the support task itself is too hard to learn given the current architecture (stem+linear with single output channel).

## 4. Training Dynamics and Convergence

### Loss Curves

**Active route loss breakdown at key epochs:**

| Epoch | Total Loss | Semantic | Support | Focus | Semantic % | Support % | Focus % |
|------:|:----------:|:--------:|:-------:|:-----:|:----------:|:---------:|:-------:|
| 1 | 2.9700 | 1.4003 | 0.6852 | 0.8845 | 47.1% | 23.1% | 29.8% |
| 10 | 1.3541 | 0.4230 | 0.6689 | 0.2623 | 31.2% | 49.4% | 19.4% |
| 20 | 1.1630 | 0.3073 | 0.6624 | 0.1933 | 26.4% | 57.0% | 16.6% |
| 40 | 1.0177 | 0.2207 | 0.6546 | 0.1423 | 21.7% | 64.3% | 14.0% |
| 61 | 0.9341 | 0.1722 | 0.6473 | 0.1145 | 18.4% | 69.3% | 12.3% |
| 80 | 0.8776 | 0.1409 | 0.6408 | 0.0958 | 16.1% | 73.0% | 10.9% |
| 100 | 0.8670 | 0.1359 | 0.6390 | 0.0922 | 15.7% | 73.7% | 10.6% |

**Support-only baseline loss at best epoch (252):**

| Total Loss | Semantic | Edge | Support | Support Reg | Support Cover | Dir | Dist |
|:----------:|:--------:|:----:|:-------:|:-----------:|:-------------:|:---:|:----:|
| 0.2956 | 0.1946 | 0.1010 | 0.1010 | 0.0367 | 0.4283 | 0.9928 | 0.3202 |

**Critical finding:** In the active route, the support loss dominates increasingly over training. By epoch 100, support accounts for 73.7% of total loss, while the semantic term (the primary objective) is only 15.7% and the focus term (the novel contribution) is only 10.6%. This means the model is overwhelmingly optimizing for support prediction rather than semantic segmentation.

The support-only baseline has a much healthier loss balance: total loss 0.2956 with semantic at 0.1946 (65.8%) and edge/support combined at 0.1010 (34.2%).

The support loss in the active route (BCE on support prediction) is essentially a floor at ~0.64 that barely decreases despite being the dominant loss term. This suggests the support head is not capable of reducing BCE much further given its architecture, and the resulting large, near-constant loss value crowds out the semantic and focus gradients.

### Convergence Analysis

| Metric | Active Route | Support-Only Baseline |
|--------|:-----------:|:--------------------:|
| 90% of best first reached | Epoch 10 (mIoU 0.6644) | Epoch 11 (mIoU 0.6819) |
| Last 10 epochs mean mIoU | 0.7128 | 0.7129 |
| Last 10 epochs stdev mIoU | 0.0050 | 0.0034 |
| Best mIoU | 0.7265 (epoch 61) | 0.7457 (epoch 252) |
| Best-to-final gap | 0.7265 - 0.7128 = 0.0137 | 0.7457 - 0.7129 = 0.0328 |

Both runs reach 90% of their respective best very early (~epoch 10-11), indicating the backbone converges quickly and fine-grained improvement is slow.

The active route's last-10-epoch mean (0.7128) is essentially identical to the baseline's (0.7129), but the baseline achieves a much higher peak (0.7457 vs. 0.7265). This suggests that:
1. Both models settle to a similar steady-state performance around 0.71-0.72.
2. The baseline's advantage comes from occasional high-quality checkpoints during its longer training run (300 vs 100 epochs).
3. The active route, if trained longer, might produce higher peaks but would need to address the loss balance problem to meaningfully surpass the baseline.

The active route's higher variance (0.0050 vs. 0.0034) is consistent with the less stable training from the dominant support loss.

### Loss Term Balance (Active Route)

At the best epoch (61):
- **Semantic loss:** 0.1722 (18.4% of total) -- this is the primary objective
- **Support loss:** 0.6473 (69.3% of total) -- binary cross-entropy on support prediction
- **Focus loss:** 0.1145 (12.3% of total) -- the semantic emphasis in boundary regions

The support loss dominates because BCE on a continuous prediction target (sigmoid output vs. continuous support_gt) produces inherently higher loss values than cross-entropy on an 8-class classification task where the model achieves ~86% accuracy. The equal weights (1.0 for all terms) are not calibrated for the different loss magnitude scales.

## 5. Problem Identification

### Problem 1: Support Loss Dominance (Critical)

The support BCE loss consumes ~70% of total gradient signal. The semantic CE and focus CE together receive only ~30%. Since the model's primary objective is semantic segmentation, this inversion is likely the single largest factor in the mIoU gap.

**Data evidence:** Support loss is 0.6473 at best epoch vs. semantic 0.1722 and focus 0.1145. Support loss decreases from 0.6852 to 0.6390 over 100 epochs (only a 6.7% relative reduction), confirming the support head architecture cannot efficiently reduce this loss.

**Impact:** The backbone receives gradient signal dominated by support prediction rather than semantic classification. This explains the balustrade and advboard regressions: classes with clear, large surfaces that benefit from clean semantic gradients are degraded when the gradient is polluted by support BCE.

### Problem 2: Support Head Stagnation

The support head (stem+linear, single channel) produces BCE loss that barely improves (~0.68 to ~0.64 over 100 epochs). The support_cover metric (Tversky overlap) only reaches 0.39, meaning the model's boundary region prediction covers less than 40% of the ground truth boundary field.

**Data evidence:** support_bce drops only 0.0175 absolute over 100 epochs. The support head may be under-parameterized for the continuous regression-like task it faces.

**Impact:** Poor support prediction means the focus weighting term (which uses GT support during training but would use predicted support at inference) has limited value at inference time.

### Problem 3: Under-Training

100 eval epochs is insufficient relative to the baseline's 300. The active route was still showing slow improvement at epoch 100 (boundary mIoU rising, loss decreasing). The support-only baseline's best came at epoch 252.

**Data evidence:** Active route last-10 mIoU mean (0.7128) is not clearly plateaued; the boundary metrics (val_boundary_mIoU 0.58 at epoch 100, up from 0.45 at epoch 1) are still trending upward.

**Impact:** The true performance gap may be smaller than 1.92pp if the active route is given equal training time.

### Problem 4: Focus Term Under-Contribution

The focus loss represents only 10-12% of total loss. Given that the focus mechanism is the novel contribution of the active route (boundary-region semantic emphasis), its signal is too weak to meaningfully steer learning.

**Data evidence:** Focus loss drops from 0.8845 (epoch 1) to 0.0922 (epoch 100), meaning the model quickly learns to classify boundary-region points well enough that the focus CE becomes small. The focus weight amplification (1 + lambda * support_gt^gamma with lambda=1, gamma=1) may be too mild to create meaningful additional emphasis.

## 6. Tuning Recommendations

Each recommendation is a concrete config variant proposal. No code changes are required -- only config parameter adjustments.

### Variant A: Support Loss Downweight

**Motivation:** The support loss dominates total loss at 70%+ (Section 4). Reducing its weight will rebalance gradients toward the semantic objective, which is the primary metric.

**Config changes:**
```python
# In semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train.py
loss = dict(
    type="SupportGuidedSemanticFocusLoss",
    support_loss_weight=0.1,   # was 1.0 -- reduce to make support ~20% of total
    focus_loss_weight=1.0,     # unchanged
    focus_lambda=1.0,          # unchanged
    focus_gamma=1.0,           # unchanged
)
```

**Expected effect:** With support_loss_weight=0.1, the support contribution at epoch 61 would be ~0.065 instead of 0.647. The new balance would be approximately: semantic 60%, focus 40%, support <3%. The backbone should allocate more capacity to semantic features, potentially recovering the balustrade and advboard regressions.

**Risk:** If support loss weight is too low, the support head may stop learning entirely, degrading the focus weighting mechanism.

### Variant B: Support Downweight + Focus Amplification

**Motivation:** Variant A addresses the support dominance but may weaken the focus mechanism (the novel contribution). This variant simultaneously increases focus_lambda to ensure boundary-region emphasis remains strong.

**Config changes:**
```python
# In semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train.py
loss = dict(
    type="SupportGuidedSemanticFocusLoss",
    support_loss_weight=0.1,   # was 1.0
    focus_loss_weight=2.0,     # was 1.0 -- double the focus term weight
    focus_lambda=2.0,          # was 1.0 -- stronger boundary emphasis
    focus_gamma=1.0,           # unchanged
)
```

**Expected effect:** The focus weighting formula becomes `1 + 2.0 * support_gt^1.0`, giving boundary-region points up to 3x the semantic CE weight (vs. 2x in the baseline config). Combined with the doubled focus_loss_weight, boundary-region semantic quality should be more aggressively optimized. This directly tests whether stronger boundary emphasis improves the per-class IoU on boundary-sensitive classes (balcony, eave) without degrading bulk classes.

**Connects to Phase 8 Direction 1 (Soft Masking):** This variant does not implement soft masking itself, but establishes whether the focus mechanism can drive boundary improvement when properly weighted, which is a prerequisite for the soft masking ablation.

### Variant C: Extended Training (Control)

**Motivation:** The active route was trained for only 100 eval epochs vs. 300 for the baseline. Before attributing the mIoU gap to the route design, it is necessary to control for training duration.

**Config changes:**
```python
# In semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train.py
trainer = dict(
    total_epoch=2000,    # unchanged (already set to 2000, was killed at 100)
    eval_epoch=100,      # unchanged
    num_workers=8,       # unchanged
    max_train_batches=None,
    max_val_batches=None,
)
# No parameter changes -- just let the existing config run to completion (or at least 300 epochs)
```

**Expected effect:** If the active route is still improving at epoch 100 (evidence: boundary metrics still trending up, loss still decreasing), running to 300+ epochs may close some or all of the 1.92pp gap. However, this is unlikely to fully close the gap if the loss balance problem remains.

**Recommendation:** Run this as a baseline comparison alongside Variant A, not as a standalone fix.

### Variant D: Soft Masking (Phase 8 Direction 1)

**Motivation:** The current support loss uses hard `valid_gt` masking. Replacing it with a continuous `support_gt.pow(alpha)` soft mask may improve support prediction quality in the boundary transition region, producing better focus weights. This directly tests Phase 8 experiment Direction 1.

**Config changes:**
```python
# In semseg-pt-v3m1-0-base-bf-support-guided-semantic-focus-train.py
loss = dict(
    type="SupportGuidedSemanticFocusLoss",
    support_loss_weight=0.1,   # combine with Variant A downweight
    focus_loss_weight=1.0,
    focus_lambda=1.0,
    focus_gamma=1.0,
    soft_mask_alpha=1.0,       # NEW: requires code addition to SupportGuidedSemanticFocusLoss
)
```

**Expected effect:** Points near the boundary edge (support_gt between 0.0 and 0.2) currently contribute zero gradient to support prediction due to the hard valid_gt mask. Soft masking gives them graduated gradient proportional to their proximity, potentially sharpening boundary detection. This is the top-priority experiment direction from the Phase 8 handoff.

**Note:** This variant requires a code change to `SupportGuidedSemanticFocusLoss` (adding the `soft_mask_alpha` parameter), so it is a tuning config variant + implementation variant, unlike A-C which are config-only.

## 7. Next Steps

**Prioritized variant execution order:**

1. **Variant A (Support Downweight)** -- Highest priority. Config-only change, directly addresses the critical loss balance problem (Problem 1). Quick to test, and its result informs whether the remaining variants are worth pursuing.

2. **Variant C (Extended Training)** -- Run in parallel with Variant A as a control. Uses the existing config unchanged; just allow the run to reach 300+ epochs. This separates the training duration effect from the loss balance effect.

3. **Variant B (Support Downweight + Focus Amplification)** -- Second priority. Tests whether the focus mechanism can drive boundary-class improvement when given more weight. Run this after Variant A results are known: if A already matches baseline, B tests whether we can exceed it on boundary classes.

4. **Variant D (Soft Masking)** -- Third priority. Requires a code change, so it is a larger investment. Run after Variants A/B establish whether the focus mechanism itself is beneficial, since soft masking is an improvement to the support prediction that feeds the focus mechanism.

**Phase 8 experiment direction mapping:**
- Variant D tests Direction 1 (Soft Masking Ablation)
- Variant B's focus_lambda tuning tests a prerequisite for Direction 3 (Alpha-Sigma Coupling)
- Direction 2 (Negative Sample Calibration) should be tested as a follow-up to Variant D
- Direction 4 (Adaptive Inference Route) is premature until a variant beats the support-only baseline

**Decision criteria for each variant:**
- **Beat baseline?** val_mIoU > 0.7457 (the support-only best)
- **Improve boundary classes?** balcony, eave per-class IoU improve vs. both baseline and current active route
- **No bulk class regression?** balustrade, advboard, wall per-class IoU does not regress more than 2pp vs. support-only baseline
