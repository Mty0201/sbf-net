# Clean-Reset Experiment Analysis: CR-A vs CR-B

**Date:** 2026-04-07
**Runs compared:**
- CR-A: semantic-only (SemanticOnlyLoss) -- 100 epochs, seed=38873367
- CR-B: support-only (RedesignedSupportFocusLoss, focus_mode="none") -- 100 epochs, seed=38873367

**Evidence boundary:** Both runs use the same seed, same data pipeline, same backbone (PT-v3m1-0-base), same optimizer (AdamW), same batch size (effective 24), same scheduler (OneCycleLR), and same eval frequency (every 100 training epochs = 1 eval epoch). The only difference is the loss function and model head: CR-A has a semantic-only head with CE loss; CR-B has semantic + support heads with RedesignedSupportFocusLoss (SmoothL1 + Tversky for support, focus_mode="none" so no focus weighting). Both ran for exactly 100 eval epochs. This is a controlled comparison with no confounding asymmetry in training duration.

**Methodology:** Follows the comparison framework established in v1.1 Phase 9 (`.planning/milestones/v1.1-phases/09-phase-7-full-training-results-analysis-and-tuning/09-ANALYSIS.md`), adapted only for naming (CR-A/CR-B instead of active route/baseline) and the symmetric training duration.

## Executive Summary

CR-A (semantic-only) outperforms CR-B (support-only) by +1.52 percentage points in best val_mIoU (0.7336 vs 0.7184) and by +1.19pp in steady-state mean (0.7215 vs 0.7096 over last 10 epochs). CR-A wins 6 of 8 classes at steady state and shows lower variance. Under controlled conditions with identical seed and training duration, adding support supervision does not improve semantic segmentation -- it degrades it. The support loss consumes 40-44% of gradient signal while providing no measurable benefit to the semantic objective.

## 1. Overall mIoU Comparison

| Metric | CR-A (Semantic-Only) | CR-B (Support-Only) | Delta |
|--------|:-------------------:|:-------------------:|:-----:|
| Best val_mIoU | 0.7336 | 0.7184 | **+0.0152** |
| Best val_mAcc | 0.8425 | 0.8426 | -0.0001 |
| Best val_allAcc | 0.8692 | 0.8605 | +0.0087 |
| Epoch of best mIoU | 70 | 57 | - |
| Total eval epochs | 100 | 100 | - |

CR-A's best mIoU (0.7336) exceeds CR-B's best (0.7184) by 1.52pp. Unlike the v1.1 Phase 9 comparison, there is no training duration asymmetry here -- both ran for exactly 100 eval epochs with the same seed, making this a clean head-to-head comparison.

## 2. Per-Class mIoU Breakdown

### 2a. At Best-mIoU Epoch (Respective Peaks)

| Class | CR-A (ep70) | CR-B (ep57) | Delta | Winner |
|-------|:-----------:|:-----------:|:-----:|:------:|
| balustrade | 0.8354 | 0.8207 | +0.0147 | CR-A |
| balcony | 0.3994 | 0.0800 | +0.3194 | CR-A* |
| advboard | 0.8087 | 0.7769 | +0.0318 | CR-A |
| wall | 0.7610 | 0.7404 | +0.0206 | CR-A |
| eave | 0.6143 | 0.7013 | -0.0870 | CR-B |
| column | 0.8403 | 0.8061 | +0.0342 | CR-A |
| window | 0.7735 | 0.7829 | -0.0094 | CR-B |
| clutter | 0.6282 | 0.6196 | +0.0086 | CR-A |

*Note on balcony at CR-B epoch 57: The 0.0800 value is a per-epoch dip, not representative of CR-B's balcony capability. CR-B's balcony fluctuates between 0.08 and 0.50 throughout training. Per-class IoU at single epochs is noisy for small/rare classes. The steady-state comparison (Section 2b) is more reliable for per-class judgments.

### 2b. At Epoch 100 (Steady-State Comparison)

This comparison uses the final epoch for both runs, avoiding the per-epoch noise that makes peak-epoch per-class comparisons unreliable:

| Class | CR-A (ep100) | CR-B (ep100) | Delta | Winner |
|-------|:------------:|:------------:|:-----:|:------:|
| balustrade | 0.8147 | 0.8711 | -0.0564 | CR-B |
| balcony | 0.5315 | 0.4320 | +0.0995 | CR-A |
| advboard | 0.8074 | 0.7873 | +0.0201 | CR-A |
| wall | 0.7518 | 0.7438 | +0.0080 | CR-A |
| eave | 0.5800 | 0.5471 | +0.0329 | CR-A |
| column | 0.8416 | 0.8276 | +0.0140 | CR-A |
| window | 0.7770 | 0.7861 | -0.0091 | CR-B |
| clutter | 0.6411 | 0.6357 | +0.0054 | CR-A |

**CR-A wins 6 of 8 classes.** CR-B wins only balustrade and window.

**Analysis:**

- **Balustrade** is the only class where CR-B shows a meaningful advantage (+5.64pp). This is also the class where CR-B consistently excels late in training (0.87+ from epoch 80 onward), suggesting support supervision specifically helps this class's large, boundary-adjacent surfaces. However, this single-class advantage does not compensate for CR-A's broader lead.
- **Balcony** is the class with the largest CR-A advantage (+9.95pp). Balcony has the lowest absolute IoU for both runs and the highest epoch-to-epoch variance, making it the most difficult class. CR-A's cleaner gradient signal appears to help here.
- **Eave** now favors CR-A (+3.29pp at epoch 100), reversing the peak-epoch comparison. This confirms that the peak-epoch eave advantage for CR-B (0.7013 at ep57) was a transient spike, not a stable advantage.
- **Window** is a marginal CR-B win (-0.91pp), within noise.

## 3. Boundary-Region Metrics (CR-B Only)

CR-B's evaluator reports boundary-region metrics that CR-A does not have. These metrics characterize how well CR-B handles the boundary supervision task:

### Boundary Metrics at Key Epochs

| Epoch | val_boundary_mIoU | val_boundary_mAcc | boundary_point_ratio | support_reg_error | support_cover | valid_ratio |
|------:|:-----------------:|:-----------------:|:--------------------:|:-----------------:|:-------------:|:-----------:|
| 1 | 0.3465 | 0.6791 | 0.1461 | 0.0389 | 0.3700 | 0.1635 |
| 10 | 0.4066 | 0.6292 | 0.1469 | 0.0438 | 0.5782 | 0.1636 |
| 20 | 0.5451 | 0.6719 | 0.1462 | 0.0431 | 0.6039 | 0.1635 |
| 40 | 0.5180 | 0.6394 | 0.1464 | 0.0454 | 0.6127 | 0.1633 |
| 57 | 0.5540 | 0.6707 | 0.1460 | 0.0431 | 0.6282 | 0.1633 |
| 80 | 0.5559 | 0.6659 | 0.1459 | 0.0436 | 0.6365 | 0.1631 |
| 100 | 0.5420 | 0.6537 | 0.1458 | 0.0463 | 0.6272 | 0.1628 |

**Key observations:**

- **val_boundary_mIoU** plateaus around 0.54-0.56 from epoch 40 onward, well below the overall mIoU of 0.71. Boundary regions remain substantially harder than the scene average.
- **support_cover** (Tversky overlap between predicted and GT support field) reaches ~0.63 by epoch 20 and plateaus. The support head learns the boundary field reasonably well but this knowledge does not translate into better semantic performance.
- **support_reg_error** (SmoothL1 regression error) stays flat at ~0.04 throughout, indicating the support regression task converges early and provides diminishing gradient signal for the rest of training.
- **boundary_point_ratio** is constant at ~0.146, confirming the boundary region definition is stable.

The critical finding: CR-B's support head does learn to predict the boundary field (support_cover = 0.63), yet this learned boundary information produces worse semantic segmentation than CR-A's pure semantic approach. The auxiliary task succeeds on its own terms but fails to help the primary objective.

## 4. Training Dynamics and Convergence

### Loss Curves

**CR-A (semantic-only) loss at key epochs:**

| Epoch | Total Loss | Semantic Loss | Semantic % |
|------:|:----------:|:-------------:|:----------:|
| 1 | 1.3842 | 0.8528 | 61.6% |
| 10 | 0.4442 | 0.2495 | 56.2% |
| 20 | 0.3240 | 0.1791 | 55.3% |
| 40 | 0.2424 | 0.1323 | 54.6% |
| 70 | 0.1706 | 0.0920 | 53.9% |
| 80 | 0.1569 | 0.0842 | 53.7% |
| 100 | 0.1467 | 0.0789 | 53.8% |

Note: Total loss includes regularization/weight decay terms beyond the semantic CE, hence semantic% < 100%.

**CR-B (support-only) loss at key epochs:**

| Epoch | Total Loss | Semantic | Semantic % | Support | Support % | Sup Reg | Sup Cover |
|------:|:----------:|:--------:|:----------:|:-------:|:---------:|:-------:|:---------:|
| 1 | 1.5668 | 1.3813 | 88.2% | 0.1855 | 11.8% | 0.0374 | 0.7407 |
| 10 | 0.5839 | 0.4350 | 74.5% | 0.1490 | 25.5% | 0.0447 | 0.5216 |
| 20 | 0.4504 | 0.3137 | 69.6% | 0.1367 | 30.4% | 0.0436 | 0.4657 |
| 40 | 0.3505 | 0.2252 | 64.3% | 0.1254 | 35.7% | 0.0417 | 0.4182 |
| 57 | 0.3027 | 0.1835 | 60.6% | 0.1192 | 39.4% | 0.0403 | 0.3945 |
| 80 | 0.2666 | 0.1534 | 57.5% | 0.1132 | 42.5% | 0.0384 | 0.3738 |
| 100 | 0.2553 | 0.1440 | 56.4% | 0.1113 | 43.6% | 0.0378 | 0.3675 |

**Critical finding:** The support loss in CR-B consumes an increasing share of gradient signal (12% at epoch 1 -> 44% at epoch 100). By the time semantic loss has converged, support loss accounts for nearly half of total gradient. This is a milder version of the loss dominance problem identified in the v1.1 Phase 9 analysis (where the old active route had 70% support dominance), but the effect is the same: the backbone receives gradient signal split between semantic classification and support regression, producing worse semantic features than CR-A's pure semantic training.

The RedesignedSupportFocusLoss (SmoothL1 + Tversky) produces lower absolute support loss than the old BCE-based loss (~0.11 vs ~0.64), but the relative share still grows because the semantic loss decreases faster.

### Convergence Analysis

| Metric | CR-A | CR-B |
|--------|:----:|:----:|
| 90% of best first reached | Epoch 8 (0.6617) | Epoch 5 (0.6572) |
| Last-10 epochs mean mIoU | 0.7215 | 0.7096 |
| Last-10 epochs stdev mIoU | 0.0030 | 0.0039 |
| Best mIoU | 0.7336 (epoch 70) | 0.7184 (epoch 57) |
| Best-to-final gap | 0.0121 | 0.0088 |
| Top-5 epoch range | ep 61-87 | ep 57-97 |

Both runs converge fast (90% of best by epoch 5-8). CR-A has a higher steady-state mean (+1.19pp), lower variance, and a higher peak. The top-5 epochs for CR-A (0.7274-0.7336) are all above CR-B's best (0.7184).

Unlike the v1.1 Phase 9 comparison, there is no training duration confound. Both ran 100 epochs. The gap is structural, not a duration artifact.

## 5. Problem Identification

### Problem 1: Support Supervision Degrades Semantic Performance

The central finding of this experiment. Under controlled conditions (identical seed, duration, data, backbone, optimizer), adding support supervision to the training objective produces worse semantic segmentation than training with semantic loss alone.

**Data evidence:**
- Best mIoU: CR-A 0.7336 vs CR-B 0.7184 (-1.52pp)
- Steady-state mean: CR-A 0.7215 vs CR-B 0.7096 (-1.19pp)
- CR-A wins 6 of 8 classes at epoch 100

**Mechanism:** The support loss accounts for 40-44% of gradient signal in CR-B. This diverts backbone capacity toward predicting a continuous boundary field (support_cover reaches 0.63) rather than optimizing for semantic classification. The support head successfully learns its task but this learning competes with the semantic objective rather than complementing it.

### Problem 2: Balcony Instability in CR-B

CR-B's balcony IoU is highly unstable, ranging from 0.04 to 0.50 across epochs, while CR-A's balcony is also variable but stabilizes higher (0.40-0.53 in the final 30 epochs). The support loss appears to particularly destabilize the hardest class.

**Data evidence:** CR-B balcony at epoch 57 (best mIoU): 0.0800. At epoch 100: 0.4320. CR-A at epoch 100: 0.5315 (+9.95pp).

### Problem 3: Diminishing Returns from Support Regression

The support regression task (SmoothL1 loss component) converges by epoch 10-20 (support_reg_error stable at ~0.04) but continues to consume gradient. The Tversky coverage metric also plateaus (0.63 by epoch 20). After early training, the support supervision provides gradient signal that is neither improving support prediction nor helping semantics.

**Data evidence:** support_reg at epoch 10: 0.0447. At epoch 100: 0.0378. Minimal change over 90 epochs.

## 6. Verdict and Decision Implications

### Verdict

**Support supervision does not help semantic segmentation under the current implementation.** CR-A (semantic-only) is the better model across all aggregate metrics and 6 of 8 classes. The result is clean: same seed, same duration, same everything except the loss function.

This result invalidates the assumption that boundary-aware support supervision -- even in the redesigned SmoothL1+Tversky form -- improves semantic quality. The support head learns to predict boundaries but this auxiliary task competes with rather than complementing the semantic objective.

### Comparison to v1.1 Phase 9 Findings

The v1.1 Phase 9 analysis identified support loss dominance (70% of gradient) as the critical problem in the old active route. The redesigned loss (RedesignedSupportFocusLoss) reduced this to ~44%, but the fundamental issue persists: any support supervision diverts gradient from the semantic objective. The v1.1 tuning recommendations (Variants A-D) proposed weight reduction and focus amplification, but this clean-reset experiment suggests the problem is more fundamental -- even at the redesigned loss's lower weight, support supervision is a net negative.

### Decision Implications for v2.0

Based on Phase 9 judgment criteria adapted for v2.0:

1. **Does support beat semantic-only?** No. CR-A > CR-B by 1.52pp (best) and 1.19pp (steady-state).
2. **Does support improve boundary classes?** Mixed but net negative. Balustrade favors CR-B (+5.64pp at ep100), but balcony, eave, advboard, column all favor CR-A. The boundary-adjacent classes do not consistently benefit from boundary supervision.
3. **Is the cost acceptable?** No. 6/8 class regressions, higher training variance, lower peak and steady-state performance.

**Status:** These results demonstrate that the **current implementation** of support supervision is a net negative. However, the underlying concept of semantic-edge dual supervision remains plausible based on prior art (DLA-Net) and the author's own successful PTv3 reproduction. The negative result is treated as an implementation/coupling/optimization failure, not as evidence that the research direction is wrong. See `docs/canonical/clean_reset_diagnostic.md` for the full diagnostic analysis and ranked failure hypotheses.

### What This Means for Next Steps

Pure semantic training produces better semantic results than the current support-supervision implementation. The boundary field is learnable (support_cover = 0.63) but the current integration design -- raw additive multi-task loss with no gradient management, no coupling mechanism (focus_mode="none"), and no feature fusion -- produces gradient competition rather than gradient synergy. The diagnostic identifies concrete implementation failure modes and proposes targeted experiments to isolate which failures are responsible.

## 7. Data Sources

- CR-A training log: `outputs/clean_reset_s38873367/semantic_only/train.log` (19MB, 120,965 lines)
- CR-B training log: `outputs/clean_reset_s38873367/support_only/train.log` (30MB, 120,949 lines)
- CR-A parsed CSVs: `outputs/clean_reset_s38873367/semantic_only/{metrics_epoch,per_class_iou}.csv`
- CR-B parsed CSVs: `outputs/clean_reset_s38873367/support_only/{metrics_epoch,per_class_iou}.csv`
- Log parser: `scripts/analysis/parse_train_log.py` (extended with `semantic_only` run type for this analysis)
- Comparison methodology: v1.1 Phase 9 (`.planning/milestones/v1.1-phases/09-phase-7-full-training-results-analysis-and-tuning/09-ANALYSIS.md`)
- Experiment configs: `configs/semantic_boundary/clean_reset_s38873367/`
