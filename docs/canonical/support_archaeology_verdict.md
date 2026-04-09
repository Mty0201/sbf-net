# Support Supervision Archaeology: Final Verdict

**Date:** 2026-04-08
**Scope:** Trace every support supervision variant in this repo's history, determine if any was reliably effective.

## Answer: No reliably effective support design has ever existed in this codebase.

## Evidence

### The 0.7457 Claim (train_good.log)

The highest support-only mIoU on record (0.7457) came from `outputs/semantic_boundary_support_only_train/train_good.log`. Investigation reveals:

- **3 independent runs** concatenated into one log file, each `Start epoch: 1`, each with `seed=None`
- Code version: commit `82b0381` (pre-GSD)
- Config: `SemanticBoundaryLoss` with `support_weight=1.0, support_cover_weight=0.25, dir_weight=0, dist_weight=0`
- Effective batch size: 24 (batch=4, accum=6)
- Each run: 100 eval epochs

**Per-run results:**

| Run | Date | Seed | Best mIoU |
|:--:|:--|:--:|:--:|
| 1 | 2026-03-25 08:44 | random | 0.7346 |
| 2 | 2026-03-25 19:27 | random | 0.7344 |
| 3 | 2026-03-26 09:25 | random | **0.7457** |

Run 1 and Run 2 are virtually identical to the semantic-only baseline range (0.7336-0.7395). Run 3 is a lucky seed outlier (+1pp above the other two).

### Controlled v2.0 Experiments (seed=38873367)

| Experiment | Loss | Best mIoU | Delta vs CR-A |
|:--|:--|:--:|:--:|
| CR-A (semantic only) | CE+Lovasz | 0.7336 | baseline |
| CR-B (support regression) | CE+Lovasz + SmoothL1+Tversky | 0.7184 | **-1.52pp** |
| CR-C old (BCE, buggy weight) | CE+Lovasz + weighted BCE | 0.7257 | -0.79pp |
| CR-C (BCE, fixed weight) | CE+Lovasz + weighted BCE | 0.7323 | -0.13pp |

### Old Uncontrolled Experiments (seed=3407)

| Experiment | Best mIoU | Delta vs semantic-only (0.7395) |
|:--|:--:|:--:|
| semantic-only | 0.7395 | baseline |
| variant_c (SmoothL1, no focus) | 0.7406 | +0.11pp (noise) |
| variant_a2 (SmoothL1 + focus) | 0.7394 | -0.01pp (noise) |

## Loss Design at commit 82b0381 (the "best" version)

```python
# SemanticBoundaryLoss, support-only config (dir_weight=0, dist_weight=0):
loss_semantic = CE(seg_logits, segment) + Lovasz(seg_logits, segment)
loss_support_reg = SmoothL1(sigmoid(pred), support_gt * valid_gt)  # weighted by valid_gt
loss_support_cover = Tversky(sigmoid(pred), valid_gt, alpha=0.3, beta=0.7)
loss_support = 1.0 * loss_support_reg + 0.25 * loss_support_cover
total = loss_semantic + loss_support
```

Model: `SharedBackboneSemanticBoundaryModel` with `EdgeHead` (stem=Linear+ReLU, 3 sub-heads for dir/dist/support, but dir/dist get zero gradient). Backbone feat shared directly to both heads, no adapter.

## What Changed Between "Good" and "Bad" Versions

Between train_good (commit 82b0381) and train_bad (commit ce10572), the **loss code for the support-only path was functionally identical**. Diff only added:
- `support_weighted_edge` option (default False, not activated)
- `**_extra` kwargs passthrough
- Removed legacy aliases

The config changed `support_cover_weight` from 0.25 to 0.20 — but this is a minor tweak.

**The performance gap (0.7457 vs 0.7255) is entirely explained by seed variance.** Run 1 and 2 of train_good (0.7346, 0.7344) are consistent with train_bad (0.7255) within the ~1pp noise band observed across all experiments.

## Conclusion

1. **No support design has reliably beaten semantic-only.** Best case is neutral (within ~0.1-0.2pp).
2. **The 0.7457 was a seed outlier**, not evidence of an effective design.
3. **The BCE redesign (CR-C) eliminated harm** from support supervision (from -1.5pp to -0.1pp), but didn't produce positive signal.
4. **Regression support (SmoothL1+Tversky) is harmful** under controlled conditions (-1.5pp).
5. All "positive" old-line results dissolve under controlled comparison or multi-seed averaging.

## Implications

The research question "does boundary-aware auxiliary supervision help semantic segmentation?" remains unanswered by this codebase's experiments. What IS answered: the specific implementations tried here (regression, BCE, with/without focus, with/without coupling) do not help. Further work in this direction requires a fundamentally different approach (e.g., the serial derivation module g, or explicit gradient management), not parameter tuning of existing designs.
