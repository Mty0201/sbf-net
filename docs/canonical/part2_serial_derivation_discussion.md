# Part 2 Architecture Discussion: Serial Derivation of Edge Field from Semantic Predictions

**Date:** 2026-04-07
**Status:** Design discussion, not implementation spec.
**Prerequisite:** Read `route_redesign_discussion.md` (Part 1 context) and `clean_reset_mechanism_analysis.md` (why parallel prediction fails).

## 1. The Insight

The edge ground truth is derived deterministically from semantic ground truth:

```
data_pre pipeline: semantic_gt + coord → boundary_centers → DBSCAN → line_fit → [dir, dist, support, valid]
```

This means `edge_gt = f(semantic_gt, coord)` — no additional information source needed.

If this is true for ground truth, then at prediction time: instead of predicting edge fields in parallel from backbone features (which causes gradient competition), we can derive them serially from semantic predictions via a learnable module g that approximates f.

## 2. Architecture Comparison

### Old route (parallel prediction — fails):
```
backbone_feat → semantic_head → seg_logits     (needs class-discriminative features)
backbone_feat → edge_head → dir/dist/support   (needs geometric-localization features)
                                                → GRADIENT COMPETITION
```

### New route (serial derivation):
```
backbone_feat → semantic_head → seg_logits → g(softmax(seg_logits), coord) → edge_pred
                                              ↑ learnable module
                                              supervised by edge_gt
```

Key difference: backbone only produces semantic features. The geometric field is derived FROM the semantic output, not predicted in parallel. Gradients from edge supervision flow through g back to backbone, but the signal they carry is "improve your semantic predictions at boundaries" — fully aligned with the semantic task.

## 3. Why This Eliminates Feature Competition

In parallel prediction, backbone must encode two types of information simultaneously:
1. "What class is this point?" → class-discriminative features
2. "How far to nearest boundary? What direction?" → geometric-localization features

These compete for backbone capacity.

In serial derivation, backbone only encodes type 1. Type 2 is computed by g from the semantic output. The gradient from edge loss through g tells the backbone: "your semantic prediction near this boundary is wrong, fix it." This is the SAME signal as semantic CE/Lovasz — just amplified at boundary regions.

## 4. Design Questions for g

### 4.1 Input representation

g's input must be differentiable (so gradients flow back to backbone):
- `softmax(seg_logits)` (N, 8) — class probabilities per point ✓
- `coord` (N, 3) — 3D coordinates ✓
- NOT `argmax(seg_logits)` — hard labels, non-differentiable, would block gradients ✗

Concatenated input: (N, 11) per point.

### 4.2 Local neighborhood requirement

Edge direction and distance are inherently local properties — they depend on how semantic labels change in a spatial neighborhood, not on a single point's class probability. A single-point MLP cannot predict boundary direction because it has no information about neighboring points' semantics.

g MUST have access to local neighborhood structure. Options:
- **KNN on coord** — query K nearest neighbors, aggregate their (softmax, coord) features
- **Ball query** — similar but with fixed radius
- **Reuse backbone's neighbor structure** — PTv3 already computes serialized neighborhoods; g could reuse these indices
- **Mini-PointNet on local patches** — shared MLP + max-pool over KNN neighbors
- **Local cross-attention** — each point attends to its K neighbors' semantic probabilities

The simplest viable option: KNN neighbors → concatenate/subtract neighbor features → shared MLP → per-point edge prediction. This is essentially "compute local semantic gradients in a learnable way."

### 4.3 Output

g produces per-point: `[dir_pred(3), dist_pred(1), support_pred(1)]`
- dir: unit direction vector toward nearest boundary (supervised by edge[:, 0:3])
- dist: distance to nearest boundary (supervised by edge[:, 3])
- support: boundary proximity confidence (supervised by edge[:, 4])

Only points where valid=1 (edge[:, 5]) receive supervision for dir and dist. Support/proximity cue supervision can cover all points (as in CR-C).

### 4.4 Relationship with CR-C (Part 1)

CR-C's boundary proximity cue (confidence-weighted BCE) could be:
- **Retained as a parallel regularizer** — backbone still gets the boundary-detection signal from CR-C, while g handles the full geometric field
- **Subsumed into g** — g's support output replaces the CR-C auxiliary head entirely
- **Used as g's gating signal** — g only activates for points where the proximity cue predicts "near boundary"

Decision deferred to implementation phase.

### 4.5 Inference behavior

At inference: backbone → seg_logits → g(softmax(seg_logits), coord) → edge_pred

Fully end-to-end. No data_pre pipeline needed. The edge field quality depends on semantic prediction quality — errors in semantic predictions propagate to edge predictions. This is actually a feature: the edge field is self-consistent with the segmentation, not an independent geometric estimate.

## 5. Literature Basis

The serial derivation pattern has validated components but the full combination is novel:

| Component | Validated by |
|-----------|-------------|
| Gradients from boundary loss through semantic predictions to backbone | Gated-SCNN DTR (ICCV 2019), CPG Loss (2024), InverseForm (CVPR 2021) |
| Soft semantic predictions as input to downstream modules | OCRNet (ECCV 2020), SoftGroup (CVPR 2022) |
| Learnable boundary processing on point clouds | JSENet (ECCV 2020), BA-GEM (AAAI 2021) |
| Full combination: learnable g(semantic_logits, coord) → geometric edge field | **Novel — literature gap** |

The closest existing work is Gated-SCNN's Dual Task Regularizer, which uses a fixed Sobel filter to extract boundaries from 2D semantic probability maps. Our proposal replaces the fixed Sobel with a learnable module operating on 3D point clouds — a necessary generalization because:
1. 3D point clouds have no regular grid (Sobel undefined)
2. We need direction + distance (Sobel only gives binary edges)
3. The data_pre pipeline's mapping is more complex than Sobel (DBSCAN + line fitting + Gaussian field)

## 6. Risks

1. **g's capacity vs f's complexity**: data_pre is a multi-stage pipeline with non-differentiable operations (DBSCAN, line fitting). g must approximate this entire chain. If g is too shallow, it won't capture the geometric structure. If too deep, it may overfit or slow training.

2. **Noisy semantic predictions early in training**: In early epochs, seg_logits are nearly random. g receives garbage input and produces garbage edge predictions. The edge loss will backpropagate large gradients through garbage, potentially destabilizing training. May need warmup (activate edge supervision only after N epochs of semantic-only training).

3. **Propagation of semantic errors**: At inference, semantic mispredictions near boundaries produce incorrect edge fields. The edge field is only as good as the segmentation — there's no independent geometric "anchor."

4. **KNN computation cost**: If g uses KNN on coord for local aggregation, this adds O(N log N) per forward pass. May be negligible relative to PTv3 backbone cost, but needs measurement.

## 7. Next Steps

1. Confirm CR-C Part 1 results (user reports it's already effective — formal write-up pending)
2. Design g's architecture in detail: input format, neighborhood aggregation, MLP structure, output heads
3. Design training schedule: warmup strategy for edge supervision, loss weighting
4. Implement and smoke-validate
5. Run experiment comparing: CR-A (semantic only) vs CR-C (proximity cue) vs CR-D (serial derivation)
