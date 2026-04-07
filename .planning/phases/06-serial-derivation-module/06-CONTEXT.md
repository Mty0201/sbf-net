# Phase 6 Context: Serial Derivation Module g

## Background

The parallel dual-branch approach to boundary field prediction failed because backbone features must simultaneously encode class-discriminative information (for semantics) and geometric-localization information (for boundary direction/distance). This gradient competition degrades semantic performance.

The serial derivation approach eliminates this competition: backbone only learns semantic features. A learnable module g derives the geometric boundary field FROM the semantic predictions, not in parallel from backbone features. Edge supervision gradients through g reinforce "improve semantic predictions at boundaries" — fully aligned with the semantic task.

## Architecture Decision (from discussion 2026-04-07)

```
backbone → semantic_head → seg_logits          (unchanged from CR-C)
backbone → support_head → support (BCE)         (unchanged from CR-C, separate branch)
           g(softmax(seg_logits), coord) → offset (3D, smooth-L1)   (NEW)
```

### Module g design:
- **Input**: softmax(seg_logits) (N,8) + coord (N,3) → feat (N,11)
- **Neighborhood**: KNN on coord, K neighbors per point (not reusing PTv3 Z-curve which has spatial jumps)
- **Edge features**: For each point i and neighbor j: `[feat_i, feat_j - feat_i, coord_j - coord_i]` → dim 11+11+3=25
- **Aggregation**: shared MLP (25→64→64) + max-pool over K neighbors → (N,64)
- **Output**: MLP (64→32→3) → offset prediction
- **Support**: remains a separate head from backbone (CR-C route), NOT inside g

### Edge representation (novel — literature gap):
- **offset (3D)**: displacement vector from point to nearest boundary. `coord + offset` = projected boundary position
- **Supervised by**: smooth-L1 vs `dir_gt × dist_gt` (columns 0:3 and 3 of edge.npy)
- **Only valid points** (edge[:,5]=1) receive offset supervision
- Replaces previous (dir, dist, support) 5-dim representation with (offset, support) 4-dim

### Training:
- No extra warmup — g shares learning rate group with semantic head
- Existing cosine annealing + LR grouping provides natural warmup
- Early epochs: seg_logits near random, g learns nothing useful, but low LR prevents harm
- As semantic predictions improve, g naturally begins learning meaningful offset mapping

## Existing Code Structure

### Models:
- `project/models/semantic_support_model.py` — `SharedBackboneSemanticSupportModel` (CR-C model, has backbone + semantic_head + support_head)
- `project/models/heads.py` — `SemanticHead`, `SupportHead`, `EdgeHead`, `ResidualFeatureAdapter`
- `project/models/__init__.py` — registry exports

### Losses:
- `project/losses/boundary_proximity_cue_loss.py` — `BoundaryProximityCueLoss` (CR-C loss, has semantic CE+Lovasz + confidence-weighted BCE)
- `project/losses/__init__.py` — `build_loss()` factory

### Trainer:
- `project/trainer/trainer.py` — `_build_loss_inputs()` routes model outputs to loss kwargs
  - Priority 1: `edge_pred` in output → full boundary loss path
  - Priority 2: `support_pred` in output → support-only loss path
  - Need to add: Priority for new model that outputs both `support_pred` AND `offset_pred`

### Config:
- `configs/semantic_boundary/clean_reset_s38873367/proximity_cue_train.py` — CR-C config (base for CR-D)
- `configs/semantic_boundary/clean_reset_s38873367/clean_reset_support_model.py` — model config
- `configs/semantic_boundary/clean_reset_s38873367/clean_reset_data.py` — data config (already has edge in pipeline)

### Edge ground truth format:
- `edge.npy` shape (N, 6): dir_x(0), dir_y(1), dir_z(2), dist(3), support(4), valid(5)
- offset_gt = edge[:, 0:3] * edge[:, 3:4]  (dir × dist = displacement vector)

## Key Constraints

1. **coord must be accessible in the model's forward()** — currently model receives `input_dict` with coord, but only passes it to backbone. g needs coord after backbone processing (same N as seg_logits).
2. **KNN computation** — need a KNN implementation. Options: torch-cluster, pytorch3d, or manual. Check what's available in the environment.
3. **Gradient flow** — softmax(seg_logits) is the input to g, so offset loss gradients flow through softmax back to seg_logits back to semantic_head back to backbone. This is the intended behavior.
4. **Trainer routing** — `_build_loss_inputs()` needs to handle the new output key `offset_pred` and pass `coord` to loss.
5. **Loss needs coord** — to compute offset_gt = dir_gt × dist_gt from edge tensor, loss doesn't need coord. But if we ever want to verify `coord + offset_pred ≈ boundary_position`, coord would be useful for monitoring.
