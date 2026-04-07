# Requirements: sbf-net

**Defined:** 2026-04-06
**Milestone:** v2.0 — Semantic-first boundary supervision reboot
**Core Value:** Semantic segmentation remains the primary objective, and any boundary-aware supervision must improve boundary-region semantic quality without dragging the semantic branch into explicit geometric-field learning.

## v1 Requirements

Requirements for milestone `v2.0 semantic-first boundary supervision reboot`. Each maps to one or more roadmap phases.

### Integration Repair

- [ ] **FIX-01**: The CR-B (support-only) config's evaluator kwargs crash (INT-01) is fixed and the pipeline runs without error.
- [ ] **FIX-02**: The CR-B config's missing `edge` key in the data pipeline (INT-02) is fixed and the full forward+loss+backward path succeeds.
- [ ] **FIX-03**: The CR-B config's `edge` field is correctly reindexed through GridSample/SphereCrop (INT-03) so that `edge.shape[0] == segment.shape[0]` in every batch.

### Clean-Reset Experiment Execution

- [ ] **EXP-01**: CR-A (semantic-only, seed=38873367) training completes for 100 epochs with results logged to `outputs/clean_reset_s38873367/semantic_only/`.
- [ ] **EXP-02**: CR-B (support-only, seed=38873367) training completes for 100 epochs with results logged to `outputs/clean_reset_s38873367/support_only/`.
- [ ] **EXP-03**: Both experiments use identical seed, data pipeline, backbone, and optimizer settings — the only variable is the presence/absence of support supervision.

### Evidence Analysis

- [ ] **ANA-01**: A structured comparison of CR-A vs CR-B results exists, covering at minimum: final mIoU, boundary-region mIoU, convergence curve shape, and per-class breakdown.
- [ ] **ANA-02**: The comparison includes a clear verdict on whether support supervision has a statistically meaningful positive effect on semantic performance under controlled conditions.

### Direction Decision

- [x] **DIR-01**: Route redesign established based on diagnostic + mechanism analysis. Support reinterpreted as boundary proximity cue (confidence-weighted BCE), not geometric regression. Part 1/Part 2 staged route defined. Canonical docs: `docs/canonical/clean_reset_diagnostic.md`, `docs/canonical/clean_reset_mechanism_analysis.md`, `docs/canonical/route_redesign_discussion.md`.

### Part 1: Boundary Proximity Cue Validation (Phase 5)

- [ ] **CUE-01**: Implement a new loss function that treats `valid` as a binary boundary/not-boundary label and `support` as a confidence weight for boundary points, using confidence-weighted BCE classification (Option L1 from route redesign). No continuous Gaussian regression.
- [ ] **CUE-02**: Create a CR-C training config identical to CR-B except for the loss function change (CUE-01) and auxiliary weight set to 0.2–0.5.
- [ ] **CUE-03**: Run CR-C for 100 epochs with seed=38873367 under the same controlled conditions as CR-A/CR-B, and produce a structured CR-C vs CR-A comparison.
- [ ] **CUE-04**: CR-C matches or exceeds CR-A semantic-only mIoU (0.7336). This is the success gate for Part 2.

### Part 2: Serial Derivation of Boundary Offset Field (Phase 6, conditional on CUE-04)

- [ ] **SER-01**: Implement module g — a learnable KNN-based local aggregation module that takes `softmax(seg_logits)` (N,8) and `coord` (N,3) as input, performs KNN neighbor lookup, computes edge features `[feat_i, feat_j - feat_i, coord_j - coord_i]`, aggregates via shared MLP + max-pool, and outputs a per-point 3D offset vector.
- [ ] **SER-02**: Implement offset supervision — smooth-L1 loss between predicted offset and `dir_gt × dist_gt` (from edge ground truth columns 0:3 and 3), applied only to valid points (edge[:,5]=1).
- [ ] **SER-03**: Integrate g into the existing model: backbone → semantic_head → seg_logits → g → offset. Support head remains separate (CR-C route). Gradients from offset loss flow through g back into backbone via seg_logits.
- [ ] **SER-04**: Create CR-D training config: CR-C + module g + offset supervision. All other settings (seed, epochs, backbone, optimizer) identical to CR-A/CR-B/CR-C.
- [ ] **SER-05**: Verify CR-D semantic mIoU does not regress below CR-C result, and offset predictions produce valid boundary projections (qualitative spot-check on sample scenes).

### Guardrails

- [ ] **GUARD-01**: All experiment configs remain within the `sbf-net` extension boundary — no Pointcept-side changes.
- [ ] **GUARD-02**: No experiment results are overstated — local smoke checks are not presented as full-train validation.
- [ ] **GUARD-03**: Canonical docs are updated to reflect the clean-reset findings and any direction changes.

## v2 Requirements (Deferred)

Carried forward from v1.1 where still relevant:

- **FOLL-03**: The repo introduces a cleaner route-schema contract so model, loss, evaluator, and trainer semantics cannot drift silently across future supervision routes.
- **FOLL-04**: The repo adds stronger automated regression coverage for route contracts and smoke validation.
- **FOLL-05**: The repo evaluates broader architectural adaptations if the semantic-first supervision signal still fails to protect semantic performance adequately.
