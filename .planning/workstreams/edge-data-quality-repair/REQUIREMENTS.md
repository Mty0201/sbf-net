# Requirements: Edge Data Pipeline Refactor and Quality Repair v1.0

## Baseline Diagnosis (Phase 1 — complete)

- [x] **DEN-01**: Diagnose whether NET-01 coverage loss primarily originates in Stage 2 (DBSCAN cluster loss in sparse regions), Stage 4 (fixed sigma=0.04 Gaussian decay), or both — **Complete:** Stage 2 primary (18.7pp survival gap), Stage 4 secondary (1.3-7.0pp valid gap)

## Part A: Algorithm-Preserving Refactor

### Behavioral Audit and Module Restructure (Phase 2)

- [x] **REF-01**: Restructure pipeline stages into modular, independently runnable units with clear input/output contracts
- [x] **REF-02**: Audit and document all hidden compatibility logic, heuristics, and cross-stage behavioral contracts — classify each as core algorithm, compatibility patch, or data-adaptation logic
- [x] **REF-03**: Separate core algorithm logic from compatibility/adaptation logic at module boundaries, so each can evolve independently

### Config Injection, Validation, and Equivalence (Phase 3)

- [x] **REF-04**: Implement per-stage configuration system that supports density-adaptive parameter injection points (values remain at current defaults) — **Complete:** 4 frozen dataclasses (Stage1-4Config) in core/config.py, all 5 scripts integrated
- [x] **REF-05**: Add intermediate validation hooks so each stage's output can be verified before feeding the next — **Complete:** 4 validate_* functions in core/validation.py covering 7 cross-stage contracts, all 5 scripts integrated
- [x] **REF-06**: Verify behavioral equivalence — refactored pipeline produces identical output for identical inputs under default parameters. Cover both final output and intermediate invariants, including compatibility-sensitive edge cases on real project data — **Complete:** 9-test equivalence gate (np.array_equal), all 4 stages + in-memory path, human-approved

## Part B: Algorithm Improvement

### Algorithm Evolution (Phase 4)

- [x] **ALG-01**: Redesign Stage 2 output contract — every cluster must satisfy direction-consistency (single tangent equivalence class) and spatial-continuity (no along-axis gaps, bounded lateral spread). Enforce via new validation hook. Move `group_tangents()` and `split_direction_group_into_runs()` from trigger_regroup.py into Stage 2 post-DBSCAN processing. — **Complete:** validate_cluster_contract checks H1/H2/H3, 7 contract invariant tests pass
- [x] **ALG-02**: Make Stage 2 DBSCAN density-aware — adaptive eps/min_samples based on local center density, so sparse-region clusters survive instead of being discarded as noise. Eliminate trigger judgment, subgroup classification, and fragment/main merging from Stage 3 (~500 lines, 25 params deleted). — **Complete:** rescue_noise_centers, trigger path eliminated (780 lines), 5 rescue unit tests pass
- [x] **ALG-03**: Verify against Phase 1 baselines — sparse-region cluster survival gap reduced from 18.7pp to <10pp, dense-region quality maintained, snake supports (path A + B) eliminated. Regenerate equivalence baseline for intentional behavior change. — **Complete:** reference_v2/ baseline generated, 6 Phase 4 equivalence tests pass, Stage 1 exact-match preserved

## Quality Repair

### NET-01: Density-Adaptive Supervision (Phase 5)

- [ ] **DEN-02**: Implement density-adaptive fix so sparse-to-dense valid edge coverage gap is significantly reduced to a pre-defined acceptable threshold (e.g. gap < 5%)
- [ ] **DEN-03**: Verify fix does not degrade dense-region edge quality (coverage rate, weight distribution)

### NET-02: Single-Side Boundary Recovery (Phase 6)

- [ ] **SSB-01**: Diagnose which boundary types and semantic classes are most affected by single-side detection failure (020101/020102 as initial analysis scenes)
- [ ] **SSB-02**: Implement single-side boundary recovery mechanism so boundary positions previously missing due to single-side semantics produce valid boundary centers
- [ ] **SSB-03**: Implement support gap repair mechanism so support element discontinuities caused by single-side boundaries are resolved
- [ ] **SSB-04**: Verify recovered boundaries have valid support elements and direction fields

### NET-03: Support Element Smoothing (Phase 7)

- [ ] **SNK-01**: Diagnose the primary cause of snake/zigzag (DBSCAN boundary irregularity, fitting sample distribution imbalance, inter-support junction discontinuity, or combination)
- [ ] **SNK-02**: Implement snake/zigzag fix mechanism targeting the diagnosed primary cause, suppressing high-frequency direction jitter while preserving genuine curvature
- [ ] **SNK-03**: Verify snake score improvement (target: snake>0.005m reduced from 22% to <10%) and direction coherence improvement

### Cross-Cutting (Phase 8)

- [ ] **VER-01**: Re-generate edge data for affected scenes, with 020101/020102 as first-batch verification scenes
- [ ] **VER-02**: Produce data quality verification report comparing before/after metrics (coverage rate, continuity, snake score, direction coherence)
- [ ] **VER-03**: Document formal diagnosis conclusions and fix rationale as workstream context for downstream training/recon consumers

## Future Requirements

_(none deferred)_

## Out of Scope

- SBF-Net retraining with repaired data — separate milestone decision
- Recon-side fixes (RECON R-01 through R-05) — separate concern
- New edge.npy format changes beyond what fixes require
- Scenes beyond 020101/020102 for initial verification (may extend later)

## Traceability

| REQ-ID | Phase | Plan | Status |
|--------|-------|------|--------|
| DEN-01 | 1 | 01-01, 01-02 | complete |
| REF-01 | 2 | 02-01, 02-02, 02-03 | complete |
| REF-02 | 2 | 02-01, 02-03 | complete |
| REF-03 | 2 | 02-02, 02-03 | complete |
| REF-04 | 3 | 03-01 | complete |
| REF-05 | 3 | 03-02 | complete |
| REF-06 | 3 | 03-01, 03-03 | complete |
| ALG-01 | 4 | 04-01, 04-03 | complete |
| ALG-02 | 4 | 04-01, 04-02, 04-03 | complete |
| ALG-03 | 4 | 04-03 | complete |
| DEN-02 | 5 | — | pending |
| DEN-03 | 5 | — | pending |
| SSB-01 | 6 | — | pending |
| SSB-02 | 6 | — | pending |
| SSB-03 | 6 | — | pending |
| SSB-04 | 6 | — | pending |
| SNK-01 | 7 | — | pending |
| SNK-02 | 7 | — | pending |
| SNK-03 | 7 | — | pending |
| VER-01 | 8 | — | pending |
| VER-02 | 8 | — | pending |
| VER-03 | 8 | — | pending |
