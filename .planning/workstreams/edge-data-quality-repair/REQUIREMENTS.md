# Requirements: Edge Data Quality Repair v1.0

## NET-01: Density-Adaptive Supervision

- [x] **DEN-01**: Diagnose whether NET-01 coverage loss primarily originates in Stage 2 (DBSCAN cluster loss in sparse regions), Stage 4 (fixed sigma=0.04 Gaussian decay), or both — **Complete:** Stage 2 primary (18.7pp survival gap), Stage 4 secondary (1.3-7.0pp valid gap)
- [ ] **DEN-02**: Implement density-adaptive fix so sparse-to-dense valid edge coverage gap is significantly reduced to a pre-defined acceptable threshold (e.g. gap < 5%)
- [ ] **DEN-03**: Verify fix does not degrade dense-region edge quality (coverage rate, weight distribution)

## NET-02: Single-Side Boundary Recovery

- [ ] **SSB-01**: Diagnose which boundary types and semantic classes are most affected by single-side detection failure (020101/020102 as initial analysis scenes)
- [ ] **SSB-02**: Implement single-side boundary recovery mechanism so boundary positions previously missing due to single-side semantics produce valid boundary centers
- [ ] **SSB-03**: Implement support gap repair mechanism so support element discontinuities caused by single-side boundaries are resolved
- [ ] **SSB-04**: Verify recovered boundaries have valid support elements and direction fields

## NET-03: Support Element Smoothing

- [ ] **SNK-01**: Diagnose the primary cause of snake/zigzag (DBSCAN boundary irregularity, fitting sample distribution imbalance, inter-support junction discontinuity, or combination)
- [ ] **SNK-02**: Implement snake/zigzag fix mechanism targeting the diagnosed primary cause, suppressing high-frequency direction jitter while preserving genuine curvature
- [ ] **SNK-03**: Verify snake score improvement (target: snake>0.005m reduced from 22% to <10%) and direction coherence improvement

## Cross-Cutting

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
| DEN-02 | 2 | — | pending |
| DEN-03 | 2 | — | pending |
| SSB-01 | 3 | — | pending |
| SSB-02 | 3 | — | pending |
| SSB-03 | 3 | — | pending |
| SSB-04 | 3 | — | pending |
| SNK-01 | 4 | — | pending |
| SNK-02 | 4 | — | pending |
| SNK-03 | 4 | — | pending |
| VER-01 | 5 | — | pending |
| VER-02 | 5 | — | pending |
| VER-03 | 5 | — | pending |
