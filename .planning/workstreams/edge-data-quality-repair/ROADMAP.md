# Roadmap: edge-data-quality-repair

## v1.0 — Edge Data Quality Repair

**Goal:** Fix edge data quality issues in data_pre/bf_edge_v3 — restore supervision coverage, continuity, and geometric fidelity; deliver verified repaired data generation pipeline.

### Phase 1: NET-01 diagnosis

**Goal:** Determine whether sparse-region coverage loss originates in Stage 2 (DBSCAN cluster loss), Stage 4 (sigma decay), or both.
**Requires:** DEN-01
**Depends on:** Nothing — first phase
**Plans:** 2 plans

Plans:
- [ ] 01-01-PLAN.md — Regenerate intermediates and run stratified density-bucketed diagnosis
- [ ] 01-02-PLAN.md — Synthesize formal diagnosis conclusion with primary/secondary ranking

### Phase 2: NET-01 fix and verify

**Goal:** Implement density-adaptive fix at identified bottleneck(s), verify coverage improvement without dense-region regression.
**Requires:** DEN-02, DEN-03
**Depends on:** Phase 1

### Phase 3: NET-02 diagnosis fix and verify

**Goal:** Diagnose single-side boundary failure patterns, implement boundary recovery and support gap repair, verify.
**Requires:** SSB-01, SSB-02, SSB-03, SSB-04
**Depends on:** Phase 2

### Phase 4: NET-03 diagnosis fix and verify

**Goal:** Diagnose snake/zigzag primary cause, implement targeted fix, verify direction quality improvement.
**Requires:** SNK-01, SNK-02, SNK-03
**Depends on:** Phase 2

### Phase 5: Re-generation and final verification

**Goal:** Apply all fixes to re-generate edge data, produce comprehensive quality report, document conclusions.
**Requires:** VER-01, VER-02, VER-03
**Depends on:** Phases 2, 3, 4
