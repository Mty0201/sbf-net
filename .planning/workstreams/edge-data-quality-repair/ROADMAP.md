# Roadmap: edge-data-quality-repair

## v1.0 — Edge Data Pipeline Refactor and Quality Repair

**Goal:** Refactor `data_pre/bf_edge_v3` into a clean, instrumentable, verifiable pipeline structure, then fix edge data quality issues (NET-01, NET-02, NET-03) on top of the refactored foundation. Deliver repaired data with coverage, continuity, and geometric fidelity improvements.

**Approach:** Refactor first, repair second. The existing Phase 1 baseline diagnosis provides evidence input for both the refactoring and the subsequent repair phases.

### Phase 1: NET-01 baseline diagnosis ✅

**Goal:** Establish baseline diagnosis of sparse-region coverage loss — determine whether the bottleneck is in Stage 2 (DBSCAN cluster loss), Stage 4 (sigma decay), or both.
**Requires:** DEN-01
**Depends on:** Nothing — first phase
**Status:** Complete (2026-04-06). Primary: Stage 2 (18.7pp survival gap). Secondary: Stage 4 (1.3-7.0pp valid gap).
**Role in new milestone:** Baseline evidence input for refactoring design and post-refactor repair.
**Plans:** 2 plans

Plans:
- [x] 01-01-PLAN.md — Regenerate intermediates and run stratified density-bucketed diagnosis
- [x] 01-02-PLAN.md — Synthesize formal diagnosis conclusion with primary/secondary ranking

### Phase 2: data_pre pipeline refactor

**Goal:** Restructure `data_pre/bf_edge_v3` into a clean, modular, instrumentable pipeline that supports per-stage configuration, intermediate validation, and density-adaptive parameter injection without entangling structural concerns with quality fixes.
**Requires:** REF-01, REF-02, REF-03, REF-04
**Depends on:** Phase 1 (diagnosis informs which extension points the refactored structure must support)

### Phase 3: NET-01 fix and verify (on refactored pipeline)

**Goal:** Implement density-adaptive fix at identified bottleneck(s), verify coverage improvement without dense-region regression.
**Requires:** DEN-02, DEN-03
**Depends on:** Phase 2

### Phase 4: NET-02 diagnosis fix and verify

**Goal:** Diagnose single-side boundary failure patterns, implement boundary recovery and support gap repair, verify.
**Requires:** SSB-01, SSB-02, SSB-03, SSB-04
**Depends on:** Phase 3

### Phase 5: NET-03 diagnosis fix and verify

**Goal:** Diagnose snake/zigzag primary cause, implement targeted fix, verify direction quality improvement.
**Requires:** SNK-01, SNK-02, SNK-03
**Depends on:** Phase 3

### Phase 6: Re-generation and final verification

**Goal:** Apply all fixes to re-generate edge data, produce comprehensive quality report, document conclusions.
**Requires:** VER-01, VER-02, VER-03
**Depends on:** Phases 3, 4, 5
