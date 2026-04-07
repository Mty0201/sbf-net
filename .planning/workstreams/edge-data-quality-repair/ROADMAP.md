# Roadmap: edge-data-quality-repair

## v1.0 — Edge Data Pipeline Refactor and Quality Repair

**Goal:** Refactor `data_pre/bf_edge_v3` into a clean, instrumentable, verifiable pipeline structure, then improve the algorithm and fix edge data quality issues (NET-01, NET-02, NET-03) on top of the refactored foundation. Deliver repaired data with coverage, continuity, and geometric fidelity improvements.

**Approach:** Two-part refactor then repair. Part A (Phases 2-3) makes the current algorithm explicit, modular, and stable without changing behavior. Part B (Phase 4) iterates on the algorithm itself — density-aware parameters, improved splitting/fitting, intentional semantic changes treated as redesign. Quality repair phases (5-8) build on both.

**A/B boundary rule:** Any change that alters current default output semantics belongs in Part B or later, not Part A. Part A may only restructure, document, and instrument — never silently change what the algorithm produces.

### Phase 1: NET-01 baseline diagnosis ✅

**Goal:** Establish baseline diagnosis of sparse-region coverage loss — determine whether the bottleneck is in Stage 2 (DBSCAN cluster loss), Stage 4 (sigma decay), or both.
**Requires:** DEN-01
**Depends on:** Nothing — first phase
**Status:** Complete (2026-04-06). Primary: Stage 2 (18.7pp survival gap). Secondary: Stage 4 (1.3-7.0pp valid gap).
**Role in milestone:** Baseline evidence input for refactoring design and post-refactor repair.
**Plans:** 2 plans

Plans:
- [x] 01-01-PLAN.md — Regenerate intermediates and run stratified density-bucketed diagnosis
- [x] 01-02-PLAN.md — Synthesize formal diagnosis conclusion with primary/secondary ranking

---

### — Part A: Algorithm-Preserving Refactor (Phases 2–3) —

*Goal: Make the current algorithm explicit, modular, configurable, and verifiable — without changing what it produces.*

### Phase 2: Behavioral audit and module restructure ✅

**Goal:** Audit the current pipeline to surface hidden compatibility logic, heuristics, and cross-stage behavioral contracts. Restructure into modular, independently runnable stages with clear I/O contracts, explicit behavioral documentation, and separation of core algorithm from compatibility/adaptation logic.
**Requires:** REF-01, REF-02, REF-03
**Depends on:** Phase 1 (diagnosis informs which extension points the refactored structure must support)
**Canonical refs:** `data_pre/bf_edge_v3/REFACTOR_TARGET.md`
**Plans:** 3/3 plans executed
**Status:** Complete (2026-04-07). Pipeline audited, restructured, parameters centralized, docs updated.

Plans:
- [x] 02-01-PLAN.md — Behavioral audit: per-module classification docs and cross-stage contracts
- [x] 02-02-PLAN.md — Module restructure: decompose supports_core.py into fitting/trigger_regroup/export sub-modules
- [x] 02-03-PLAN.md — Parameter extraction to params.py, behavioral spot-check, pipeline docs update

### Phase 3: Config injection, validation hooks, and equivalence gate ✅

**Goal:** Implement per-stage configuration system with density-adaptive parameter injection points, add intermediate validation hooks, and verify behavioral equivalence — refactored pipeline produces identical output under default parameters.
**Requires:** REF-04, REF-05, REF-06
**Depends on:** Phase 2
**Status:** Complete (2026-04-07). 4 frozen config dataclasses, 4 validation hooks, 31 tests (9 equivalence, bit-identical). Human-approved.
**Plans:** 3 plans

Plans:
- [x] 03-01-PLAN.md — Reference data generation, test infrastructure, config dataclasses, script integration (REF-04)
- [x] 03-02-PLAN.md — Validation hooks implementation and script integration (REF-05)
- [x] 03-03-PLAN.md — Equivalence gate pytest suite with human verification (REF-06)

---

### — Part B: Algorithm Improvement (Phase 4) —

*Goal: Iterate on the algorithm itself — intentional semantic changes treated explicitly as redesign, not folded into refactor.*

### Phase 4: Algorithm evolution — density-aware parameters and improved splitting/fitting

**Goal:** Introduce density-adaptive behavior at identified bottleneck stages (Stage 2 eps, Stage 4 sigma), improve clustering/splitting/fitting logic, and redesign compatibility strategies where the current heuristics are inadequate. Every behavioral change is explicitly marked as algorithm redesign.
**Requires:** ALG-01, ALG-02, ALG-03
**Depends on:** Phase 3 (equivalence gate must pass before algorithm changes begin)

---

### — Quality Repair (Phases 5–8) —

*Goal: Fix NET-01/02/03 on top of the refactored and improved pipeline.*

### Phase 5: NET-01 fix and verify (on improved pipeline)

**Goal:** Apply density-adaptive fix at identified bottleneck(s), verify coverage improvement without dense-region regression.
**Requires:** DEN-02, DEN-03
**Depends on:** Phase 4

### Phase 6: NET-02 diagnosis fix and verify

**Goal:** Diagnose single-side boundary failure patterns, implement boundary recovery and support gap repair, verify.
**Requires:** SSB-01, SSB-02, SSB-03, SSB-04
**Depends on:** Phase 5

### Phase 7: NET-03 diagnosis fix and verify

**Goal:** Diagnose snake/zigzag primary cause, implement targeted fix, verify direction quality improvement.
**Requires:** SNK-01, SNK-02, SNK-03
**Depends on:** Phase 5

### Phase 8: Re-generation and final verification

**Goal:** Apply all fixes to re-generate edge data, produce comprehensive quality report, document conclusions.
**Requires:** VER-01, VER-02, VER-03
**Depends on:** Phases 5, 6, 7
